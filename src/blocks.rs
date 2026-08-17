//! Sub-function repeated-fragment detection.
//!
//! [`crate::similarity`] compares whole function bodies, so it structurally
//! cannot see a chunk of code copy-pasted *inside* two differently shaped
//! functions -- the surrounding statements differ, so the units never look
//! alike as wholes. This module finds that kind of duplication directly, by
//! hashing sliding windows of the normalized token stream instead of whole
//! units.
//!
//! # Identity survives motion, same as everywhere else in this crate
//!
//! [`crate::report`] explains why every finding in this crate carries a
//! content hash rather than a location: a location-keyed finding re-triggers
//! the moment anything above it in the file shifts line numbers, and a tool
//! that cries wolf on every unrelated edit gets muted. [`find_blocks`] hashes
//! the *extended* run -- not the seed window it was found from -- so the same
//! fragment, wherever it is embedded, always carries the same
//! [`ContentHash`].
//!
//! # Why seed windows are anchored before they are extended
//!
//! A naive version of this detector hashes every window of `min_tokens`
//! tokens, groups equal hashes, and for every pair extends forward while
//! tokens agree. That alone is wrong: a single 40-token repeat with
//! `min_tokens = 10` seeds a match at *every* interior offset of the run --
//! offset 0, 1, 2, ... -- because the window starting one token later still
//! matches too. Extending each of those forward independently reports up to
//! 31 overlapping findings for one copy-paste.
//!
//! The fix is to only extend a seed pair that is *left-maximal*: one where
//! the token immediately before the window differs between the two sides (or
//! the window starts at the very beginning of a stream). Every interior
//! offset of a real run fails that check, because the token before it is
//! part of the same run and therefore agrees on both sides -- so only the
//! run's true start ever gets extended, and extending forward from a
//! left-maximal start reaches the run's true end by construction. One
//! maximal run in, one finding out. Pinned by
//! `the_whole_shared_fragment_produces_one_finding_not_many_overlapping_ones`.
//!
//! # Why one occurrence anchors the rest, not a complete graph
//!
//! A fragment occurring N times in a codebase is not N *findings* worth of
//! news -- it is one fact ("this fragment is duplicated") with N locations.
//! Pairing every occurrence against every other one reports it
//! N * (N - 1) / 2 times: for a fragment repeated 71 times (real, seen
//! scanning a several-hundred-file repository) that is 2,485 findings
//! that all say the same thing, and it is quadratic in group size on top --
//! 9,553 fragment groups producing 67,818 pairs total on that same tree,
//! most of it repetition rather than information. So each occurrence group
//! sorts its members by `(file, start)` and pairs only the first against
//! every other one, for N - 1 findings instead of N * (N - 1) / 2. That
//! still keeps every occurrence in the output -- each one appears in exactly
//! the pair connecting it back to the group's earliest occurrence, so a
//! reader can still see every place the fragment lives and the delta engine
//! still gets one stable per-pair key per location -- it just stops
//! reporting the same fact once per *combination* of locations instead of
//! once per location. Sorting by `(file, start)` rather than, say, insertion
//! order matters for the same reason every other identity in this crate is
//! content- or position-derived rather than incidental: the delta engine
//! diffs two independent scans (branch and merge-base), and an anchor chosen
//! by an order that is not reproducible across those two scans would make an
//! unchanged fragment's anchor differ between them, which would misreport it
//! as new. The test-only reference implementation `naive_find_blocks` applies the identical rule
//! independently, and `a_fragment_repeated_n_times_yields_n_minus_one_findings_covering_every_occurrence`
//! pins both the count and the coverage.
//!
//! # Why windows are hashed with a rolling integer hash, not [`ContentHash`]
//!
//! [`ContentHash::of`] is [`blake3`] over every token *name* in the window,
//! recomputed from scratch. Calling it once per sliding window is O(n *
//! min_tokens) cryptographic hashing over a whole tree -- for a
//! several-hundred-file repository that is millions of windows, each paying
//! for a fresh hash of ~dozens of strings, and it dominates wall clock on
//! real input.
//!
//! `rolling_hashes` instead interns every token name to a `u32` once (see
//! [`crate::token::Interner`]) and hashes windows of `u32`s with a Rabin–Karp
//! polynomial rolling hash: `O(1)` work to slide the window one token,
//! `O(n)` total per file instead of `O(n * min_tokens)`. The tradeoff is that
//! a 64-bit integer hash *can* collide where a 128-bit cryptographic one
//! effectively never does in practice -- so every candidate pair the rolling
//! hash's buckets produce is re-confirmed by `windows_match`, a direct
//! comparison of the actual interned ids, before anything downstream ever
//! sees it. A collision can cost a wasted comparison; it can never produce or
//! drop a finding. `a_sixty_four_bit_hash_collision_is_rejected_by_the_confirmation_step`
//! and `a_sixty_four_bit_hash_collision_produces_no_finding_through_find_pairs`
//! construct a genuine collision (not a stand-in) and prove exactly that,
//! against the confirmation primitive and the real matching code in turn.
//! `the_rolling_hash_prefilter_matches_a_naive_all_windows_reference` proves
//! the fast path and a deliberately slow, obviously-correct
//! [`ContentHash`]-per-window implementation agree on a whole corpus.
//! [`ContentHash::of`] is still what [`find_blocks`] hashes the *emitted*
//! [`BlockPair`] with, unchanged -- that value is the cross-process identity
//! the delta engine depends on, and nothing about how a candidate was found
//! should be allowed to affect it.

use std::collections::HashMap;
use std::path::Path;

use tree_sitter::Parser;

use crate::extract::SourceFile;
use crate::normalize;
use crate::report::{BlockPair, BlockRef};
use crate::token::{ContentHash, Interner};

/// A normalized token plus the 1-based source line it came from.
///
/// Re-exported from [`crate::normalize`], which owns the traversal that
/// produces it — see that module's docs for why there is exactly one
/// descent implementation in this crate.
pub use crate::normalize::PlacedToken;

/// Tunables for [`find_blocks`].
#[derive(Debug, Clone)]
pub struct BlockOptions {
    /// Shortest repeated run worth reporting, in normalized tokens.
    pub min_tokens: usize,
}

/// Normalize a file into tokens that remember where they came from.
///
/// Parses the file, then delegates the actual walk to
/// [`crate::normalize::placed_tokens`] — the single traversal
/// implementation this crate has, so that a block finding and a
/// function-level clone finding describe the same notion of "duplicate" and
/// are never in tension with each other. A divergence here would make block
/// findings incomparable with function-level ones.
///
/// # Panics
/// If the parser returns no tree. That happens only when a parse is
/// cancelled or times out, neither of which is configured here -- the same
/// condition [`crate::extract::Extractor::extract`] panics on.
pub fn placed_tokens(file: &SourceFile) -> Vec<PlacedToken> {
    let mut parser = Parser::new();
    parser
        .set_language(&file.language.grammar())
        .expect("registered grammars are ABI-compatible; see lang::tests");
    let tree = parser.parse(&file.text, None).expect("no timeout or cancellation flag is set");
    normalize::placed_tokens(tree.root_node(), file.language)
}

/// Odd 64-bit multiplier for `rolling_hashes`'s polynomial rolling hash.
///
/// This is `round(2^64 / phi)`, the constant `splitmix64` and Fibonacci
/// hashing use for the same reason it works here: an odd multiplier is
/// invertible mod 2^64, which spreads nearby inputs across the whole 64-bit
/// range instead of clustering them. It has no cryptographic role -- every
/// candidate this hash groups together is re-confirmed by `windows_match`
/// before it is trusted, which is what makes an ordinary (fast, non-secure)
/// hash the right tool here at all.
const ROLLING_BASE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Rolling 64-bit hash of every `min_tokens`-token window in `ids`, indexed
/// by window start.
///
/// `O(n)`: the first window costs `O(min_tokens)`, and each later one is
/// derived from the previous in `O(1)` by subtracting the token that just
/// left and mixing in the one that just entered, standard Rabin–Karp. This
/// is the prefilter [`find_blocks`] buckets candidates with; see the module
/// docs for why a 64-bit hash here is safe despite not being cryptographic.
fn rolling_hashes(ids: &[u32], min_tokens: usize) -> Vec<u64> {
    if ids.len() < min_tokens {
        return Vec::new();
    }
    let mut leading_power = 1u64;
    for _ in 0..min_tokens.saturating_sub(1) {
        leading_power = leading_power.wrapping_mul(ROLLING_BASE);
    }

    let mut hash = 0u64;
    for &id in &ids[..min_tokens] {
        hash = hash.wrapping_mul(ROLLING_BASE).wrapping_add(u64::from(id));
    }

    let mut hashes = Vec::with_capacity(ids.len() - min_tokens + 1);
    hashes.push(hash);
    for start in 0..ids.len() - min_tokens {
        let leaving = u64::from(ids[start]);
        let entering = u64::from(ids[start + min_tokens]);
        hash = hash.wrapping_sub(leaving.wrapping_mul(leading_power));
        hash = hash.wrapping_mul(ROLLING_BASE).wrapping_add(entering);
        hashes.push(hash);
    }
    hashes
}

/// Confirm that two windows the rolling hash bucketed together really are
/// the same `min_tokens` ids, not a 64-bit collision.
fn windows_match(a: &[u32], sa: usize, b: &[u32], sb: usize, min_tokens: usize) -> bool {
    a[sa..sa + min_tokens] == b[sb..sb + min_tokens]
}

/// Find maximal runs of normalized tokens repeated in two places.
///
/// See the module docs for why seed windows are anchored before being
/// extended, and for why they are found via a rolling hash instead of
/// [`ContentHash`]. The steps:
///
/// 1. Normalize every file with [`placed_tokens`] and intern every token
///    name to a `u32` with a shared [`Interner`], so the rest of the work is
///    over integers rather than strings.
/// 2. Hash every window of exactly `min_tokens` consecutive ids with
///    `rolling_hashes` and group equal hashes.
/// 3. For each pair of occurrences in a group, confirm it with
///    `windows_match` (rejecting a rare 64-bit collision), skip it unless
///    it is left-maximal, then extend forward while the ids keep agreeing.
/// 4. Emit one [`BlockPair`] per surviving pair, hashing the *extended*
///    run's token *names* with [`ContentHash::of`] -- not the seed window,
///    and not the rolling hash -- so the fragment's identity is independent
///    of which offset it happened to be first noticed at, and is the same
///    stable, cross-process identity every other finding in this crate uses.
///
/// Deterministic: results are sorted before returning, independent of
/// [`HashMap`]'s randomized iteration order.
pub fn find_blocks(files: &[SourceFile], options: &BlockOptions) -> Vec<BlockPair> {
    let min_tokens = options.min_tokens;
    let streams: Vec<Vec<PlacedToken>> = files.iter().map(placed_tokens).collect();

    let mut interner = Interner::new();
    let ids: Vec<Vec<u32>> =
        streams.iter().map(|tokens| tokens.iter().map(|t| interner.intern(&t.name)).collect()).collect();

    let mut pairs = find_pairs(files, &streams, &ids, min_tokens);
    pairs.sort_by_key(|p| {
        (
            std::cmp::Reverse(p.tokens),
            p.hash.clone(),
            p.a.file.clone(),
            p.a.start_line,
            p.a.end_line,
            p.b.file.clone(),
            p.b.start_line,
            p.b.end_line,
        )
    });
    pairs
}

/// The candidate-matching core of [`find_blocks`], taking already-normalized
/// streams and already-interned ids rather than building them itself.
///
/// Split out so a genuine 64-bit rolling-hash collision -- astronomically
/// rare with real input, and only reachable by *constructing* colliding ids
/// directly, not by feeding real source through the interner -- can be
/// exercised against the actual matching code, in
/// `a_sixty_four_bit_hash_collision_is_rejected_by_find_pairs`, without
/// needing a multi-billion-token corpus to provoke it.
fn find_pairs(
    files: &[SourceFile],
    streams: &[Vec<PlacedToken>],
    ids: &[Vec<u32>],
    min_tokens: usize,
) -> Vec<BlockPair> {
    let mut buckets: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    for (file_index, file_ids) in ids.iter().enumerate() {
        for (start, hash) in rolling_hashes(file_ids, min_tokens).into_iter().enumerate() {
            buckets.entry(hash).or_default().push((file_index, start));
        }
    }

    let mut pairs = Vec::new();
    for occurrences in buckets.values_mut() {
        // Sorted explicitly rather than trusted to arrive in this order:
        // insertion order happens to already be `(file, start)` ascending
        // (files are processed in order, and so are the start positions
        // within one), but the anchor -- and therefore whether an unchanged
        // fragment gets an identical set of pairs on a second scan -- must
        // not depend on that staying true. See the module docs for why.
        occurrences.sort_unstable();
        // Every bucket got here via `.or_default().push(...)`, so it is
        // never empty.
        let (fa, sa) = occurrences[0];
        for &(fb, sb) in &occurrences[1..] {
            if !windows_match(&ids[fa], sa, &ids[fb], sb, min_tokens) {
                continue;
            }
            if !is_left_maximal(&ids[fa], sa, &ids[fb], sb) {
                continue;
            }
            let length = extend_forward(&ids[fa], sa, &ids[fb], sb, min_tokens);
            let names: Vec<&str> = streams[fa][sa..sa + length].iter().map(|t| t.name.as_str()).collect();
            pairs.push(BlockPair {
                a: block_ref(&files[fa].path, &streams[fa], sa, length),
                b: block_ref(&files[fb].path, &streams[fb], sb, length),
                tokens: length,
                hash: ContentHash::of(&names),
            });
        }
    }
    pairs
}

/// Whether a seed match at `(sa, sb)` is the true start of its run.
///
/// True at either stream's start, or when the token immediately before the
/// window differs -- see the module docs for why this is what keeps one
/// repeated run from being reported once per interior offset it can be seeded
/// from.
fn is_left_maximal(a: &[u32], sa: usize, b: &[u32], sb: usize) -> bool {
    sa == 0 || sb == 0 || a[sa - 1] != b[sb - 1]
}

/// Extend a `min_tokens`-token seed match as far as the ids keep agreeing.
fn extend_forward(a: &[u32], sa: usize, b: &[u32], sb: usize, min_tokens: usize) -> usize {
    let mut length = min_tokens;
    while sa + length < a.len() && sb + length < b.len() && a[sa + length] == b[sb + length] {
        length += 1;
    }
    length
}

/// Build a [`BlockRef`] spanning a run's first and last token.
///
/// `file` uses `/` separators regardless of platform, matching every other
/// path the report format carries.
fn block_ref(path: &Path, tokens: &[PlacedToken], start: usize, length: usize) -> BlockRef {
    BlockRef {
        file: path.to_string_lossy().replace('\\', "/"),
        start_line: tokens[start].line,
        end_line: tokens[start + length - 1].line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    fn python_file(path: &str, text: &str) -> SourceFile {
        SourceFile {
            path: PathBuf::from(path),
            language: lang::by_name("python").expect("python is registered"),
            text: text.to_string(),
        }
    }

    fn js_file(path: &str, text: &str) -> SourceFile {
        SourceFile {
            path: PathBuf::from(path),
            language: lang::by_name("javascript").expect("javascript is registered"),
            text: text.to_string(),
        }
    }

    fn opts(min_tokens: usize) -> BlockOptions {
        BlockOptions { min_tokens }
    }

    fn find(files: Vec<SourceFile>, min_tokens: usize) -> Vec<BlockPair> {
        find_blocks(&files, &opts(min_tokens))
    }

    fn pair_is(pair: &BlockPair, x: &str, y: &str) -> bool {
        (pair.a.file == x && pair.b.file == y) || (pair.a.file == y && pair.b.file == x)
    }

    /// The biggest block connecting files `x` and `y`. Results are sorted by
    /// descending token count, so this is also the *first* match -- which
    /// matters here because a low `min_tokens` can additionally surface
    /// small, uninteresting shared idioms (e.g. two `if` conditions that
    /// happen to both compare with `>`) alongside the fragment a test is
    /// actually about.
    fn biggest<'a>(blocks: &'a [BlockPair], x: &str, y: &str) -> &'a BlockPair {
        // A static message, not an interpolated one: CONTRIBUTING forbids
        // assertions with a failure-only branch, and a `panic!` built from
        // format arguments is exactly that -- the formatting only runs when
        // the block is missing, so it would sit permanently uncovered on the
        // success path every test here actually takes.
        blocks.iter().find(|p| pair_is(p, x, y)).expect("no block found for the given file pair")
    }

    /// Three-statement fragment used across most fixtures below: an
    /// assignment with a nested arithmetic expression, a method call, and an
    /// `if` guarding a boolean assignment. Each of the three has a shape
    /// found nowhere else in these fixtures, which is deliberate --
    /// normalization abstracts identifiers and literal values away, so two
    /// *different* simple statements (say, two plain-binary-operator
    /// assignments differing only by which operator they use) would still
    /// share nearly all of their normalized tokens and pollute a test with
    /// unrelated matches.
    fn alpha() -> SourceFile {
        python_file(
            "alpha.py",
            "def alpha(n):\n    result = base * rate + offset\n    values.append(result)\n    if result > threshold:\n        flagged = True\n    return result\n",
        )
    }

    fn beta() -> SourceFile {
        python_file(
            "beta.py",
            "def beta(x, y, z):\n    log(x, y, z)\n    result = base * rate + offset\n    values.append(result)\n    if result > threshold:\n        flagged = True\n    print(result)\n",
        )
    }

    // ---------------------------------------------------------- behaviour 1

    #[test]
    fn a_fragment_repeated_inside_two_differently_shaped_functions_is_found() {
        let blocks = find(vec![alpha(), beta()], 15);

        let found = biggest(&blocks, "alpha.py", "beta.py");
        assert!(found.tokens > 20);
    }

    // ---------------------------------------------------------- behaviour 2

    #[test]
    fn the_whole_shared_fragment_produces_one_finding_not_many_overlapping_ones() {
        // min_tokens well below the true run length is what makes every
        // interior offset of the run seed its own (wrong) match if left
        // unfiltered -- see the module docs.
        let full = find(vec![alpha(), beta()], 1);
        let longest = biggest(&full, "alpha.py", "beta.py").tokens;
        assert!(longest > 20);

        let at_fifteen = find(vec![alpha(), beta()], 15);
        let matching_ab: Vec<usize> =
            at_fifteen.iter().filter(|p| pair_is(p, "alpha.py", "beta.py")).map(|p| p.tokens).collect();
        assert_eq!(matching_ab, vec![longest]);
    }

    // ---------------------------------------------------------- behaviour 3

    fn gamma() -> SourceFile {
        python_file(
            "gamma.py",
            "def gamma(items):\n    total = 0\n    for item in items:\n        if item > 0:\n            total = total + item\n    return total\n",
        )
    }

    fn delta() -> SourceFile {
        python_file(
            "delta.py",
            "def delta(items, bonus):\n    extra = bonus * 2\n    total = 0\n    for item in items:\n        if item > 0:\n            total = total + item\n    print(total, extra)\n",
        )
    }

    #[test]
    fn different_fragments_never_share_a_hash() {
        let blocks = find(vec![alpha(), beta(), gamma(), delta()], 15);

        let ab = biggest(&blocks, "alpha.py", "beta.py");
        let gd = biggest(&blocks, "gamma.py", "delta.py");
        assert_ne!(ab.hash, gd.hash);
    }

    fn epsilon() -> SourceFile {
        python_file(
            "epsilon.py",
            "def epsilon(q):\n    if q:\n        pass\n    result = base * rate + offset\n    values.append(result)\n    if result > threshold:\n        flagged = True\n    yield result\n",
        )
    }

    #[test]
    fn the_same_fragment_in_a_different_file_pair_hashes_the_same() {
        let blocks = find(vec![alpha(), beta(), epsilon()], 15);

        let ab = biggest(&blocks, "alpha.py", "beta.py");
        let ae = biggest(&blocks, "alpha.py", "epsilon.py");
        assert_eq!(ab.hash, ae.hash);
    }

    // ---------------------------------------------------------- behaviour 4

    #[test]
    fn a_repeat_inside_a_single_file_is_found_without_duplicating_itself() {
        let both = python_file(
            "both.py",
            "def one(n):\n    result = base * rate + offset\n    values.append(result)\n    if result > threshold:\n        flagged = True\n    return result\n\ndef two(x, y, z):\n    log(x, y, z)\n    result = base * rate + offset\n    values.append(result)\n    if result > threshold:\n        flagged = True\n    print(result)\n",
        );

        let blocks = find(vec![both], 15);

        let found = biggest(&blocks, "both.py", "both.py");
        assert!(found.tokens > 20);
        assert_ne!(found.a.start_line, found.b.start_line);
    }

    // ---------------------------------------------------------- behaviour 5

    #[test]
    fn a_run_of_exactly_min_tokens_is_reported_but_one_shorter_is_not() {
        let full = find(vec![alpha(), beta()], 1);
        let longest = biggest(&full, "alpha.py", "beta.py").tokens;

        let at_boundary = find(vec![alpha(), beta()], longest);
        let one_more = find(vec![alpha(), beta()], longest + 1);

        // One closure reused for both calls, not two copies of it: at
        // `longest + 1` no window of that size exists anywhere in the
        // fixture, so `one_more` is empty and a *second*, separately written
        // closure would never run its body -- an unrelated, permanently
        // uncovered line rather than a real gap in the property being tested.
        let has_full_match = |blocks: &[BlockPair]| {
            blocks.iter().any(|p| pair_is(p, "alpha.py", "beta.py") && p.tokens == longest)
        };
        let found = (has_full_match(&at_boundary), has_full_match(&one_more));
        assert_eq!(found, (true, false));
    }

    // ---------------------------------------------------------- behaviour 6

    #[test]
    fn a_blind_renamed_copy_is_still_one_duplicate() {
        let zeta = python_file(
            "zeta.py",
            "def zeta(n):\n    result = base * rate + offset\n    values.append(result)\n    if result > threshold:\n        flagged = True\n    return result\n",
        );
        // Same shape as `alpha`/`beta`'s fragment, every identifier renamed.
        let eta = python_file(
            "eta.py",
            "def eta(x, y, z):\n    log(x, y, z)\n    outcome = m * k + shift\n    bucket.append(outcome)\n    if outcome > limit:\n        hit = True\n    print(outcome)\n",
        );

        let full = find(vec![alpha(), beta()], 1);
        let longest = biggest(&full, "alpha.py", "beta.py").tokens;

        let blocks = find(vec![zeta, eta], 15);
        assert_eq!(biggest(&blocks, "zeta.py", "eta.py").tokens, longest);
    }

    // ---------------------------------------------------------- behaviour 7

    #[test]
    fn output_is_deterministic_and_sorted_by_descending_tokens() {
        let files = vec![alpha(), beta(), gamma(), delta()];
        let first = find_blocks(&files, &opts(15));
        let second = find_blocks(&files, &opts(15));

        assert_eq!(first, second);
        assert!(first.len() >= 2);
        assert!(first.windows(2).all(|w| w[0].tokens >= w[1].tokens));
    }

    // ---------------------------------------------------------- behaviour 8

    #[test]
    fn empty_input_produces_no_findings() {
        assert_eq!(find_blocks(&[], &opts(5)), Vec::new());
    }

    #[test]
    fn a_single_file_with_no_internal_repeat_produces_no_findings() {
        assert_eq!(find(vec![alpha()], 15), Vec::new());
    }

    #[test]
    fn a_min_tokens_above_every_files_length_produces_no_findings() {
        assert_eq!(find(vec![alpha(), beta()], 10_000), Vec::new());
    }

    // ---------------------------------------------------------- behaviour 9

    #[test]
    fn block_ref_lines_are_one_based_inclusive_and_file_uses_forward_slashes() {
        let one = js_file(
            "src/nested/one.js",
            "function one(n) {\n  let result = base * rate + offset;\n  values.push(result);\n  if (result > threshold) {\n    flagged = true;\n  }\n  return result;\n}\n",
        );
        // A path containing a literal backslash -- on this platform that is
        // just an ordinary filename character, which is exactly why the
        // conversion cannot be left to `Path` alone and must replace it by
        // hand.
        let two = js_file(
            "win\\pkg\\two.js",
            "function two(x, y, z) {\n  log(x, y, z);\n  let result = base * rate + offset;\n  values.push(result);\n  if (result > threshold) {\n    flagged = true;\n  }\n  console.log(result);\n}\n",
        );

        let blocks = find(vec![one, two], 15);

        let found = biggest(&blocks, "src/nested/one.js", "win/pkg/two.js");
        assert_eq!(
            (found.a.file.as_str(), found.a.start_line, found.a.end_line),
            ("src/nested/one.js", 2, 6)
        );
        assert_eq!((found.b.file.as_str(), found.b.start_line, found.b.end_line), ("win/pkg/two.js", 3, 7));
    }

    // ------------------------------------------------------- anchor pairing

    /// One occurrence of the shared fragment: no prefix and an identical
    /// (parameterless) signature in every file, so every occurrence's
    /// backward context is byte-identical up to the very start of the file --
    /// only the suffix, which sits *after* the fragment and so cannot affect
    /// which occurrences a backward extension groups together, tells them
    /// apart. That is deliberate: giving occurrences merely *different*
    /// prefixes (tried first) still lets pairs of non-anchor occurrences
    /// share a coincidentally-identical tail -- every prefix closes with at
    /// least one `#END`, itself the same token regardless of what it closes
    /// -- which seeds *extra* anchor groups among the non-anchor occurrences
    /// themselves. Identical prefixes side-step that: there is nothing for
    /// two non-anchor occurrences to coincidentally share that the anchor
    /// does not *also* share, so the whole group -- anchor included -- stays
    /// one bucket, not several.
    fn repeated_occurrence(path: &str, suffix: &str) -> SourceFile {
        python_file(
            path,
            &format!(
                "def f():\n    result = base * rate + offset\n    values.append(result)\n    if result > threshold:\n        flagged = True\n    {suffix}\n"
            ),
        )
    }

    #[test]
    fn a_fragment_repeated_n_times_yields_n_minus_one_findings_covering_every_occurrence() {
        // Five occurrences of the same fragment -- a complete graph would
        // report `5 * 4 / 2 = 10` pairs; anchored pairing reports `5 - 1 = 4`.
        // Suffixes are mutually distinct statement kinds so a forward
        // extension cannot run past the fragment into another match.
        let file_names = ["site_a.py", "site_b.py", "site_c.py", "site_d.py", "site_e.py"];
        let suffixes = ["return result", "print(result)", "yield result", "assert result", "pass"];
        let files: Vec<SourceFile> =
            file_names.iter().zip(suffixes).map(|(path, suffix)| repeated_occurrence(path, suffix)).collect();
        let occurrence_count = files.len();

        let blocks = find(files, 15);

        assert_eq!(blocks.len(), occurrence_count - 1);

        let covered: BTreeSet<&str> =
            blocks.iter().flat_map(|p| [p.a.file.as_str(), p.b.file.as_str()]).collect();
        let expected: BTreeSet<&str> = file_names.into_iter().collect();
        assert_eq!(covered, expected);
    }

    // ------------------------------------------------------- prefilter perf

    /// Reference implementation of [`find_blocks`], used to cross-check the
    /// rolling-hash prefilter. Deliberately naive: hashes every window with
    /// [`ContentHash::of`] -- the real, cryptographic hash, recomputed from
    /// scratch each time -- instead of a rolling integer one, and compares
    /// token *names* directly instead of interned ids. Everything else
    /// (left-maximal anchor, forward extension, final sort) is identical, so
    /// any disagreement between the two proves the prefilter dropped or
    /// invented a finding, not that the two implementations disagree about
    /// what counts as a match.
    fn naive_find_blocks(files: &[SourceFile], options: &BlockOptions) -> Vec<BlockPair> {
        let min_tokens = options.min_tokens;
        let streams: Vec<Vec<PlacedToken>> = files.iter().map(placed_tokens).collect();

        let mut buckets: HashMap<ContentHash, Vec<(usize, usize)>> = HashMap::new();
        for (file_index, tokens) in streams.iter().enumerate() {
            if tokens.len() < min_tokens {
                continue;
            }
            for start in 0..=(tokens.len() - min_tokens) {
                let names: Vec<&str> =
                    tokens[start..start + min_tokens].iter().map(|t| t.name.as_str()).collect();
                buckets.entry(ContentHash::of(&names)).or_default().push((file_index, start));
            }
        }

        let mut pairs = Vec::new();
        for occurrences in buckets.values_mut() {
            // Same anchoring rule as `find_pairs`, arrived at independently:
            // one occurrence group is one fact, not one finding per pair of
            // its locations, so only the earliest occurrence (by `(file,
            // start)`, sorted explicitly so this is not incidental) is
            // paired against every other one.
            occurrences.sort_unstable();
            let (fa, sa) = occurrences[0];
            for &(fb, sb) in &occurrences[1..] {
                let a = &streams[fa];
                let b = &streams[fb];
                let left_maximal = sa == 0 || sb == 0 || a[sa - 1].name != b[sb - 1].name;
                if !left_maximal {
                    continue;
                }
                let mut length = min_tokens;
                while sa + length < a.len()
                    && sb + length < b.len()
                    && a[sa + length].name == b[sb + length].name
                {
                    length += 1;
                }
                let names: Vec<&str> = a[sa..sa + length].iter().map(|t| t.name.as_str()).collect();
                pairs.push(BlockPair {
                    a: block_ref(&files[fa].path, a, sa, length),
                    b: block_ref(&files[fb].path, b, sb, length),
                    tokens: length,
                    hash: ContentHash::of(&names),
                });
            }
        }

        pairs.sort_by_key(|p| {
            (
                std::cmp::Reverse(p.tokens),
                p.hash.clone(),
                p.a.file.clone(),
                p.a.start_line,
                p.a.end_line,
                p.b.file.clone(),
                p.b.start_line,
                p.b.end_line,
            )
        });
        pairs
    }

    #[test]
    fn the_rolling_hash_prefilter_matches_a_naive_all_windows_reference() {
        let corpus = vec![alpha(), beta(), gamma(), delta(), epsilon()];
        let thresholds = [1usize, 8, 20, 200];

        let fast: Vec<Vec<BlockPair>> = thresholds.iter().map(|&m| find(corpus.clone(), m)).collect();
        let naive: Vec<Vec<BlockPair>> =
            thresholds.iter().map(|&m| naive_find_blocks(&corpus, &opts(m))).collect();

        assert!(fast.iter().any(|blocks| !blocks.is_empty()));
        assert_eq!(fast, naive);
    }

    #[test]
    fn a_sixty_four_bit_hash_collision_is_rejected_by_the_confirmation_step() {
        // ROLLING_BASE is round(2^64 / phi), so its continued fraction is the
        // golden ratio's -- all 1s -- and Fibonacci numbers are exactly its
        // best small-denominator rational approximations. That gives an
        // honest, constructive (not simulated) 64-bit collision: fib(47) is
        // the largest Fibonacci number under 2^32, and multiplying it by
        // ROLLING_BASE lands within 2^32 of an exact multiple of 2^64 --
        // close enough that a second, all-zero window reaches the identical
        // wrapped hash. Verified by direct computation below, not asserted
        // on faith.
        let fib_47 = 2_971_215_073u32;
        let offset = 50_920_843u32;
        let window_a: &[u32] = &[fib_47, offset];
        let window_b: &[u32] = &[0, 0];
        assert_ne!(window_a, window_b);

        let hash_a = rolling_hashes(window_a, 2);
        let hash_b = rolling_hashes(window_b, 2);
        assert_eq!(hash_a, hash_b);

        // The confirmation step must still reject this pair despite the
        // colliding rolling hash: a 64-bit collision must never produce a
        // false finding.
        assert!(!windows_match(window_a, 0, window_b, 0, 2));
    }

    #[test]
    fn a_sixty_four_bit_hash_collision_produces_no_finding_through_find_pairs() {
        // The test above proves the confirmation primitive rejects a
        // collision in isolation. This one drives the actual matching code
        // `find_blocks` runs -- `find_pairs`, working on ids and streams
        // exactly as it receives them from a real scan -- so the rejection is
        // proven on the real code path, not just the primitive it is built
        // from. The ids are fabricated deliberately: reaching id `2_971_215_073`
        // through the real interner would need billions of distinct
        // previously-unseen token names in one file, which is not a
        // reachable test fixture.
        let fib_47 = 2_971_215_073u32;
        let offset = 50_920_843u32;
        assert_eq!(rolling_hashes(&[fib_47, offset], 2), rolling_hashes(&[0, 0], 2));

        let files = vec![python_file("colliding_a.py", ""), python_file("colliding_b.py", "")];
        // Genuinely different token names on each side -- if the confirmation
        // step failed to reject the collision, these would still be reported
        // as a match, which is exactly the false finding this guards against.
        let streams = vec![
            vec![
                PlacedToken { name: "alpha_token".to_string(), line: 1 },
                PlacedToken { name: "beta_token".to_string(), line: 1 },
            ],
            vec![
                PlacedToken { name: "gamma_token".to_string(), line: 1 },
                PlacedToken { name: "delta_token".to_string(), line: 1 },
            ],
        ];
        let ids = vec![vec![fib_47, offset], vec![0u32, 0u32]];

        assert_eq!(find_pairs(&files, &streams, &ids, 2), Vec::new());
    }

    // ------------------------------------------------------------ plumbing

    #[test]
    fn block_options_and_placed_token_clone_and_debug() {
        let options = BlockOptions { min_tokens: 5 };
        let debug = format!("{options:?}").contains("min_tokens");
        assert_eq!((options.clone().min_tokens, debug), (5, true));

        let token = PlacedToken { name: "ID".to_string(), line: 3 };
        let debug_token = format!("{token:?}").contains("PlacedToken");
        assert_eq!((token.clone(), debug_token), (token, true));
    }

    #[test]
    fn placed_tokens_records_a_line_per_token() {
        // module, expression_statement, assignment, ID, "=", #END("="), NUM,
        // #END(assignment), #END(expression_statement), #END(module) -- the
        // trailing newline puts the file's last line at row 1 (line 2), which
        // is where the module's own closing token lands.
        let file = python_file("s.py", "x = 1\n");
        let placed = placed_tokens(&file);
        let names: Vec<&str> = placed.iter().map(|t| t.name.as_str()).collect();
        let lines: Vec<usize> = placed.iter().map(|t| t.line).collect();
        assert_eq!(
            (names, lines),
            (
                vec![
                    "module",
                    "expression_statement",
                    "assignment",
                    "ID",
                    "=",
                    "#END",
                    "NUM",
                    "#END",
                    "#END",
                    "#END"
                ],
                vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            )
        );
    }
}
