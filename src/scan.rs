//! Function-level near-duplicate detection: the O(n²) core of a scan.
//!
//! [`find_clones`] compares every pair of extracted [`Unit`]s and reports the
//! ones similar enough to matter. On a real tree that is millions of
//! comparisons, so two things keep it tractable without changing what it
//! finds:
//!
//! - A [`similarity::Matcher`] is built once per right-hand unit and reused
//!   across every left-hand unit compared against it, so the O(n) indexing
//!   cost is paid once per unit rather than once per pair.
//! - The three-tier prune from [`similarity`] rules a pair out as cheaply as
//!   possible: [`similarity::real_quick_ratio`] from lengths alone, then
//!   [`similarity::quick_ratio`] from a token multiset, and only then the real
//!   [`similarity::ratio`]. Each bound is proven `>= ratio`, so pruning can
//!   never drop a pair that would have passed — see
//!   `find_clones_agrees_with_a_naive_all_pairs_scan` below, which is the test
//!   that would catch it if that ever stopped being true.
//!
//! [`cluster`] groups the resulting pairs into connected components for a
//! human to read, via [`crate::unionfind::UnionFind`].

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

use crate::extract::Unit;
use crate::report::{ClonePair, UnitRef};
use crate::similarity;
use crate::unionfind::UnionFind;

/// Compare every pair of units and return those at or above `min_similarity`.
///
/// Each unordered pair is considered once (left index strictly less than right
/// index), so a unit is never compared with itself and no pair is reported
/// twice. A pair whose units share a file and whose line spans nest — a thin
/// wrapper and the function nested directly inside it, say — is skipped even
/// if it clears the threshold: the outer unit's token stream contains the
/// inner one's by construction, so reporting the pair is noise, not a finding.
///
/// The outer loop runs on [`rayon`]'s pool, so pairs are found in a
/// nondeterministic order; the result is sorted (descending similarity, then
/// [`ClonePair::key`]) before it is returned, so two scans of the same units
/// always produce the same output.
pub fn find_clones(units: &[Unit], min_similarity: f64) -> Vec<ClonePair> {
    let mut pairs: Vec<ClonePair> = (0..units.len())
        .into_par_iter()
        .flat_map_iter(|j| {
            let right = &units[j];
            let matcher = similarity::Matcher::new(right.stream.tokens());
            (0..j).filter_map(move |i| {
                let left = &units[i];
                if nested_in_same_file(left, right) {
                    return None;
                }
                let left_tokens = left.stream.tokens();
                // Cheapest bound first: from lengths alone, no allocation.
                if matcher.real_quick_ratio(left_tokens) < min_similarity {
                    return None;
                }
                // Next cheapest: order-blind multiset overlap, O(n).
                if matcher.quick_ratio(left_tokens) < min_similarity {
                    return None;
                }
                let similarity = matcher.ratio(left_tokens);
                if similarity < min_similarity {
                    return None;
                }
                Some(ClonePair { similarity, a: unit_ref(left), b: unit_ref(right) })
            })
        })
        .collect();

    pairs.sort_by(|x, y| {
        y.similarity
            .partial_cmp(&x.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.key().cmp(&y.key()))
    });
    pairs
}

/// Whether `a` and `b` live in the same file with one's line span containing
/// the other's.
///
/// A thin wrapper function and the function nested directly inside it satisfy
/// this by construction — the outer body's token stream always contains the
/// inner one's — so without this check every such pair would score as a
/// near-perfect clone of itself.
fn nested_in_same_file(a: &Unit, b: &Unit) -> bool {
    a.path == b.path
        && ((a.start_line <= b.start_line && b.end_line <= a.end_line)
            || (b.start_line <= a.start_line && a.end_line <= b.end_line))
}

/// Render a [`Unit`] as the [`UnitRef`] a report carries.
fn unit_ref(unit: &Unit) -> UnitRef {
    UnitRef {
        file: normalize_path(&unit.path),
        qualname: unit.qualname.clone(),
        start_line: unit.start_line,
        end_line: unit.end_line,
        hash: unit.stream.hash().clone(),
    }
}

/// Render a path with `/` separators, so a report scanned on Windows and one
/// scanned on Linux describe the same file the same way and compare equal.
fn normalize_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Group units into clone classes via the pairs that link them, as indices
/// into `units`. Each returned class is an ascending list of two or more
/// member indices; classes are ordered by their lowest member.
///
/// A unit no pair links to anything is not returned as a class of one — there
/// is nothing for a reader to look at.
///
/// # Clustering is transitive; similarity is not
///
/// If `A` resembles `B` and `B` resembles `C`, all three land in one class
/// even when `A` and `C` share nothing at all — resemblance does not chain.
/// On a large tree this can pull unrelated units into a single class. The
/// *pair*, with its own measured similarity, is the finding; a class is only
/// a reading aid for grouping related pairs, never a claim that every member
/// resembles every other member.
pub fn cluster(pairs: &[ClonePair], units: &[Unit]) -> Vec<Vec<usize>> {
    // A unit's file plus its line span identifies it uniquely within one
    // extraction: two units can never share both, since sibling spans differ
    // and a nested span is strictly smaller than its enclosing one.
    let mut index: HashMap<(String, usize, usize), usize> = HashMap::new();
    for (i, unit) in units.iter().enumerate() {
        index.insert((normalize_path(&unit.path), unit.start_line, unit.end_line), i);
    }

    let mut forest = UnionFind::new(units.len());
    for pair in pairs {
        let a = *index
            .get(&(pair.a.file.clone(), pair.a.start_line, pair.a.end_line))
            .expect("a ClonePair's endpoints are units this same scan extracted");
        let b = *index
            .get(&(pair.b.file.clone(), pair.b.start_line, pair.b.end_line))
            .expect("a ClonePair's endpoints are units this same scan extracted");
        forest.union(a, b);
    }

    forest.groups().into_iter().filter(|group| group.len() >= 2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::Extractor;
    use crate::lang;
    use crate::token::{Interner, TokenStream};
    use std::path::PathBuf;

    fn python() -> Extractor {
        Extractor::new(lang::by_name("python").expect("python is registered"))
    }

    /// Extract every unit from a Python source string with a permissive size
    /// threshold, sharing one interner as a real scan would.
    fn units_of(source: &str) -> Vec<Unit> {
        let mut interner = Interner::new();
        python().extract(source, Path::new("sample.py"), 1, &mut interner).units
    }

    fn pair_qualnames(p: &ClonePair) -> (String, String) {
        (p.a.qualname.clone(), p.b.qualname.clone())
    }

    // -------------------------------------------------- prune soundness (#1)

    /// Deliberately naive reference: computes the real ratio for every
    /// unordered pair with no pruning, applying the same containment skip and
    /// the same sort as `find_clones`. This is the second opinion the prune
    /// tiers are checked against, not a copy of the optimized path.
    fn naive_find_clones(units: &[Unit], min_similarity: f64) -> Vec<ClonePair> {
        let mut pairs = Vec::new();
        for i in 0..units.len() {
            for j in (i + 1)..units.len() {
                if nested_in_same_file(&units[i], &units[j]) {
                    continue;
                }
                let sim = similarity::ratio(units[i].stream.tokens(), units[j].stream.tokens());
                if sim >= min_similarity {
                    pairs.push(ClonePair { similarity: sim, a: unit_ref(&units[i]), b: unit_ref(&units[j]) });
                }
            }
        }
        pairs.sort_by(|x, y| {
            y.similarity
                .partial_cmp(&x.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.key().cmp(&y.key()))
        });
        pairs
    }

    #[test]
    fn find_clones_agrees_with_a_naive_all_pairs_scan() {
        // A deliberately mixed corpus: pairs that differ enough in length to
        // be pruned by real_quick_ratio alone, pairs the same length but
        // built from different token multisets, pairs that share a multiset
        // but not an order, and one exact duplicate. If pruning ever dropped
        // a pair the real ratio would have kept, this is what would notice.
        let source = "\
def a(x):
    return x

def b(w, x, y, z, total):
    total = w + x
    total = total + y
    total = total + z
    if total > 10:
        total = total - 1
    return total

def c(w, x, y, z, total):
    total = w + x
    total = total + y
    total = total + z
    if total > 10:
        total = total - 1
    return total

def d(p, q, r, s, sum):
    sum = p - q
    sum = sum - r
    sum = sum - s
    if sum < 0:
        sum = sum + 1
    return sum

def e(m):
    return m * 2

def f(n):
    return n + 3

def outer_wrap():
    def inner_wrap():
        return 1
    return inner_wrap()
";
        let units = units_of(source);
        for threshold in [0.0, 0.3, 0.6, 0.9] {
            let got = find_clones(&units, threshold);
            let want = naive_find_clones(&units, threshold);
            assert_eq!(got, want);
        }
        // The corpus must actually exercise agreement on a nonempty result,
        // or the loop above would pass vacuously.
        assert!(!naive_find_clones(&units, 0.0).is_empty());
    }

    /// Build a `Unit` from an explicit token name list, for tests that need
    /// exact control over multiset overlap and ordering -- a shape that
    /// choosing real source text to hit precisely is impractical for.
    fn build_unit(
        file: &str,
        qualname: &str,
        start_line: usize,
        end_line: usize,
        tokens: &[&str],
        interner: &mut Interner,
    ) -> Unit {
        Unit {
            path: PathBuf::from(file),
            qualname: qualname.to_string(),
            start_line,
            end_line,
            node_count: tokens.len(),
            stream: TokenStream::intern(tokens, interner),
        }
    }

    #[test]
    fn each_prune_tier_rejects_a_pair_the_next_cheaper_tier_would_have_let_through() {
        // Both pairs below have equal-length streams, so real_quick_ratio is
        // always 1.0 and is never itself the reason for a skip -- isolating
        // the next two tiers.
        let mut interner = Interner::new();

        // Same length, low multiset overlap: real_quick_ratio passes (1.0),
        // quick_ratio (0.5) does not clear 0.6. Exercises the quick_ratio
        // prune branch.
        let low_overlap_left = build_unit(
            "p.py",
            "low_overlap_left",
            1,
            2,
            &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            &mut interner,
        );
        let low_overlap_right = build_unit(
            "p.py",
            "low_overlap_right",
            10,
            11,
            &["a", "b", "c", "d", "e", "k", "l", "m", "n", "o"],
            &mut interner,
        );

        // Same multiset, reordered: quick_ratio is order-blind and reports a
        // perfect 1.0, but the real ratio -- which only counts contiguous
        // runs -- is 0.5 (the exact gap `ratio_penalises_reordering_that_
        // the_multiset_bound_cannot_see` in similarity.rs names). Exercises
        // the final ratio prune branch.
        let shuffled_left =
            build_unit("p.py", "shuffled_left", 20, 21, &["t1", "t2", "t3", "t4", "t5", "t6"], &mut interner);
        let shuffled_right = build_unit(
            "p.py",
            "shuffled_right",
            30,
            31,
            &["t4", "t5", "t6", "t1", "t2", "t3"],
            &mut interner,
        );

        let units = vec![low_overlap_left, low_overlap_right, shuffled_left, shuffled_right];
        let names: Vec<(String, String)> = find_clones(&units, 0.6).iter().map(pair_qualnames).collect();
        assert!(!names.contains(&("low_overlap_left".to_string(), "low_overlap_right".to_string())));
        assert!(!names.contains(&("shuffled_left".to_string(), "shuffled_right".to_string())));
    }

    // ------------------------------------------------ self and dedup (#2)

    #[test]
    fn a_unit_is_never_paired_with_itself_and_each_pair_is_reported_once() {
        // Three functions with identical logic (only the name differs) are
        // mutually a perfect clone: with 3 units that is exactly 3 unordered
        // pairs, never 6 ordered ones and never a pair of a unit with itself.
        let source = "\
def one(a, b):
    return a + b

def two(a, b):
    return a + b

def three(a, b):
    return a + b
";
        let units = units_of(source);
        let pairs = find_clones(&units, 1.0);
        assert_eq!(pairs.len(), 3);
        // All three pairs tie on similarity, so their relative order depends
        // on content hashes -- sort by name here for a deterministic
        // assertion instead of asserting the tie-break order itself.
        let mut names: Vec<(String, String)> = pairs.iter().map(pair_qualnames).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                ("one".to_string(), "three".to_string()),
                ("one".to_string(), "two".to_string()),
                ("two".to_string(), "three".to_string()),
            ]
        );
        let self_pairs: usize = pairs.iter().filter(|p| p.a.qualname == p.b.qualname).count();
        assert_eq!(self_pairs, 0);
    }

    // ----------------------------------------------------- containment (#3)

    #[test]
    fn a_wrapper_is_not_reported_as_a_clone_of_the_function_nested_inside_it() {
        // `wrapper`'s token stream contains `helper`'s in its entirety, so at
        // min_similarity 0.0 -- which nothing else could fail -- the only way
        // to end up with zero pairs is the containment skip.
        let source = "\
def wrapper(a, b):
    def helper(a, b):
        return a + b
    return helper(a, b)
";
        let units = units_of(source);
        assert_eq!(
            units.iter().map(|u| u.qualname.as_str()).collect::<Vec<_>>(),
            vec!["wrapper", "wrapper.helper"]
        );
        assert!(find_clones(&units, 0.0).is_empty());
    }

    // --------------------------------------------------- not over-applied (#4)

    #[test]
    fn two_similar_functions_in_the_same_file_that_do_not_nest_are_reported() {
        let source = "\
def alpha(x, y):
    return x + y

def beta(x, y):
    return x + y
";
        let units = units_of(source);
        let pairs = find_clones(&units, 1.0);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pair_qualnames(&pairs[0]), ("alpha".to_string(), "beta".to_string()));
    }

    // -------------------------------------------------------- determinism (#5)

    #[test]
    fn repeated_scans_of_the_same_units_produce_identical_output() {
        let source = "\
def one(a, b, c):
    total = a + b
    total = total + c
    return total

def two(a, b, c):
    total = a + b
    total = total + c
    return total

def three(a, b, c):
    total = a - b
    total = total - c
    return total

def four(a, b, c):
    return a * b * c
";
        let units = units_of(source);
        let first = find_clones(&units, 0.0);
        let second = find_clones(&units, 0.0);
        assert!(!first.is_empty());
        assert_eq!(first, second);
    }

    // ------------------------------------------------------- threshold (#6)

    #[test]
    fn a_pair_at_the_threshold_is_reported_and_just_below_it_is_not() {
        let source = "\
def full(a, b, c):
    total = a + b
    total = total + c
    return total

def partial(a, b, c):
    total = a + b
    return total
";
        let units = units_of(source);
        let exact = similarity::ratio(units[0].stream.tokens(), units[1].stream.tokens());
        assert!(exact > 0.0 && exact < 1.0);

        let at_threshold = find_clones(&units, exact);
        let just_above = find_clones(&units, exact + 1e-9);
        assert_eq!(at_threshold.len(), 1);
        assert!(just_above.is_empty());
    }

    // --------------------------------------------------- empty / single (#7)

    #[test]
    fn empty_and_single_unit_inputs_produce_no_pairs_and_do_not_panic() {
        let empty: Vec<Unit> = Vec::new();
        assert!(find_clones(&empty, 0.0).is_empty());

        let single = units_of("def only(a):\n    return a\n");
        assert_eq!(single.len(), 1);
        assert!(find_clones(&single, 0.0).is_empty());
    }

    // --------------------------------------------------------- path style (#8)

    #[test]
    fn unit_ref_file_renders_with_forward_slashes() {
        let mut interner = Interner::new();
        let unit = Unit {
            path: PathBuf::from("windows\\style\\path.py"),
            qualname: "f".to_string(),
            start_line: 1,
            end_line: 2,
            node_count: 1,
            stream: TokenStream::intern(&["ID"], &mut interner),
        };
        assert_eq!(unit_ref(&unit).file, "windows/style/path.py");
    }

    // ------------------------------------------------------------- cluster (#9)

    #[test]
    fn linked_units_form_a_class_and_unlinked_units_are_not_singletons() {
        let source = "\
def one(a, b):
    return a + b

def two(a, b):
    return a + b

def three(a, b):
    return a + b

def unrelated(x):
    return x * x - 7
";
        let units = units_of(source);
        let pairs = find_clones(&units, 1.0);
        // `one`, `two`, `three` are mutually identical; `unrelated` shares no
        // pair with anything at this threshold.
        assert!(pairs.iter().all(|p| p.a.qualname != "unrelated" && p.b.qualname != "unrelated"));

        let classes = cluster(&pairs, &units);
        assert_eq!(classes, vec![vec![0usize, 1, 2]]);
    }

    #[test]
    fn a_class_can_chain_members_that_do_not_directly_resemble_each_other() {
        // `bridge` resembles both `left` and `right`, which is enough to link
        // all three into one class even though `left` and `right` need not be
        // similar to each other -- the transitivity the doc comment warns
        // about, pinned so nobody "fixes" cluster into requiring mutual
        // resemblance.
        let left = unit_ref_fixture("f.py", "left", 1, 3);
        let bridge = unit_ref_fixture("f.py", "bridge", 5, 7);
        let right = unit_ref_fixture("f.py", "right", 9, 11);
        let lone = unit_ref_fixture("f.py", "lone", 13, 15);

        let units =
            vec![unit_from_ref(&left), unit_from_ref(&bridge), unit_from_ref(&right), unit_from_ref(&lone)];
        let pairs = vec![
            ClonePair { similarity: 0.9, a: left.clone(), b: bridge.clone() },
            ClonePair { similarity: 0.9, a: bridge, b: right },
        ];

        assert_eq!(cluster(&pairs, &units), vec![vec![0usize, 1, 2]]);
    }

    /// A minimal `UnitRef` for tests that only need identity, not a real
    /// token stream.
    fn unit_ref_fixture(file: &str, qualname: &str, start_line: usize, end_line: usize) -> UnitRef {
        let mut interner = Interner::new();
        UnitRef {
            file: file.to_string(),
            qualname: qualname.to_string(),
            start_line,
            end_line,
            hash: TokenStream::intern(&[qualname], &mut interner).hash().clone(),
        }
    }

    /// The `Unit` a `unit_ref_fixture` would have been rendered from.
    fn unit_from_ref(r: &UnitRef) -> Unit {
        let mut interner = Interner::new();
        Unit {
            path: PathBuf::from(&r.file),
            qualname: r.qualname.clone(),
            start_line: r.start_line,
            end_line: r.end_line,
            node_count: 1,
            stream: TokenStream::intern(&[r.qualname.as_str()], &mut interner),
        }
    }

    // ------------------------------------------------------------ containment

    #[test]
    fn nested_in_same_file_holds_in_either_direction_and_not_across_files() {
        let outer = units_of("def outer():\n    def inner():\n        return 1\n    return inner\n");
        assert_eq!(outer.len(), 2);
        assert!(nested_in_same_file(&outer[0], &outer[1]));
        assert!(nested_in_same_file(&outer[1], &outer[0]));

        let mut other_file = outer[1].clone();
        other_file.path = PathBuf::from("elsewhere.py");
        assert!(!nested_in_same_file(&outer[0], &other_file));

        let siblings = units_of("def a():\n    return 1\n\ndef b():\n    return 2\n");
        assert!(!nested_in_same_file(&siblings[0], &siblings[1]));
    }
}
