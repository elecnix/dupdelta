//! Ratcliff–Obershelp similarity over interned token streams.
//!
//! This is the measure that decides whether two pieces of code are "the same
//! logic". It is deliberately the same measure Python's
//! `difflib.SequenceMatcher(autojunk=False).ratio()` computes, so that results
//! are comparable with the large body of clone-detection work built on it:
//!
//! > `ratio = 2 * M / T`, where `T` is the combined length of both sequences
//! > and `M` is the total size of the matching blocks found by recursively
//! > taking the longest matching block and recursing into the regions left and
//! > right of it.
//!
//! # Why not plain LCS or edit distance
//!
//! Longest-common-subsequence lets matches interleave arbitrarily, so two
//! functions that share only scattered boilerplate tokens (`if`, `return`,
//! `ID`) score high. Ratcliff–Obershelp only counts *contiguous* runs, which
//! is what "copy-pasted then edited" actually looks like.
//!
//! # The three-tier prune
//!
//! Scanning a tree is O(n²) in the number of units, and [`ratio`] is the
//! expensive part. Two cheap upper bounds run first, in increasing cost, and
//! either can rule a pair out without ever computing the real ratio:
//!
//! 1. [`real_quick_ratio`] — from the two lengths alone, O(1).
//! 2. [`quick_ratio`] — from token multiset intersection, O(n).
//! 3. [`ratio`] — the real thing.
//!
//! Both bounds are guaranteed `>= ratio`, so a pair pruned by either could
//! never have passed. That guarantee is load-bearing and is asserted in the
//! tests: if a bound were ever too low, the scanner would silently miss real
//! duplication — the exact failure this tool exists to prevent.

use std::collections::HashMap;

/// A run of tokens common to both sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match {
    /// Start index in the first sequence.
    pub a: usize,
    /// Start index in the second sequence.
    pub b: usize,
    /// Length of the run.
    pub size: usize,
}

/// Upper bound on [`ratio`] derived from sequence lengths alone.
///
/// `M` can never exceed the length of the shorter sequence, so
/// `2 * min(la, lb) / (la + lb)` bounds the ratio. Two empty sequences are
/// defined as identical (ratio `1.0`), matching `difflib`.
pub fn real_quick_ratio(la: usize, lb: usize) -> f64 {
    let total = la + lb;
    if total == 0 {
        return 1.0;
    }
    2.0 * la.min(lb) as f64 / total as f64
}

/// Upper bound on [`ratio`] derived from token multiset intersection.
///
/// Ignores ordering entirely, so it counts every token that *could* take part
/// in some matching block. Always `>= ratio`.
pub fn quick_ratio(a: &[u32], b: &[u32]) -> f64 {
    Matcher::new(b).quick_ratio(a)
}

/// Ratcliff–Obershelp similarity of two token sequences, in `0.0..=1.0`.
///
/// Convenience wrapper; build a [`Matcher`] instead when comparing one
/// sequence against many, which is what the scanner does.
pub fn ratio(a: &[u32], b: &[u32]) -> f64 {
    Matcher::new(b).ratio(a)
}

/// The maximal contiguous common runs of `a` and `b`, left to right.
///
/// Unlike `difflib`, no zero-length terminator is appended — an empty result
/// simply means the sequences share nothing — and no block-merging pass runs;
/// see [`Matcher::matching_blocks`] for why none is needed.
pub fn matching_blocks(a: &[u32], b: &[u32]) -> Vec<Match> {
    Matcher::new(b).matching_blocks(a)
}

/// A right-hand sequence with its index precomputed, for repeated comparison.
///
/// Building the occurrence index is the setup cost of a comparison. In an
/// O(n²) scan each unit is the right-hand side of many comparisons, so paying
/// that cost once per unit rather than once per pair is the difference between
/// a scan that finishes and one that does not.
pub struct Matcher<'b> {
    b: &'b [u32],
    /// Ascending occurrence positions of each token in `b`.
    occurrences: HashMap<u32, Vec<usize>>,
    /// Multiplicity of each token in `b`, for [`Matcher::quick_ratio`].
    counts: HashMap<u32, usize>,
}

impl<'b> Matcher<'b> {
    /// Index `b` for comparison against many left-hand sequences.
    pub fn new(b: &'b [u32]) -> Self {
        let mut occurrences: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for (j, &token) in b.iter().enumerate() {
            occurrences.entry(token).or_default().push(j);
            *counts.entry(token).or_insert(0) += 1;
        }
        Matcher { b, occurrences, counts }
    }

    /// The indexed right-hand sequence.
    pub fn b(&self) -> &[u32] {
        self.b
    }

    /// Upper bound on [`Matcher::ratio`] from lengths alone.
    pub fn real_quick_ratio(&self, a: &[u32]) -> f64 {
        real_quick_ratio(a.len(), self.b.len())
    }

    /// Upper bound on [`Matcher::ratio`] from token multiset intersection.
    pub fn quick_ratio(&self, a: &[u32]) -> f64 {
        let total = a.len() + self.b.len();
        if total == 0 {
            return 1.0;
        }
        let mut remaining = self.counts.clone();
        let mut shared = 0usize;
        for token in a {
            if let Some(left) = remaining.get_mut(token) {
                if *left > 0 {
                    *left -= 1;
                    shared += 1;
                }
            }
        }
        2.0 * shared as f64 / total as f64
    }

    /// Ratcliff–Obershelp similarity of `a` against the indexed sequence.
    pub fn ratio(&self, a: &[u32]) -> f64 {
        let total = a.len() + self.b.len();
        if total == 0 {
            return 1.0;
        }
        let matched: usize = self.matching_blocks(a).iter().map(|m| m.size).sum();
        2.0 * matched as f64 / total as f64
    }

    /// The maximal contiguous common runs of `a` and the indexed sequence.
    ///
    /// # Why there is no block-merging pass
    ///
    /// `difflib` finishes by fusing adjacent blocks, because its junk handling
    /// can split one run in two. This implementation has no junk handling, and
    /// without it two returned blocks can never abut in *both* sequences: if
    /// they did, their union would be a strictly longer contiguous common run
    /// inside the very window the parent search had already declared `m` the
    /// longest match of — a contradiction. So the merge pass would be dead
    /// code, and dead code that looks load-bearing is worse than no code.
    /// `matching_blocks_never_returns_runs_that_abut_in_both` pins this.
    pub fn matching_blocks(&self, a: &[u32]) -> Vec<Match> {
        let mut blocks = Vec::new();
        // Scratch buffers reused across the whole recursion. `previous_row` and
        // `current_row` hold run lengths ending at each index of `b`; the
        // `touched_*` lists let a row be cleared in time proportional to what
        // it actually wrote rather than to the length of `b`.
        let mut scratch = LongestMatchScratch::new(self.b.len());

        let mut queue = vec![(0usize, a.len(), 0usize, self.b.len())];
        while let Some((a_lo, a_hi, b_lo, b_hi)) = queue.pop() {
            let m = self.find_longest_match(a, a_lo, a_hi, b_lo, b_hi, &mut scratch);
            if m.size == 0 {
                continue;
            }
            blocks.push(m);
            if a_lo < m.a && b_lo < m.b {
                queue.push((a_lo, m.a, b_lo, m.b));
            }
            if m.a + m.size < a_hi && m.b + m.size < b_hi {
                queue.push((m.a + m.size, a_hi, m.b + m.size, b_hi));
            }
        }

        blocks.sort_unstable_by_key(|m| (m.a, m.b));
        blocks
    }

    /// Longest contiguous run common to `a[a_lo..a_hi]` and `b[b_lo..b_hi]`.
    ///
    /// Ties resolve to the earliest run, matching `difflib`.
    fn find_longest_match(
        &self,
        a: &[u32],
        a_lo: usize,
        a_hi: usize,
        b_lo: usize,
        b_hi: usize,
        scratch: &mut LongestMatchScratch,
    ) -> Match {
        let mut best = Match { a: a_lo, b: b_lo, size: 0 };
        scratch.reset();

        for (i, token) in a.iter().enumerate().take(a_hi).skip(a_lo) {
            scratch.begin_row();
            if let Some(positions) = self.occurrences.get(token) {
                for &j in positions {
                    if j < b_lo {
                        continue;
                    }
                    if j >= b_hi {
                        break;
                    }
                    let run = if j > 0 { scratch.previous_row[j - 1] } else { 0 } + 1;
                    scratch.write(j, run);
                    if run > best.size {
                        best = Match { a: i + 1 - run, b: j + 1 - run, size: run };
                    }
                }
            }
            scratch.end_row();
        }
        best
    }
}

/// Reusable row buffers for [`Matcher::find_longest_match`].
struct LongestMatchScratch {
    previous_row: Vec<usize>,
    current_row: Vec<usize>,
    previous_touched: Vec<usize>,
    current_touched: Vec<usize>,
}

impl LongestMatchScratch {
    fn new(b_len: usize) -> Self {
        LongestMatchScratch {
            previous_row: vec![0; b_len],
            current_row: vec![0; b_len],
            previous_touched: Vec::new(),
            current_touched: Vec::new(),
        }
    }

    /// Zero both rows, in time proportional to what was written.
    fn reset(&mut self) {
        for &j in &self.previous_touched {
            self.previous_row[j] = 0;
        }
        for &j in &self.current_touched {
            self.current_row[j] = 0;
        }
        self.previous_touched.clear();
        self.current_touched.clear();
    }

    fn begin_row(&mut self) {
        self.current_touched.clear();
    }

    fn write(&mut self, j: usize, run: usize) {
        self.current_row[j] = run;
        self.current_touched.push(j);
    }

    /// Promote the current row to previous, leaving a zeroed buffer behind.
    fn end_row(&mut self) {
        for &j in &self.previous_touched {
            self.previous_row[j] = 0;
        }
        std::mem::swap(&mut self.previous_row, &mut self.current_row);
        std::mem::swap(&mut self.previous_touched, &mut self.current_touched);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round to 12 decimals so computed ratios can be compared with `assert_eq!`
    /// on whole vectors instead of pairwise `assert!(… < epsilon)`, which would
    /// need a failure-only branch. Well below the precision either path loses.
    fn round12(x: f64) -> f64 {
        (x * 1e12).round() / 1e12
    }

    /// Reference implementation of the definition, used to cross-check the
    /// optimized path. Deliberately naive: O(n³), no pruning, no scratch reuse.
    fn reference_ratio(a: &[u32], b: &[u32]) -> f64 {
        fn longest(
            a: &[u32],
            b: &[u32],
            alo: usize,
            ahi: usize,
            blo: usize,
            bhi: usize,
        ) -> (usize, usize, usize) {
            let (mut bi, mut bj, mut bs) = (alo, blo, 0usize);
            for i in alo..ahi {
                for j in blo..bhi {
                    let mut k = 0;
                    while i + k < ahi && j + k < bhi && a[i + k] == b[j + k] {
                        k += 1;
                    }
                    if k > bs {
                        bi = i;
                        bj = j;
                        bs = k;
                    }
                }
            }
            (bi, bj, bs)
        }
        fn total(a: &[u32], b: &[u32], alo: usize, ahi: usize, blo: usize, bhi: usize) -> usize {
            let (i, j, k) = longest(a, b, alo, ahi, blo, bhi);
            if k == 0 {
                return 0;
            }
            k + total(a, b, alo, i, blo, j) + total(a, b, i + k, ahi, j + k, bhi)
        }
        let t = a.len() + b.len();
        if t == 0 {
            return 1.0;
        }
        2.0 * total(a, b, 0, a.len(), 0, b.len()) as f64 / t as f64
    }

    // -------------------------------------------------------- real_quick_ratio

    #[test]
    fn real_quick_ratio_of_two_empty_sequences_is_one() {
        assert_eq!(real_quick_ratio(0, 0), 1.0);
    }

    #[test]
    fn real_quick_ratio_of_equal_lengths_is_one() {
        assert_eq!(real_quick_ratio(4, 4), 1.0);
    }

    #[test]
    fn real_quick_ratio_falls_with_length_mismatch() {
        assert_eq!(real_quick_ratio(1, 3), 0.5);
    }

    // ------------------------------------------------------------- quick_ratio

    #[test]
    fn quick_ratio_of_two_empty_sequences_is_one() {
        assert_eq!(quick_ratio(&[], &[]), 1.0);
    }

    #[test]
    fn quick_ratio_ignores_order() {
        assert_eq!(quick_ratio(&[1, 2, 3], &[3, 2, 1]), 1.0);
    }

    #[test]
    fn quick_ratio_respects_multiplicity() {
        // One `1` on the right can only be matched once, however many are left.
        assert_eq!(quick_ratio(&[1, 1, 1], &[1]), 2.0 * 1.0 / 4.0);
    }

    #[test]
    fn quick_ratio_is_zero_when_nothing_is_shared() {
        assert_eq!(quick_ratio(&[1, 2], &[3, 4]), 0.0);
    }

    // ------------------------------------------------------------------- ratio

    #[test]
    fn ratio_of_two_empty_sequences_is_one() {
        assert_eq!(ratio(&[], &[]), 1.0);
    }

    #[test]
    fn ratio_of_an_empty_and_a_nonempty_sequence_is_zero() {
        assert_eq!(ratio(&[], &[1, 2]), 0.0);
    }

    #[test]
    fn ratio_of_identical_sequences_is_one() {
        assert_eq!(ratio(&[1, 2, 3, 4], &[1, 2, 3, 4]), 1.0);
    }

    #[test]
    fn ratio_of_disjoint_sequences_is_zero() {
        assert_eq!(ratio(&[1, 2, 3], &[4, 5, 6]), 0.0);
    }

    #[test]
    fn ratio_penalises_reordering_that_the_multiset_bound_cannot_see() {
        // Same tokens, same multiplicities, blocks swapped. `quick_ratio` is
        // order-blind and calls this a perfect match; the real ratio must not,
        // because "the same statements in a different order" is not the same
        // code. This gap is precisely why quick_ratio is only ever a prune.
        let a = [1u32, 2, 3, 4, 5, 6];
        let b = [4u32, 5, 6, 1, 2, 3];
        assert_eq!(quick_ratio(&a, &b), 1.0);
        assert_eq!(ratio(&a, &b), 0.5);
    }

    #[test]
    fn ratio_matches_the_reference_implementation_on_assorted_inputs() {
        let cases: &[(&[u32], &[u32])] = &[
            (&[1, 2, 3], &[1, 2, 3]),
            (&[1, 2, 3], &[1, 3, 2]),
            (&[1, 1, 1, 1], &[1, 1]),
            (&[1, 2, 3, 4, 5], &[3, 4, 5, 1, 2]),
            (&[5, 4, 3, 2, 1], &[1, 2, 3, 4, 5]),
            (&[1, 2, 2, 1, 2], &[2, 1, 2, 2, 1]),
            (&[7], &[7, 7, 7]),
            (&[1, 2, 3, 4], &[]),
            (&[], &[]),
        ];
        // Both sides are computed unconditionally and compared as whole
        // vectors. A custom `assert!` message -- or a `push` inside an
        // `if failed` -- compiles to a branch that never runs while the test
        // passes, which the 100% coverage gate correctly flags as untested.
        // This shape keeps full diagnostics with no failure-only branch, and
        // is the house pattern for every table-driven test here.
        let got: Vec<f64> = cases.iter().map(|(a, b)| round12(ratio(a, b))).collect();
        let want: Vec<f64> = cases.iter().map(|(a, b)| round12(reference_ratio(a, b))).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn ratio_is_symmetric() {
        let a = &[1u32, 2, 3, 4, 2, 1];
        let b = &[2u32, 1, 3, 4, 1, 2];
        assert!((ratio(a, b) - ratio(b, a)).abs() < 1e-12);
    }

    // ------------------------------------------------------- prune soundness

    #[test]
    fn both_upper_bounds_are_never_below_the_real_ratio() {
        // The load-bearing property: a pair pruned by a bound could never have
        // passed the real threshold. Violating this silently loses findings.
        let sequences: &[&[u32]] = &[
            &[],
            &[1],
            &[1, 2],
            &[2, 1],
            &[1, 1, 2, 2],
            &[1, 2, 1, 2, 1],
            &[3, 3, 3],
            &[1, 2, 3, 4, 5, 6],
            &[6, 5, 4, 3, 2, 1],
        ];
        // One triple per ordered pair, in a deterministic order: on failure the
        // index of the offending triple identifies the pair.
        let mut bounds_hold: Vec<(bool, bool, bool)> = Vec::new();
        for a in sequences {
            for b in sequences {
                let real = ratio(a, b);
                let quick = quick_ratio(a, b);
                let real_quick = real_quick_ratio(a.len(), b.len());
                bounds_hold.push((
                    quick >= real - 1e-12,
                    real_quick >= real - 1e-12,
                    real_quick >= quick - 1e-12,
                ));
            }
        }
        assert_eq!(bounds_hold.len(), sequences.len() * sequences.len());
        assert_eq!(bounds_hold, vec![(true, true, true); bounds_hold.len()]);
    }

    // --------------------------------------------------------- matching_blocks

    #[test]
    fn matching_blocks_reports_position_in_both_sequences() {
        let blocks = matching_blocks(&[9, 9, 1, 2], &[1, 2, 8]);
        assert_eq!(blocks, vec![Match { a: 2, b: 0, size: 2 }]);
    }

    #[test]
    fn matching_blocks_reports_a_fully_shared_sequence_as_one_run() {
        let blocks = matching_blocks(&[1, 2, 3, 4], &[1, 2, 3, 4]);
        assert_eq!(blocks, vec![Match { a: 0, b: 0, size: 4 }]);
    }

    #[test]
    fn matching_blocks_never_returns_runs_that_abut_in_both() {
        // Pins the argument in `Matcher::matching_blocks`' docs, which is what
        // licenses the absence of a merge pass. If this ever fails, the merge
        // pass has to come back -- `ratio` would still be right, but callers
        // reading blocks would see one run reported as two.
        let sequences: &[&[u32]] = &[
            &[1, 2, 3, 4, 5, 6],
            &[1, 2, 9, 3, 4],
            &[1, 1, 2, 2, 1, 1],
            &[4, 5, 1, 2, 3],
            &[1, 2, 3, 1, 2, 3, 1],
            &[7, 7, 7, 7],
        ];
        let mut abuts_in_both: Vec<bool> = Vec::new();
        for a in sequences {
            for b in sequences {
                for pair in matching_blocks(a, b).windows(2) {
                    let (l, r) = (pair[0], pair[1]);
                    abuts_in_both.push(l.a + l.size == r.a && l.b + l.size == r.b);
                }
            }
        }
        // Guards against a vacuous pass: the corpus must actually produce
        // multi-block results for the property to mean anything.
        assert!(!abuts_in_both.is_empty());
        assert_eq!(abuts_in_both, vec![false; abuts_in_both.len()]);
    }

    #[test]
    fn matching_blocks_keeps_runs_that_abut_in_only_one_sequence_separate() {
        let blocks = matching_blocks(&[1, 2, 3], &[1, 9, 2, 3]);
        assert_eq!(blocks, vec![Match { a: 0, b: 0, size: 1 }, Match { a: 1, b: 2, size: 2 }]);
    }

    #[test]
    fn matching_blocks_is_empty_when_nothing_matches() {
        assert!(matching_blocks(&[1], &[2]).is_empty());
    }

    #[test]
    fn matching_blocks_recurses_into_both_sides_of_the_longest_run() {
        // `5,6,7` is longest; `1` lies left of it and `9` right, so both
        // recursive branches must fire.
        let blocks = matching_blocks(&[1, 4, 5, 6, 7, 8, 9], &[1, 3, 5, 6, 7, 2, 9]);
        assert_eq!(
            blocks,
            vec![Match { a: 0, b: 0, size: 1 }, Match { a: 2, b: 2, size: 3 }, Match { a: 6, b: 6, size: 1 },]
        );
    }

    #[test]
    fn matching_blocks_ties_resolve_to_the_earliest_run() {
        let blocks = matching_blocks(&[1, 2, 9, 1, 2], &[1, 2]);
        assert_eq!(blocks, vec![Match { a: 0, b: 0, size: 2 }]);
    }

    #[test]
    fn match_is_copyable_and_debuggable() {
        let m = Match { a: 1, b: 2, size: 3 };
        let copy = m;
        assert_eq!(m, copy);
        assert!(format!("{m:?}").contains("size"));
    }

    // ----------------------------------------------------------------- Matcher

    #[test]
    fn matcher_exposes_the_indexed_sequence() {
        let b = [1u32, 2];
        assert_eq!(Matcher::new(&b).b(), &b);
    }

    #[test]
    fn matcher_reuse_gives_the_same_answers_as_the_free_functions() {
        let b = [1u32, 2, 3, 2, 1];
        let m = Matcher::new(&b);
        for a in [&[1u32, 2, 3][..], &[3, 2, 1][..], &[9][..], &[][..]] {
            assert_eq!(m.ratio(a), ratio(a, &b));
            assert_eq!(m.quick_ratio(a), quick_ratio(a, &b));
            assert_eq!(m.real_quick_ratio(a), real_quick_ratio(a.len(), b.len()));
            assert_eq!(m.matching_blocks(a), matching_blocks(a, &b));
        }
    }

    #[test]
    fn matcher_scratch_is_correct_when_reused_across_many_comparisons() {
        // Row buffers are reused across recursion steps and across calls; a
        // stale entry would inflate a later run's length. Compare a batch
        // against the reference to prove no leakage between calls.
        let b = [1u32, 2, 3, 1, 2, 3, 1];
        let m = Matcher::new(&b);
        let cases: &[&[u32]] =
            &[&[1, 2, 3, 1, 2, 3, 1], &[3, 1, 2], &[1, 1, 1, 1], &[2, 3, 1, 2], &[9, 9, 9]];
        let reused: Vec<f64> = cases.iter().map(|a| round12(m.ratio(a))).collect();
        let expected: Vec<f64> = cases.iter().map(|a| round12(reference_ratio(a, &b))).collect();
        assert_eq!(reused, expected);
    }

    #[test]
    fn matcher_skips_occurrences_outside_the_current_window() {
        // Forces both the `j < b_lo` continue and the `j >= b_hi` break inside
        // find_longest_match: token 1 occurs at both ends of `b`, while the
        // recursion narrows the window to the middle.
        let a = [1u32, 5, 6, 7, 1];
        let b = [1u32, 8, 5, 6, 7, 8, 1];
        assert!((ratio(&a, &b) - reference_ratio(&a, &b)).abs() < 1e-12);
    }
}
