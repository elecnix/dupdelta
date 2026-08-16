//! The delta engine: what a change introduced, relative to its merge-base.
//!
//! Every other layer in this crate answers "how much duplication is in this
//! tree?" — a large number, mostly pre-existing, mostly already accepted by
//! the time anyone reads it, and barely different between one commit and the
//! next. A CI check built directly on that number warns about the same
//! findings on every pull request and gets muted within a week, at which
//! point it protects nothing.
//!
//! This module asks the question that is actually actionable: **did this
//! diff make it worse?** It never re-scans anything. It loads a [`Report`]
//! computed from the branch and one computed from the merge-base, and reports
//! only what the first has that the second did not.
//!
//! # Why a delta beats a committed baseline or an allowlist file
//!
//! The alternative design — a checked-in file of "known, accepted" findings,
//! snippet- or pair-keyed, that new findings get diffed against — sounds
//! equivalent, but it isn't, for two reasons:
//!
//! - **Clone findings are combinatorial.** An unrelated edit to *either* side
//!   of a pair (a rename, a reformat, a line inserted above it) changes that
//!   pair's identity or position without changing whether it is duplication.
//!   A baseline keyed on anything less stable than a content hash goes stale
//!   on nearly every PR that touches either file, so someone has to
//!   regenerate it — and the moment regenerating the baseline is a step in
//!   "make CI green" rather than "I looked at this and accepted it", the file
//!   has become a rubber stamp instead of a record of review.
//! - **The merge-base delta needs no human-maintained state at all.** There
//!   is nothing to regenerate, nothing to go stale, nothing to accidentally
//!   commit through a moment of impatience. "Does the branch's report have a
//!   finding the merge-base's report didn't?" is the right question on every
//!   single PR, for free, because both inputs are freshly computed and
//!   [`ClonePair::key`], [`VocabPair::key`] and [`BlockPair::key`] are
//!   already location-independent — see `report.rs` for why identity is a
//!   content hash and not a `(file, line)` pair.
//!
//! # Known limitation: vocabulary identity is file-based
//!
//! [`VocabPair::key`] is keyed on the two file paths, not a content hash,
//! because a module has no single body to hash the way a function or a code
//! fragment does. The consequence — documented on that method, and repeated
//! here because it is the one place this module's honesty could quietly slip
//! — is that **renaming a file makes its vocabulary pairs look new.** This
//! module does not try to paper over that with a fuzzy match; a false "new"
//! finding on a rename is a nuisance a reviewer can dismiss in one glance,
//! which is a far cheaper failure mode than the alternative of a fuzzy match
//! that occasionally, silently, matches the wrong pair and goes quiet on
//! real new duplication.

use std::collections::{HashMap, HashSet};

use crate::annotate::{Annotation, Summary};
use crate::report::{BlockPair, BlockRef, ClonePair, Report, UnitRef, VocabPair};

/// Why a vocabulary pair is being reported.
#[derive(Debug, Clone, PartialEq)]
pub enum VocabChange {
    /// The pair did not exist in the base report at all.
    New,
    /// The pair existed, but neither side had lost its inbound imports before.
    BecameUnreferenced,
    /// The pair existed and its overlap grew materially.
    Worsened {
        /// Overlap in the base report.
        from: f64,
        /// Overlap in the head report.
        to: f64,
    },
}

/// A vocabulary finding, with the reason it is being reported.
#[derive(Debug, Clone, PartialEq)]
pub struct VocabFinding {
    /// Why this pair is being reported.
    pub change: VocabChange,
    /// The pair itself, as it reads in the head report.
    pub pair: VocabPair,
}

/// What a change introduced, relative to its merge-base.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Delta {
    /// Clone pairs present in the head report but absent from the base.
    pub new_clones: Vec<ClonePair>,
    /// Vocabulary pairs that are new, newly unreferenced, or worsened.
    pub vocab: Vec<VocabFinding>,
    /// Block pairs present in the head report but absent from the base.
    pub new_blocks: Vec<BlockPair>,
}

/// Tuning for [`Delta::compute`].
#[derive(Debug, Clone)]
pub struct DeltaOptions {
    /// Only report clone pairs at or above this similarity.
    pub min_similarity: f64,
    /// How much an existing vocabulary pair's overlap must grow to count as
    /// worsened. A growth of exactly this amount is still noise; the excess
    /// must be strictly greater.
    pub worsened_delta: f64,
    /// Cap on findings reported per category. `None` means no cap. `Some(0)`
    /// means report nothing in that category — zero is a real value here,
    /// not a stand-in for "unlimited".
    pub max_findings: Option<usize>,
}

/// Truncate `items` to `max`, in place. `None` leaves it untouched.
///
/// A free function rather than three inlined copies of `if let Some ...`,
/// because the "`Some(0)` truncates to nothing, `None` does not truncate at
/// all" behaviour is exactly the kind of thing that is easy to get backwards
/// once and never notice, and one place is one place to get it right.
fn truncate<T>(items: &mut Vec<T>, max: Option<usize>) {
    if let Some(max) = max {
        items.truncate(max);
    }
}

/// Decide whether a vocabulary pair that exists in *both* reports should be
/// reported, and why. `None` means nothing material changed and the pair
/// stays silent. (A pair absent from the base entirely is [`VocabChange::New`]
/// unconditionally — that case is handled by the caller, before this
/// function is reached, precisely so this function's `None` has exactly one
/// meaning: "present in both, unchanged".)
///
/// "Became unreferenced" is checked ahead of "worsened": a pair that lost
/// its last importer *and* grew its overlap in the same diff is far more
/// actionable framed as "this is now dead code duplicating a live module's
/// vocabulary" than as a percentage change.
fn vocab_change_for_existing(pair: &VocabPair, base: &VocabPair, worsened_delta: f64) -> Option<VocabChange> {
    if pair.zero_inbound && !base.zero_inbound {
        return Some(VocabChange::BecameUnreferenced);
    }
    if pair.overlap > base.overlap + worsened_delta {
        return Some(VocabChange::Worsened { from: base.overlap, to: pair.overlap });
    }
    None
}

/// One annotation for one side of a two-location finding (a clone or a block
/// pair): anchored on `this` side, describing `other`.
fn side_annotation(file: &str, line: usize, message: String) -> Annotation {
    Annotation::warning(file.to_string(), Some(line), message)
}

fn clone_side_message(similarity: f64, other: &UnitRef) -> String {
    format!(
        "{:.0}% duplicate of `{}` at {}:{}-{}",
        similarity * 100.0,
        other.qualname,
        other.file,
        other.start_line,
        other.end_line
    )
}

fn block_side_message(tokens: usize, other: &BlockRef) -> String {
    format!("{tokens} normalized tokens duplicated at {}:{}-{}", other.file, other.start_line, other.end_line)
}

fn vocab_message(change: &VocabChange, pair: &VocabPair, other_file: &str) -> String {
    match change {
        VocabChange::New => {
            format!(
                "new vocabulary overlap with {other_file}: {:.0}% ({} shared identifiers)",
                pair.overlap * 100.0,
                pair.shared
            )
        }
        VocabChange::BecameUnreferenced => {
            format!(
                "vocabulary overlap with {other_file} ({:.0}%) and neither side has inbound imports anymore",
                pair.overlap * 100.0
            )
        }
        VocabChange::Worsened { from, to } => {
            format!(
                "vocabulary overlap with {other_file} grew from {:.0}% to {:.0}%",
                from * 100.0,
                to * 100.0
            )
        }
    }
}

fn vocab_change_label(change: &VocabChange) -> &'static str {
    match change {
        VocabChange::New => "new",
        VocabChange::BecameUnreferenced => "became unreferenced",
        VocabChange::Worsened { .. } => "worsened",
    }
}

impl Delta {
    /// Compute what `head` has that `base` did not.
    ///
    /// `head` and `base` are used exactly as given — the caller is expected
    /// to have loaded, not re-scanned, both. Findings are reported in
    /// strongest-first order: this clones `head` and calls [`Report::sort`]
    /// on the clone internally, rather than trusting that the caller already
    /// sorted it, so `max_findings` truncation is correct even when it
    /// wasn't.
    pub fn compute(head: &Report, base: &Report, options: &DeltaOptions) -> Delta {
        let mut head = head.clone();
        head.sort();

        let base_clone_keys: HashSet<_> = base.clones.iter().map(ClonePair::key).collect();
        let base_block_keys: HashSet<_> = base.blocks.iter().map(BlockPair::key).collect();
        let base_vocab: HashMap<_, &VocabPair> = base.vocab.iter().map(|pair| (pair.key(), pair)).collect();

        let mut new_clones: Vec<ClonePair> = head
            .clones
            .into_iter()
            .filter(|pair| {
                pair.similarity >= options.min_similarity && !base_clone_keys.contains(&pair.key())
            })
            .collect();
        truncate(&mut new_clones, options.max_findings);

        let mut vocab: Vec<VocabFinding> = head
            .vocab
            .into_iter()
            .filter_map(|pair| {
                let change = match base_vocab.get(&pair.key()) {
                    None => VocabChange::New,
                    Some(base_pair) => vocab_change_for_existing(&pair, base_pair, options.worsened_delta)?,
                };
                Some(VocabFinding { change, pair })
            })
            .collect();
        truncate(&mut vocab, options.max_findings);

        let mut new_blocks: Vec<BlockPair> =
            head.blocks.into_iter().filter(|pair| !base_block_keys.contains(&pair.key())).collect();
        truncate(&mut new_blocks, options.max_findings);

        Delta { new_clones, vocab, new_blocks }
    }

    /// Whether nothing new was found in any category.
    pub fn is_empty(&self) -> bool {
        self.new_clones.is_empty() && self.vocab.is_empty() && self.new_blocks.is_empty()
    }

    /// Total findings across all three categories.
    pub fn finding_count(&self) -> usize {
        self.new_clones.len() + self.vocab.len() + self.new_blocks.len()
    }

    /// One annotation per vocabulary finding; clone and block pairs each
    /// produce two, one anchored on each side, because a reviewer standing
    /// on either file needs the other file's location to judge the finding,
    /// and GitHub only renders an annotation on the file it names.
    pub fn annotations(&self) -> Vec<Annotation> {
        let mut out =
            Vec::with_capacity(2 * self.new_clones.len() + self.vocab.len() + 2 * self.new_blocks.len());

        for pair in &self.new_clones {
            out.push(side_annotation(
                &pair.a.file,
                pair.a.start_line,
                clone_side_message(pair.similarity, &pair.b),
            ));
            out.push(side_annotation(
                &pair.b.file,
                pair.b.start_line,
                clone_side_message(pair.similarity, &pair.a),
            ));
        }

        for finding in &self.vocab {
            let pair = &finding.pair;
            out.push(Annotation::warning(
                pair.a.clone(),
                None,
                vocab_message(&finding.change, pair, &pair.b),
            ));
        }

        for pair in &self.new_blocks {
            out.push(side_annotation(
                &pair.a.file,
                pair.a.start_line,
                block_side_message(pair.tokens, &pair.b),
            ));
            out.push(side_annotation(
                &pair.b.file,
                pair.b.start_line,
                block_side_message(pair.tokens, &pair.a),
            ));
        }

        out
    }

    /// A markdown digest of the whole delta, for the GitHub Actions job
    /// summary. One table per non-empty category; empty categories are
    /// omitted rather than rendered as an empty table nobody needs to see.
    pub fn summary(&self) -> Summary {
        let mut summary = Summary::new();
        summary.heading(2, "Duplication delta");

        if self.is_empty() {
            summary.paragraph("No new duplication vs the merge-base.");
            return summary;
        }

        summary.paragraph(
            "New duplication introduced by this change, relative to its merge-base. Nothing here \
             blocks a merge: extract the shared logic where that makes sense, or leave it if the \
             similarity is coincidental.",
        );

        if !self.new_clones.is_empty() {
            summary.heading(3, "New clone pairs");
            let rows: Vec<Vec<String>> = self
                .new_clones
                .iter()
                .map(|pair| {
                    vec![
                        format!("{:.0}%", pair.similarity * 100.0),
                        format!(
                            "{}:{}-{} (`{}`)",
                            pair.a.file, pair.a.start_line, pair.a.end_line, pair.a.qualname
                        ),
                        format!(
                            "{}:{}-{} (`{}`)",
                            pair.b.file, pair.b.start_line, pair.b.end_line, pair.b.qualname
                        ),
                    ]
                })
                .collect();
            summary.table(&["Similarity", "A", "B"], &rows);
        }

        if !self.vocab.is_empty() {
            summary.heading(3, "Vocabulary findings");
            let rows: Vec<Vec<String>> = self
                .vocab
                .iter()
                .map(|finding| {
                    vec![
                        vocab_change_label(&finding.change).to_string(),
                        finding.pair.a.clone(),
                        finding.pair.b.clone(),
                        format!("{:.0}%", finding.pair.overlap * 100.0),
                    ]
                })
                .collect();
            summary.table(&["Change", "A", "B", "Overlap"], &rows);
        }

        if !self.new_blocks.is_empty() {
            summary.heading(3, "New duplicated blocks");
            let rows: Vec<Vec<String>> = self
                .new_blocks
                .iter()
                .map(|pair| {
                    vec![
                        format!("{} tokens", pair.tokens),
                        format!("{}:{}-{}", pair.a.file, pair.a.start_line, pair.a.end_line),
                        format!("{}:{}-{}", pair.b.file, pair.b.start_line, pair.b.end_line),
                    ]
                })
                .collect();
            summary.table(&["Size", "A", "B"], &rows);
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::ContentHash;

    // -------------------------------------------------------------- fixtures

    fn unit(file: &str, qualname: &str, token: &str, start_line: usize, end_line: usize) -> UnitRef {
        UnitRef {
            file: file.to_string(),
            qualname: qualname.to_string(),
            start_line,
            end_line,
            hash: ContentHash::of(&[token]),
        }
    }

    fn clone_pair(similarity: f64, a: UnitRef, b: UnitRef) -> ClonePair {
        ClonePair { similarity, a, b }
    }

    fn vocab_pair(a: &str, b: &str, overlap: f64, zero_inbound: bool) -> VocabPair {
        VocabPair {
            a: a.to_string(),
            b: b.to_string(),
            overlap,
            shared: 12,
            a_vocabulary: 40,
            b_vocabulary: 30,
            a_inbound_imports: if zero_inbound { 0 } else { 3 },
            b_inbound_imports: 3,
            zero_inbound,
            sample_shared: vec!["rate".to_string()],
        }
    }

    fn block_ref(file: &str, start_line: usize, end_line: usize) -> BlockRef {
        BlockRef { file: file.to_string(), start_line, end_line }
    }

    fn block_pair(token: &str, tokens: usize, a: BlockRef, b: BlockRef) -> BlockPair {
        BlockPair { a, b, tokens, hash: ContentHash::of(&[token]) }
    }

    fn options() -> DeltaOptions {
        DeltaOptions { min_similarity: 0.0, worsened_delta: 0.05, max_findings: None }
    }

    fn report(clones: Vec<ClonePair>, vocab: Vec<VocabPair>, blocks: Vec<BlockPair>) -> Report {
        Report { clones, vocab, blocks, ..Report::default() }
    }

    // ------------------------------------------------------------ rule 1: clones

    #[test]
    fn a_clone_pair_absent_from_base_is_new() {
        let head = report(
            vec![clone_pair(0.9, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vec![],
            vec![],
        );
        let base = Report::default();
        let delta = Delta::compute(&head, &base, &options());
        assert_eq!(delta.new_clones.len(), 1);
    }

    #[test]
    fn a_clone_pair_that_moved_files_and_lines_stays_silent() {
        // Same content hashes, different files and line numbers: this is
        // exactly what content-hash identity is for.
        let base_pair = clone_pair(0.9, unit("old_a.py", "f", "f", 1, 5), unit("old_b.py", "g", "g", 10, 20));
        let head_pair = clone_pair(
            0.9,
            unit("new_a.py", "f_renamed", "f", 100, 104),
            unit("new_b.py", "g_renamed", "g", 200, 210),
        );
        let head = report(vec![head_pair], vec![], vec![]);
        let base = report(vec![base_pair], vec![], vec![]);
        let delta = Delta::compute(&head, &base, &options());
        assert!(delta.new_clones.is_empty());
    }

    #[test]
    fn a_clone_pair_below_min_similarity_is_not_reported_even_when_new() {
        let head = report(
            vec![clone_pair(0.5, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vec![],
            vec![],
        );
        let base = Report::default();
        let opts = DeltaOptions { min_similarity: 0.8, ..options() };
        let delta = Delta::compute(&head, &base, &opts);
        assert!(delta.new_clones.is_empty());
    }

    #[test]
    fn a_clone_pair_at_exactly_min_similarity_is_reported() {
        let head = report(
            vec![clone_pair(0.8, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vec![],
            vec![],
        );
        let base = Report::default();
        let opts = DeltaOptions { min_similarity: 0.8, ..options() };
        let delta = Delta::compute(&head, &base, &opts);
        assert_eq!(delta.new_clones.len(), 1);
    }

    // ---------------------------------------------------------- rule 2: vocab

    #[test]
    fn a_vocab_pair_absent_from_base_is_new() {
        let head = report(vec![], vec![vocab_pair("a.py", "b.py", 0.4, false)], vec![]);
        let base = Report::default();
        let delta = Delta::compute(&head, &base, &options());
        assert_eq!(
            delta.vocab,
            vec![VocabFinding { change: VocabChange::New, pair: vocab_pair("a.py", "b.py", 0.4, false) }]
        );
    }

    #[test]
    fn a_vocab_pair_that_lost_its_last_importer_became_unreferenced() {
        let base_pair = vocab_pair("a.py", "b.py", 0.4, false);
        let head_pair = vocab_pair("a.py", "b.py", 0.4, true);
        let head = report(vec![], vec![head_pair.clone()], vec![]);
        let base = report(vec![], vec![base_pair], vec![]);
        let delta = Delta::compute(&head, &base, &options());
        assert_eq!(
            delta.vocab,
            vec![VocabFinding { change: VocabChange::BecameUnreferenced, pair: head_pair }]
        );
    }

    #[test]
    fn a_vocab_pair_that_was_already_unreferenced_is_not_reported_again() {
        let base_pair = vocab_pair("a.py", "b.py", 0.4, true);
        let head_pair = vocab_pair("a.py", "b.py", 0.4, true);
        let head = report(vec![], vec![head_pair], vec![]);
        let base = report(vec![], vec![base_pair], vec![]);
        let delta = Delta::compute(&head, &base, &options());
        assert!(delta.vocab.is_empty());
    }

    #[test]
    fn a_vocab_overlap_growth_of_exactly_worsened_delta_is_noise() {
        let base_pair = vocab_pair("a.py", "b.py", 0.40, false);
        let head_pair = vocab_pair("a.py", "b.py", 0.45, false); // +0.05, equals worsened_delta
        let head = report(vec![], vec![head_pair], vec![]);
        let base = report(vec![], vec![base_pair], vec![]);
        let delta = Delta::compute(&head, &base, &options());
        assert!(delta.vocab.is_empty());
    }

    #[test]
    fn a_vocab_overlap_growth_beyond_worsened_delta_is_worsened() {
        let base_pair = vocab_pair("a.py", "b.py", 0.40, false);
        let head_pair = vocab_pair("a.py", "b.py", 0.46, false); // +0.06 > worsened_delta
        let head = report(vec![], vec![head_pair.clone()], vec![]);
        let base = report(vec![], vec![base_pair], vec![]);
        let delta = Delta::compute(&head, &base, &options());
        assert_eq!(
            delta.vocab,
            vec![VocabFinding { change: VocabChange::Worsened { from: 0.40, to: 0.46 }, pair: head_pair }]
        );
    }

    #[test]
    fn an_unchanged_vocab_pair_is_silent() {
        let pair = vocab_pair("a.py", "b.py", 0.4, false);
        let head = report(vec![], vec![pair.clone()], vec![]);
        let base = report(vec![], vec![pair], vec![]);
        let delta = Delta::compute(&head, &base, &options());
        assert!(delta.vocab.is_empty());
    }

    // ---------------------------------------------------------- rule 3: blocks

    #[test]
    fn a_block_pair_absent_from_base_is_new() {
        let head = report(
            vec![],
            vec![],
            vec![block_pair("frag", 50, block_ref("a.py", 1, 5), block_ref("b.py", 10, 14))],
        );
        let base = Report::default();
        let delta = Delta::compute(&head, &base, &options());
        assert_eq!(delta.new_blocks.len(), 1);
    }

    #[test]
    fn a_block_pair_that_moved_lines_stays_silent() {
        let base_pair = block_pair("frag", 50, block_ref("a.py", 1, 5), block_ref("b.py", 10, 14));
        let head_pair = block_pair("frag", 50, block_ref("a.py", 100, 104), block_ref("b.py", 200, 204));
        let head = report(vec![], vec![], vec![head_pair]);
        let base = report(vec![], vec![], vec![base_pair]);
        let delta = Delta::compute(&head, &base, &options());
        assert!(delta.new_blocks.is_empty());
    }

    // ------------------------------------------------------------ rule 4: cap

    fn three_clones() -> Vec<ClonePair> {
        vec![
            clone_pair(0.99, unit("a.py", "f1", "f1", 1, 5), unit("b.py", "g1", "g1", 1, 5)),
            clone_pair(0.95, unit("a.py", "f2", "f2", 1, 5), unit("b.py", "g2", "g2", 1, 5)),
            clone_pair(0.90, unit("a.py", "f3", "f3", 1, 5), unit("b.py", "g3", "g3", 1, 5)),
        ]
    }

    #[test]
    fn max_findings_none_reports_everything() {
        let head = report(three_clones(), vec![], vec![]);
        let delta =
            Delta::compute(&head, &Report::default(), &DeltaOptions { max_findings: None, ..options() });
        assert_eq!(delta.new_clones.len(), 3);
    }

    #[test]
    fn max_findings_some_zero_reports_nothing() {
        let head = report(three_clones(), vec![], vec![]);
        let delta =
            Delta::compute(&head, &Report::default(), &DeltaOptions { max_findings: Some(0), ..options() });
        assert!(delta.new_clones.is_empty());
    }

    #[test]
    fn max_findings_truncation_keeps_the_strongest_findings() {
        // Deliberately unsorted input: compute() must sort before truncating.
        let mut clones = three_clones();
        clones.reverse();
        let head = report(clones, vec![], vec![]);
        let delta =
            Delta::compute(&head, &Report::default(), &DeltaOptions { max_findings: Some(2), ..options() });
        let similarities: Vec<f64> = delta.new_clones.iter().map(|c| c.similarity).collect();
        assert_eq!(similarities, vec![0.99, 0.95]);
    }

    // -------------------------------------------------------------- rule 5: empty

    #[test]
    fn identical_reports_produce_an_empty_delta() {
        let head = report(
            three_clones(),
            vec![vocab_pair("a.py", "b.py", 0.4, false)],
            vec![block_pair("frag", 50, block_ref("a.py", 1, 5), block_ref("b.py", 10, 14))],
        );
        let base = head.clone();
        let delta = Delta::compute(&head, &base, &options());
        assert!(delta.is_empty());
        assert_eq!(delta.finding_count(), 0);
    }

    #[test]
    fn an_empty_deltas_summary_says_so_in_words() {
        let delta = Delta::compute(&Report::default(), &Report::default(), &options());
        assert!(delta.summary().render().contains("No new duplication vs the merge-base."));
    }

    // ------------------------------------------------------- rule 6: removals

    #[test]
    fn a_finding_present_only_in_base_is_not_reported() {
        let head = Report::default();
        let base = report(
            three_clones(),
            vec![vocab_pair("a.py", "b.py", 0.4, false)],
            vec![block_pair("frag", 50, block_ref("a.py", 1, 5), block_ref("b.py", 10, 14))],
        );
        let delta = Delta::compute(&head, &base, &options());
        assert!(delta.is_empty());
    }

    // ----------------------------------------------------------- rule 7: annotate

    #[test]
    fn a_new_clone_pair_annotates_both_sides_with_similarity_and_the_other_location() {
        let pair = clone_pair(0.87, unit("a.py", "f", "f", 3, 9), unit("b.py", "g", "g", 40, 46));
        let delta = Delta { new_clones: vec![pair], vocab: vec![], new_blocks: vec![] };
        let annotations = delta.annotations();
        assert_eq!(annotations.len(), 2);
        assert_eq!(annotations[0].file, "a.py");
        assert_eq!(annotations[0].start_line, Some(3));
        assert!(annotations[0].message.contains("87%"));
        assert!(annotations[0].message.contains("b.py:40-46"));
        assert_eq!(annotations[1].file, "b.py");
        assert_eq!(annotations[1].start_line, Some(40));
        assert!(annotations[1].message.contains("a.py:3-9"));
    }

    #[test]
    fn a_new_block_pair_annotates_both_sides_with_token_count_and_the_other_location() {
        let pair = block_pair("frag", 64, block_ref("a.py", 3, 9), block_ref("b.py", 40, 46));
        let delta = Delta { new_clones: vec![], vocab: vec![], new_blocks: vec![pair] };
        let annotations = delta.annotations();
        assert_eq!(annotations.len(), 2);
        assert_eq!(annotations[0].file, "a.py");
        assert!(annotations[0].message.contains("64 normalized tokens"));
        assert!(annotations[0].message.contains("b.py:40-46"));
        assert_eq!(annotations[1].file, "b.py");
        assert!(annotations[1].message.contains("a.py:3-9"));
    }

    #[test]
    fn a_vocab_finding_annotates_the_a_side_with_the_b_files_location() {
        let finding =
            VocabFinding { change: VocabChange::New, pair: vocab_pair("a.py", "b.py", 0.42, false) };
        let delta = Delta { new_clones: vec![], vocab: vec![finding], new_blocks: vec![] };
        let annotations = delta.annotations();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].file, "a.py");
        assert_eq!(annotations[0].start_line, None);
        assert!(annotations[0].message.contains("b.py"));
        assert!(annotations[0].message.contains("42%"));
    }

    #[test]
    fn vocab_annotation_messages_differ_by_change_reason() {
        let unreferenced = VocabFinding {
            change: VocabChange::BecameUnreferenced,
            pair: vocab_pair("a.py", "b.py", 0.5, true),
        };
        let worsened = VocabFinding {
            change: VocabChange::Worsened { from: 0.3, to: 0.5 },
            pair: vocab_pair("a.py", "b.py", 0.5, false),
        };
        let delta = Delta { new_clones: vec![], vocab: vec![unreferenced, worsened], new_blocks: vec![] };
        let annotations = delta.annotations();
        assert!(annotations[0].message.contains("inbound imports"));
        assert!(annotations[1].message.contains("30%"));
        assert!(annotations[1].message.contains("50%"));
    }

    #[test]
    fn annotations_are_empty_when_the_delta_is_empty() {
        assert!(Delta::default().annotations().is_empty());
    }

    // ------------------------------------------------------------- rule 8: summary

    #[test]
    fn summary_renders_a_table_per_non_empty_category_and_omits_empty_ones() {
        let delta = Delta {
            new_clones: vec![clone_pair(0.9, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vocab: vec![],
            new_blocks: vec![],
        };
        let rendered = delta.summary().render();
        assert!(rendered.contains("New clone pairs"));
        assert!(!rendered.contains("Vocabulary findings"));
        assert!(!rendered.contains("New duplicated blocks"));
    }

    #[test]
    fn summary_advises_extracting_or_leaving_the_duplication_and_blocks_nothing() {
        let delta = Delta {
            new_clones: vec![clone_pair(0.9, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vocab: vec![],
            new_blocks: vec![],
        };
        let rendered = delta.summary().render();
        assert!(rendered.contains("extract"));
        assert!(rendered.contains("leave it"));
    }

    #[test]
    fn summary_vocab_table_labels_became_unreferenced_and_worsened_reasons() {
        let delta = Delta {
            new_clones: vec![],
            vocab: vec![
                VocabFinding {
                    change: VocabChange::BecameUnreferenced,
                    pair: vocab_pair("a.py", "b.py", 0.5, true),
                },
                VocabFinding {
                    change: VocabChange::Worsened { from: 0.3, to: 0.5 },
                    pair: vocab_pair("c.py", "d.py", 0.5, false),
                },
            ],
            new_blocks: vec![],
        };
        let rendered = delta.summary().render();
        assert!(!rendered.contains("New clone pairs"));
        assert!(rendered.contains("became unreferenced"));
        assert!(rendered.contains("worsened"));
    }

    #[test]
    fn summary_with_all_three_categories_renders_all_three_tables() {
        let delta = Delta {
            new_clones: vec![clone_pair(0.9, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vocab: vec![VocabFinding {
                change: VocabChange::New,
                pair: vocab_pair("a.py", "b.py", 0.4, false),
            }],
            new_blocks: vec![block_pair("frag", 50, block_ref("a.py", 1, 5), block_ref("b.py", 10, 14))],
        };
        let rendered = delta.summary().render();
        assert!(rendered.contains("New clone pairs"));
        assert!(rendered.contains("Vocabulary findings"));
        assert!(rendered.contains("New duplicated blocks"));
    }

    // ------------------------------------------------------------------ plumbing

    #[test]
    fn finding_count_totals_all_three_categories() {
        let delta = Delta {
            new_clones: vec![clone_pair(0.9, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vocab: vec![VocabFinding {
                change: VocabChange::New,
                pair: vocab_pair("a.py", "b.py", 0.4, false),
            }],
            new_blocks: vec![block_pair("frag", 50, block_ref("a.py", 1, 5), block_ref("b.py", 10, 14))],
        };
        assert_eq!(delta.finding_count(), 3);
    }

    #[test]
    fn delta_and_vocab_types_clone_compare_and_debug() {
        let delta = Delta {
            new_clones: vec![],
            vocab: vec![VocabFinding {
                change: VocabChange::Worsened { from: 0.1, to: 0.2 },
                pair: vocab_pair("a.py", "b.py", 0.2, false),
            }],
            new_blocks: vec![],
        };
        assert_eq!(delta.clone(), delta);
        assert!(format!("{delta:?}").contains("Delta"));
        assert!(format!("{:?}", delta.vocab[0].change).contains("Worsened"));
        assert_eq!(VocabChange::New, VocabChange::New);
        assert_ne!(VocabChange::New, VocabChange::BecameUnreferenced);
    }

    #[test]
    fn a_default_delta_is_empty() {
        assert!(Delta::default().is_empty());
    }

    #[test]
    fn delta_options_are_cloneable_and_debuggable() {
        let opts = options();
        let cloned = opts.clone();
        assert!(format!("{cloned:?}").contains("min_similarity"));
    }
}
