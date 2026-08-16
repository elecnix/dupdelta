//! The scan report — the contract between detectors and the delta engine.
//!
//! A report is what one scan of one tree produced. The delta engine never
//! re-scans anything; it loads two reports, one from the branch and one from
//! the merge-base, and asks what is in the first that was not in the second.
//! Everything else in this crate exists to fill a report or to read one.
//!
//! Because it is a file format, this is also the extension point: anything that
//! can emit this JSON participates in the delta machinery, whatever language it
//! is written in and whatever technique it uses to find duplication.
//!
//! # Identity is a content hash, never a location
//!
//! Every finding carries a [`ContentHash`] of the normalized code it describes.
//! That is what makes "already existed, just moved" distinguishable from
//! "genuinely new". Matching on `(file, line)` instead would re-report an
//! untouched duplicate the moment anything above it shifted the line numbers,
//! and a report that cries wolf on every unrelated edit gets muted — at which
//! point the tool has no value left.
//!
//! Block findings are where this matters most: token-based detectors
//! traditionally have no stable per-fragment identity, so they fall back to
//! "was this file touched?" and nag about the same untouched fragment forever.
//! Hashing the normalized fragment removes that limitation.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::token::ContentHash;

/// Version of the report format.
///
/// Bumped on any incompatible change. [`Report::from_json`] refuses a version
/// it does not know rather than reading it optimistically: a delta computed
/// from a misread report would be silently wrong, which is worse than an error.
pub const REPORT_VERSION: u32 = 1;

/// Where a unit lives, and what it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitRef {
    /// Path relative to the scanned tree's root, using `/` separators.
    pub file: String,
    /// Dotted path to the unit inside the file.
    pub qualname: String,
    /// First line, 1-based.
    pub start_line: usize,
    /// Last line, 1-based and inclusive.
    pub end_line: usize,
    /// Location-independent identity of the unit's normalized body.
    pub hash: ContentHash,
}

/// Two units similar enough to report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClonePair {
    /// Ratcliff–Obershelp similarity, in `0.0..=1.0`.
    pub similarity: f64,
    /// One side.
    pub a: UnitRef,
    /// The other side.
    pub b: UnitRef,
}

impl ClonePair {
    /// Order-independent identity of the pair.
    ///
    /// A pair is the same pair whichever side got listed first, so the key
    /// sorts the two hashes. Without this, swapping the sides between two scans
    /// would make an unchanged pair look new.
    pub fn key(&self) -> (ContentHash, ContentHash) {
        let (a, b) = (self.a.hash.clone(), self.b.hash.clone());
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }
}

/// Two files sharing an unusual amount of identifier vocabulary.
///
/// This catches what function-level comparison structurally cannot: a module
/// that reimplements another's rules with a completely different set of
/// functions. It scores near zero on any clone metric while sharing the domain
/// nouns that gave the original module its meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocabPair {
    /// One file.
    pub a: String,
    /// The other file.
    pub b: String,
    /// Shared identifiers over the smaller vocabulary, in `0.0..=1.0`.
    pub overlap: f64,
    /// Number of identifiers common to both.
    pub shared: usize,
    /// Distinct identifiers in `a`.
    pub a_vocabulary: usize,
    /// Distinct identifiers in `b`.
    pub b_vocabulary: usize,
    /// How many other files import `a`.
    pub a_inbound_imports: usize,
    /// How many other files import `b`.
    pub b_inbound_imports: usize,
    /// Whether either side has no inbound imports at all.
    ///
    /// Heavy vocabulary overlap *and* nothing importing it is the signature of
    /// a module that duplicates a live one's concepts while nothing reaches it.
    pub zero_inbound: bool,
    /// A sample of the shared identifiers, for the reader to judge by.
    pub sample_shared: Vec<String>,
}

impl VocabPair {
    /// Order-independent identity of the pair.
    ///
    /// Files, not hashes: a module has no single normalized body to hash. The
    /// consequence is that renaming a file makes its pairs look new — a known
    /// limitation, documented rather than papered over.
    pub fn key(&self) -> (String, String) {
        let (a, b) = (self.a.clone(), self.b.clone());
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }
}

/// Where a duplicated fragment sits inside a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockRef {
    /// Path relative to the scanned tree's root.
    pub file: String,
    /// First line, 1-based.
    pub start_line: usize,
    /// Last line, 1-based and inclusive.
    pub end_line: usize,
}

/// A run of code repeated in two places, possibly inside larger, differently
/// shaped functions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockPair {
    /// One occurrence.
    pub a: BlockRef,
    /// The other occurrence.
    pub b: BlockRef,
    /// Length of the repeated run, in normalized tokens.
    pub tokens: usize,
    /// Identity of the repeated fragment. Both sides share it, by definition.
    pub hash: ContentHash,
}

impl BlockPair {
    /// Identity of the finding: the fragment, plus which two files hold it.
    ///
    /// The fragment hash alone would collapse every occurrence of a widely
    /// repeated idiom into one finding; including the files keeps each pair
    /// distinct while staying independent of line numbers.
    pub fn key(&self) -> (ContentHash, String, String) {
        let (a, b) = (self.a.file.clone(), self.b.file.clone());
        let (first, second) = if a <= b { (a, b) } else { (b, a) };
        (self.hash.clone(), first, second)
    }
}

/// Everything one scan of one tree found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Format version; see [`REPORT_VERSION`].
    pub version: u32,
    /// How many files were parsed.
    pub files_scanned: usize,
    /// How many units cleared the size threshold and were compared.
    pub units_considered: usize,
    /// Files the grammar reported syntax errors in, sorted and deduplicated.
    ///
    /// Not fatal, but load-bearing: a scan of a tree it cannot parse finds no
    /// duplication and looks identical to a scan of a clean one.
    pub files_with_syntax_errors: Vec<String>,
    /// Function-level near-duplicate pairs.
    pub clones: Vec<ClonePair>,
    /// Module-level vocabulary-overlap pairs.
    pub vocab: Vec<VocabPair>,
    /// Sub-function repeated fragments.
    pub blocks: Vec<BlockPair>,
}

impl Default for Report {
    fn default() -> Self {
        Report {
            version: REPORT_VERSION,
            files_scanned: 0,
            units_considered: 0,
            files_with_syntax_errors: Vec::new(),
            clones: Vec::new(),
            vocab: Vec::new(),
            blocks: Vec::new(),
        }
    }
}

/// Why a report could not be read or written.
#[derive(Debug)]
pub enum ReportError {
    /// The file could not be read or written.
    Io {
        /// Path involved.
        path: PathBuf,
        /// Underlying cause.
        source: std::io::Error,
    },
    /// The JSON was malformed.
    Malformed {
        /// Path involved, if the report came from a file.
        path: Option<PathBuf>,
        /// Underlying cause.
        source: serde_json::Error,
    },
    /// The report announced a format version this build cannot read.
    UnsupportedVersion {
        /// Version the report claims.
        found: u32,
        /// Version this build understands.
        expected: u32,
    },
}

impl std::fmt::Display for ReportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportError::Io { path, source } => {
                write!(f, "could not access report {}: {source}", path.display())
            }
            ReportError::Malformed { path: Some(path), source } => {
                write!(f, "malformed report {}: {source}", path.display())
            }
            ReportError::Malformed { path: None, source } => write!(f, "malformed report: {source}"),
            ReportError::UnsupportedVersion { found, expected } => write!(
                f,
                "report format version {found} is not readable by this build, which understands {expected}"
            ),
        }
    }
}

impl std::error::Error for ReportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReportError::Io { source, .. } => Some(source),
            ReportError::Malformed { source, .. } => Some(source),
            ReportError::UnsupportedVersion { .. } => None,
        }
    }
}

impl Report {
    /// Sort every finding into a deterministic order.
    ///
    /// Two scans of identical trees must produce byte-identical reports, or a
    /// diff of the files shows churn that is not there. Clones sort by
    /// descending similarity so the strongest finding is read first.
    pub fn sort(&mut self) {
        self.clones.sort_by(|x, y| {
            y.similarity
                .partial_cmp(&x.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.key().cmp(&y.key()))
        });
        self.vocab.sort_by(|x, y| {
            y.overlap
                .partial_cmp(&x.overlap)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.key().cmp(&y.key()))
        });
        self.blocks.sort_by(|x, y| y.tokens.cmp(&x.tokens).then_with(|| x.key().cmp(&y.key())));
        let unique: BTreeSet<String> = self.files_with_syntax_errors.iter().cloned().collect();
        self.files_with_syntax_errors = unique.into_iter().collect();
    }

    /// Total findings across all three detectors.
    pub fn finding_count(&self) -> usize {
        self.clones.len() + self.vocab.len() + self.blocks.len()
    }

    /// Render as pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a Report contains only serializable types")
    }

    /// Parse from JSON, rejecting a format version this build cannot read.
    pub fn from_json(text: &str) -> Result<Report, ReportError> {
        Self::from_json_at(text, None)
    }

    fn from_json_at(text: &str, path: Option<&Path>) -> Result<Report, ReportError> {
        let report: Report = serde_json::from_str(text)
            .map_err(|source| ReportError::Malformed { path: path.map(Path::to_path_buf), source })?;
        if report.version != REPORT_VERSION {
            return Err(ReportError::UnsupportedVersion { found: report.version, expected: REPORT_VERSION });
        }
        Ok(report)
    }

    /// Write the report to a file as JSON.
    pub fn write_to(&self, path: &Path) -> Result<(), ReportError> {
        std::fs::write(path, self.to_json())
            .map_err(|source| ReportError::Io { path: path.to_path_buf(), source })
    }

    /// Read a report from a file.
    pub fn read_from(path: &Path) -> Result<Report, ReportError> {
        let text = std::fs::read_to_string(path)
            .map_err(|source| ReportError::Io { path: path.to_path_buf(), source })?;
        Self::from_json_at(&text, Some(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -------------------------------------------------------------- fixtures

    fn unit(file: &str, name: &str, token: &str) -> UnitRef {
        UnitRef {
            file: file.to_string(),
            qualname: name.to_string(),
            start_line: 1,
            end_line: 9,
            hash: ContentHash::of(&[token]),
        }
    }

    fn clone_pair(similarity: f64, a: &str, b: &str) -> ClonePair {
        ClonePair { similarity, a: unit("a.py", a, a), b: unit("b.py", b, b) }
    }

    fn vocab_pair(a: &str, b: &str, overlap: f64) -> VocabPair {
        VocabPair {
            a: a.to_string(),
            b: b.to_string(),
            overlap,
            shared: 12,
            a_vocabulary: 40,
            b_vocabulary: 30,
            a_inbound_imports: 0,
            b_inbound_imports: 3,
            zero_inbound: true,
            sample_shared: vec!["rate".to_string()],
        }
    }

    fn block_pair(file_a: &str, file_b: &str, tokens: usize) -> BlockPair {
        BlockPair {
            a: BlockRef { file: file_a.to_string(), start_line: 3, end_line: 9 },
            b: BlockRef { file: file_b.to_string(), start_line: 40, end_line: 46 },
            tokens,
            hash: ContentHash::of(&["fragment"]),
        }
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("dupdelta-report-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&path).expect("temp dir is creatable");
            TempDir(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // -------------------------------------------------------- pair identities

    #[test]
    fn a_clone_pair_has_the_same_key_whichever_side_is_listed_first() {
        // Swapping the sides between two scans must not make an unchanged pair
        // look new.
        let forward = clone_pair(0.9, "x", "y");
        let reversed = ClonePair { similarity: 0.9, a: forward.b.clone(), b: forward.a.clone() };
        assert_eq!(forward.key(), reversed.key());
    }

    #[test]
    fn clone_pair_keys_distinguish_different_content() {
        assert_ne!(clone_pair(0.9, "x", "y").key(), clone_pair(0.9, "x", "z").key());
    }

    #[test]
    fn a_vocab_pair_has_the_same_key_whichever_side_is_listed_first() {
        assert_eq!(vocab_pair("a.py", "b.py", 0.4).key(), vocab_pair("b.py", "a.py", 0.4).key());
    }

    #[test]
    fn a_block_pair_key_combines_the_fragment_with_both_files() {
        let forward = block_pair("a.py", "b.py", 50);
        assert_eq!(forward.key(), block_pair("b.py", "a.py", 50).key());
        // The same fragment in a different pair of files is a different finding.
        assert_ne!(forward.key(), block_pair("a.py", "c.py", 50).key());
    }

    // ----------------------------------------------------------------- sorting

    #[test]
    fn clones_sort_by_descending_similarity() {
        let mut report = Report {
            clones: vec![clone_pair(0.86, "a", "b"), clone_pair(0.99, "c", "d"), clone_pair(0.9, "e", "f")],
            ..Report::default()
        };
        report.sort();
        assert_eq!(report.clones.iter().map(|c| c.similarity).collect::<Vec<_>>(), vec![0.99, 0.9, 0.86]);
    }

    #[test]
    fn vocab_pairs_sort_by_descending_overlap() {
        let mut report =
            Report { vocab: vec![vocab_pair("a", "b", 0.3), vocab_pair("c", "d", 0.7)], ..Report::default() };
        report.sort();
        assert_eq!(report.vocab.iter().map(|v| v.overlap).collect::<Vec<_>>(), vec![0.7, 0.3]);
    }

    #[test]
    fn blocks_sort_by_descending_token_count() {
        let mut report =
            Report { blocks: vec![block_pair("a", "b", 30), block_pair("c", "d", 90)], ..Report::default() };
        report.sort();
        assert_eq!(report.blocks.iter().map(|b| b.tokens).collect::<Vec<_>>(), vec![90, 30]);
    }

    #[test]
    fn equal_scores_break_their_tie_deterministically() {
        // Two scans of identical trees must produce byte-identical reports.
        let build = || Report {
            clones: vec![clone_pair(0.9, "z", "y"), clone_pair(0.9, "a", "b")],
            blocks: vec![block_pair("m", "n", 10), block_pair("c", "d", 10)],
            ..Report::default()
        };
        let mut first = build();
        let mut second = build();
        second.clones.reverse();
        second.blocks.reverse();
        first.sort();
        second.sort();
        assert_eq!(first, second);
    }

    #[test]
    fn sorting_deduplicates_and_orders_the_syntax_error_list() {
        let mut report = Report {
            files_with_syntax_errors: vec!["z.py".into(), "a.py".into(), "z.py".into()],
            ..Report::default()
        };
        report.sort();
        assert_eq!(report.files_with_syntax_errors, vec!["a.py".to_string(), "z.py".to_string()]);
    }

    #[test]
    fn a_score_that_cannot_be_ordered_does_not_abort_the_run() {
        // NaN can only arrive from a hand-edited or foreign-produced report,
        // but sorting must degrade rather than panic.
        let mut report = Report {
            clones: vec![clone_pair(f64::NAN, "a", "b"), clone_pair(0.9, "c", "d")],
            vocab: vec![vocab_pair("a", "b", f64::NAN), vocab_pair("c", "d", 0.5)],
            ..Report::default()
        };
        report.sort();
        assert_eq!((report.clones.len(), report.vocab.len()), (2, 2));
    }

    // ------------------------------------------------------------- accounting

    #[test]
    fn finding_count_totals_all_three_detectors() {
        let report = Report {
            clones: vec![clone_pair(0.9, "a", "b")],
            vocab: vec![vocab_pair("a", "b", 0.4), vocab_pair("c", "d", 0.4)],
            blocks: vec![block_pair("a", "b", 50)],
            ..Report::default()
        };
        assert_eq!(report.finding_count(), 4);
    }

    #[test]
    fn an_empty_report_declares_the_current_version_and_no_findings() {
        let report = Report::default();
        assert_eq!((report.version, report.finding_count()), (REPORT_VERSION, 0));
    }

    // ------------------------------------------------------------ round trips

    #[test]
    fn a_report_survives_a_json_round_trip_intact() {
        let report = Report {
            files_scanned: 7,
            units_considered: 42,
            files_with_syntax_errors: vec!["broken.py".to_string()],
            clones: vec![clone_pair(0.91, "a", "b")],
            vocab: vec![vocab_pair("a.py", "b.py", 0.42)],
            blocks: vec![block_pair("a.py", "b.py", 64)],
            ..Report::default()
        };
        assert_eq!(Report::from_json(&report.to_json()).unwrap(), report);
    }

    #[test]
    fn a_report_survives_a_file_round_trip_intact() {
        let dir = TempDir::new();
        let path = dir.join("report.json");
        let report = Report { files_scanned: 3, ..Report::default() };
        report.write_to(&path).unwrap();
        assert_eq!(Report::read_from(&path).unwrap(), report);
    }

    // ----------------------------------------------------- refusing bad input

    #[test]
    fn a_report_from_an_unknown_format_version_is_refused() {
        // Reading it optimistically would produce a silently wrong delta.
        let json = Report { version: REPORT_VERSION + 1, ..Report::default() }.to_json();
        let error = Report::from_json(&json).unwrap_err();
        assert!(matches!(error, ReportError::UnsupportedVersion { .. }));
        assert!(error.to_string().contains(&(REPORT_VERSION + 1).to_string()));
    }

    #[test]
    fn malformed_json_is_refused_and_names_the_file_when_there_is_one() {
        let dir = TempDir::new();
        let path = dir.join("bad.json");
        std::fs::write(&path, "{ not json").unwrap();

        let from_file = Report::read_from(&path).unwrap_err().to_string();
        let from_text = Report::from_json("{ not json").unwrap_err().to_string();
        assert!(from_file.contains("bad.json"));
        assert!(!from_text.contains("bad.json"));
        assert!(from_text.contains("malformed report"));
    }

    #[test]
    fn reading_a_missing_report_names_the_path() {
        let error = Report::read_from(Path::new("/nonexistent/dupdelta/report.json")).unwrap_err();
        assert!(matches!(error, ReportError::Io { .. }));
        assert!(error.to_string().contains("report.json"));
    }

    #[test]
    fn writing_to_an_unwritable_path_reports_the_failure() {
        let error = Report::default().write_to(Path::new("/nonexistent/dupdelta/out.json")).unwrap_err();
        assert!(matches!(error, ReportError::Io { .. }));
    }

    #[test]
    fn every_error_exposes_its_cause_where_it_has_one() {
        use std::error::Error;
        let dir = TempDir::new();
        let path = dir.join("bad.json");
        std::fs::write(&path, "{").unwrap();

        let io = Report::read_from(Path::new("/nonexistent/dupdelta/x.json")).unwrap_err();
        let malformed = Report::read_from(&path).unwrap_err();
        let version = Report::from_json(&Report { version: 99, ..Report::default() }.to_json()).unwrap_err();

        assert_eq!(
            [io.source().is_some(), malformed.source().is_some(), version.source().is_some()],
            [true, true, false]
        );
        assert!(format!("{io:?}").contains("Io"));
    }

    // --------------------------------------------------------------- plumbing

    #[test]
    fn report_types_clone_compare_and_debug() {
        let report = Report {
            clones: vec![clone_pair(0.9, "a", "b")],
            vocab: vec![vocab_pair("a", "b", 0.3)],
            blocks: vec![block_pair("a", "b", 20)],
            ..Report::default()
        };
        assert_eq!(report.clone(), report);
        assert!(format!("{report:?}").contains("Report"));
        assert!(format!("{:?}", report.clones[0]).contains("similarity"));
        assert!(format!("{:?}", report.vocab[0]).contains("overlap"));
        assert!(format!("{:?}", report.blocks[0]).contains("tokens"));
        assert!(format!("{:?}", report.clones[0].a).contains("qualname"));
        assert!(format!("{:?}", report.blocks[0].a).contains("start_line"));
    }
}
