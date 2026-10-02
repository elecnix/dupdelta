//! How findings reach a human: GitHub Actions workflow-command annotations
//! (which render inline on a pull request's diff) and a markdown job summary.
//!
//! # The escaping is the substance of this module
//!
//! Workflow commands are a line-oriented text protocol
//! (`::warning file=...,line=3::message`), so any `%`, newline, `:` or `,`
//! *inside* a value would otherwise be parsed as protocol syntax rather than
//! content. GitHub Actions documents a real escaping spec for this, and it is
//! asymmetric: the free-text message needs three characters escaped, but a
//! `key=value` property needs `:` and `,` escaped as well, because those are
//! the property-list delimiters. Getting the order wrong (escaping `%` last)
//! double-escapes the very sequences that were just produced — so `%` always
//! goes first.
//!
//! Getting this wrong doesn't error, it silently mangles or truncates a
//! finding on the very PR the tool exists to protect. That is the loud vs.
//! quiet failure `CONTRIBUTING.md` warns about, applied to a text format
//! instead of a number.
//!
//! # How findings reach here
//!
//! This module is where a finding becomes text, and nothing else is. It
//! receives finished findings — a [`Delta`] and the
//! report pairs inside it — and never decides *what* was found, only how the
//! answer reads: which sentence, which columns, which section heading, and
//! the escaping and markdown underneath. The counterpart module,
//! [`delta`](crate::delta), answers the other half of the question and holds
//! no opinion about output formats.
//!
//! The two are joined by exactly one thing crossing one edge: the findings.
//! Nothing here reaches back into delta computation, and nothing in delta
//! knows a row of markdown exists.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::Path;

use crate::delta::{Delta, VocabChange, VocabFinding};
use crate::report::{BlockPair, BlockRef, ClonePair, UnitRef, VocabPair};

/// How loudly a finding is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Informational — surfaced, but does not draw the eye.
    Notice,
    /// Worth a look; does not fail the check by itself.
    Warning,
    /// The change introduced new duplication that must be addressed.
    Error,
}

impl Severity {
    /// The workflow-command name for this severity (`::notice`, `::warning`,
    /// `::error`).
    fn command(self) -> &'static str {
        match self {
            Severity::Notice => "notice",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

/// Escape a workflow-command **message** (the free-text part after `::`).
///
/// Order matters: `%` is replaced first. If a later replacement ran first and
/// produced a literal `%`, a subsequent `%` -> `%25` pass would re-escape it,
/// corrupting the payload it just built.
fn escape_message(s: &str) -> String {
    s.replace('%', "%25").replace('\r', "%0D").replace('\n', "%0A")
}

/// Escape a workflow-command **property value** (`file=`, `title=`, …).
///
/// Same three substitutions as [`escape_message`], plus `:` and `,`, because
/// those two characters delimit the property list itself
/// (`file=a,line=3:title=x`-shaped ambiguity) and are not otherwise special
/// in the message.
fn escape_property(s: &str) -> String {
    escape_message(s).replace(':', "%3A").replace(',', "%2C")
}

/// One finding, located in a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Annotation {
    /// How loudly this finding is rendered.
    pub severity: Severity,
    /// Path to the file the finding is about, relative to the repo root.
    pub file: String,
    /// First affected line, if known.
    pub start_line: Option<usize>,
    /// Last affected line, if known and different from `start_line`.
    pub end_line: Option<usize>,
    /// Short title shown above the message.
    pub title: Option<String>,
    /// The finding itself. May be multi-line.
    pub message: String,
}

impl Annotation {
    /// Build a [`Severity::Warning`] annotation — the common case for a
    /// duplication finding, which blocks nothing but should be seen.
    pub fn warning(file: impl Into<String>, line: Option<usize>, message: impl Into<String>) -> Self {
        Annotation {
            severity: Severity::Warning,
            file: file.into(),
            start_line: line,
            end_line: None,
            title: None,
            message: message.into(),
        }
    }

    /// Render as a GitHub Actions workflow command: a single line, regardless
    /// of how many newlines `message` contains, because the runner's log
    /// parser treats each *line* of stdout as one (possible) command.
    pub fn to_workflow_command(&self) -> String {
        let mut props = vec![format!("file={}", escape_property(&self.file))];
        if let Some(line) = self.start_line {
            props.push(format!("line={line}"));
        }
        if let Some(end_line) = self.end_line {
            props.push(format!("endLine={end_line}"));
        }
        if let Some(title) = &self.title {
            props.push(format!("title={}", escape_property(title)));
        }
        format!("::{} {}::{}", self.severity.command(), props.join(","), escape_message(&self.message))
    }
}

/// One block of a [`Summary`] under construction.
///
/// Kept as structured pieces rather than pre-rendered strings so `is_empty`
/// doesn't need to re-parse rendered markdown to answer "was anything added".
#[derive(Debug, Clone, PartialEq, Eq)]
enum Block {
    Heading(usize, String),
    Paragraph(String),
    Bullet(String),
    Table { headers: Vec<String>, rows: Vec<Vec<String>> },
}

/// An incrementally built markdown digest (the GitHub Actions job summary).
///
/// Methods return `&mut Self` so a summary can be built as one chained
/// expression, matching how a scan report is assembled — heading, some
/// paragraphs, a table — without a local mutable binding at every step.
#[derive(Debug, Default, Clone)]
pub struct Summary {
    blocks: Vec<Block>,
}

impl Summary {
    /// An empty summary. [`Summary::render`] on it is `""`.
    pub fn new() -> Self {
        Summary::default()
    }

    /// Append a heading. `level` must be `1..=6`, matching markdown's `#`
    /// nesting; anything else is a programming error in the caller, not a
    /// value that arrived from the outside world, so it panics rather than
    /// silently clamping to a level the caller didn't ask for.
    pub fn heading(&mut self, level: usize, text: &str) -> &mut Self {
        assert!((1..=6).contains(&level), "heading level must be 1..=6, got {level}");
        self.blocks.push(Block::Heading(level, text.to_string()));
        self
    }

    /// Append a paragraph.
    pub fn paragraph(&mut self, text: &str) -> &mut Self {
        self.blocks.push(Block::Paragraph(text.to_string()));
        self
    }

    /// Append a single bullet-list item.
    ///
    /// Consecutive bullets render as one markdown list because each renders
    /// as its own `- ` line with no blank line separating it from the next.
    pub fn bullet(&mut self, text: &str) -> &mut Self {
        self.blocks.push(Block::Bullet(text.to_string()));
        self
    }

    /// Append a markdown table.
    ///
    /// Every row's cell count must equal `headers.len()`: this is the same
    /// class of caller bug as a wrong `heading` level, and padding or
    /// truncating a mismatched row would silently misattribute a value to
    /// the wrong column in a rendered report nobody re-checks by hand — so
    /// it panics instead.
    pub fn table(&mut self, headers: &[&str], rows: &[Vec<String>]) -> &mut Self {
        for (i, row) in rows.iter().enumerate() {
            assert!(
                row.len() == headers.len(),
                "table row {i} has {} cell(s), header has {}",
                row.len(),
                headers.len()
            );
        }
        self.blocks.push(Block::Table {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            rows: rows.to_vec(),
        });
        self
    }

    /// Whether any content has been added.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Render the accumulated blocks as markdown.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for block in &self.blocks {
            match block {
                Block::Heading(level, text) => {
                    out.push_str(&"#".repeat(*level));
                    out.push(' ');
                    out.push_str(text);
                    out.push_str("\n\n");
                }
                Block::Paragraph(text) => {
                    out.push_str(text);
                    out.push_str("\n\n");
                }
                Block::Bullet(text) => {
                    out.push_str("- ");
                    out.push_str(text);
                    out.push('\n');
                }
                Block::Table { headers, rows } => {
                    render_table(&mut out, headers, rows);
                    out.push('\n');
                }
            }
        }
        // Bullets and tables leave a trailing blank line to separate them
        // from whatever follows; the very last block shouldn't leave one
        // dangling at the end of the document.
        while out.ends_with('\n') {
            out.pop();
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out
    }

    /// Append the rendered markdown to `path`, creating it if absent.
    ///
    /// Appends rather than truncates so a multi-job workflow (scan, then
    /// report) can each call this once and land in the same summary, which
    /// is how GitHub's own `$GITHUB_STEP_SUMMARY` file is meant to be used.
    pub fn append_to(&self, path: &Path) -> std::io::Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(self.render().as_bytes())
    }
}

/// Render a markdown table, escaping `|` in every cell so an embedded pipe
/// can't be mistaken for a column delimiter and break the table layout.
fn render_table(out: &mut String, headers: &[String], rows: &[Vec<String>]) {
    let escape = |s: &str| s.replace('|', "\\|");
    out.push('|');
    for h in headers {
        out.push(' ');
        out.push_str(&escape(h));
        out.push_str(" |");
    }
    out.push('\n');
    out.push('|');
    for _ in headers {
        out.push_str(" --- |");
    }
    out.push('\n');
    for row in rows {
        out.push('|');
        for cell in row {
            out.push(' ');
            out.push_str(&escape(cell));
            out.push_str(" |");
        }
        out.push('\n');
    }
}

// --------------------------------------------------------- rendering a delta
//
// Everything below turns a [`Delta`] into output. It is here, rather than in
// `delta.rs`, because none of it is a question about what changed: it is a
// question about how an unchanged answer should read. A new finding category
// and a prettier summary are independent changes, and keeping them in one
// module made each one an edit to both halves at once.
//
// The rows below are built next to [`Summary::table`], which is where the
// cell-count check lives — so the thing that could violate it and the check
// that rejects it are no longer on opposite sides of a module boundary with
// nothing between them but a runtime assertion.

// ------------------------------------------------------------- annotations

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

/// One annotation per vocabulary finding; clone and block pairs each produce
/// two, one anchored on each side, because a reviewer standing on either
/// file needs the other file's location to judge the finding, and GitHub only
/// renders an annotation on the file it names.
pub fn delta_annotations(delta: &Delta) -> Vec<Annotation> {
    let mut out =
        Vec::with_capacity(2 * delta.new_clones.len() + delta.vocab.len() + 2 * delta.new_blocks.len());

    for pair in &delta.new_clones {
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

    for finding in &delta.vocab {
        let pair = &finding.pair;
        out.push(Annotation::warning(pair.a.clone(), None, vocab_message(&finding.change, pair, &pair.b)));
    }

    for pair in &delta.new_blocks {
        out.push(side_annotation(&pair.a.file, pair.a.start_line, block_side_message(pair.tokens, &pair.b)));
        out.push(side_annotation(&pair.b.file, pair.b.start_line, block_side_message(pair.tokens, &pair.a)));
    }

    out
}

// ----------------------------------------------------------------- summary

fn vocab_change_label(change: &VocabChange) -> &'static str {
    match change {
        VocabChange::New => "new",
        VocabChange::BecameUnreferenced => "became unreferenced",
        VocabChange::Worsened { .. } => "worsened",
    }
}

/// State how many findings `max_findings` withheld, when any were.
///
/// Silence about a cap is the failure mode worth avoiding: a reader who sees
/// fifty findings and is not told there were four hundred will act as though
/// they have seen all of them.
fn note_withheld(delta: &Delta, summary: &mut Summary) {
    if delta.withheld > 0 {
        summary.paragraph(&format!(
            "{} further finding(s) withheld by the `report.max_findings` cap. Raise or remove \
             it to see them all.",
            delta.withheld
        ));
    }
}

/// The clone-pair table: similarity first, then each side located in full —
/// file, line range and the enclosing name, because the name is what tells a
/// reader whether two long functions are genuinely the same shape.
fn clone_rows(clones: &[ClonePair]) -> Vec<Vec<String>> {
    clones
        .iter()
        .map(|pair| {
            vec![
                format!("{:.0}%", pair.similarity * 100.0),
                format!("{}:{}-{} (`{}`)", pair.a.file, pair.a.start_line, pair.a.end_line, pair.a.qualname),
                format!("{}:{}-{} (`{}`)", pair.b.file, pair.b.start_line, pair.b.end_line, pair.b.qualname),
            ]
        })
        .collect()
}

fn vocab_rows(vocab: &[VocabFinding]) -> Vec<Vec<String>> {
    vocab
        .iter()
        .map(|finding| {
            vec![
                vocab_change_label(&finding.change).to_string(),
                finding.pair.a.clone(),
                finding.pair.b.clone(),
                format!("{:.0}%", finding.pair.overlap * 100.0),
            ]
        })
        .collect()
}

fn block_rows(blocks: &[BlockPair]) -> Vec<Vec<String>> {
    blocks
        .iter()
        .map(|pair| {
            vec![
                format!("{} tokens", pair.tokens),
                format!("{}:{}-{}", pair.a.file, pair.a.start_line, pair.a.end_line),
                format!("{}:{}-{}", pair.b.file, pair.b.start_line, pair.b.end_line),
            ]
        })
        .collect()
}

/// A markdown digest of the whole delta, for the GitHub Actions job summary.
/// One table per non-empty category; empty categories are omitted rather than
/// rendered as an empty table nobody needs to see.
pub fn delta_summary(delta: &Delta) -> Summary {
    let mut summary = Summary::new();
    summary.heading(2, "Duplication delta");

    if delta.is_empty() {
        summary.paragraph("No new duplication vs the merge-base.");
        note_withheld(delta, &mut summary);
        return summary;
    }

    summary.paragraph(
        "New duplication introduced by this change, relative to its merge-base. Nothing here \
         blocks a merge: extract the shared logic where that makes sense, or leave it if the \
         similarity is coincidental.",
    );
    note_withheld(delta, &mut summary);

    if !delta.new_clones.is_empty() {
        summary.heading(3, "New clone pairs");
        let rows = clone_rows(&delta.new_clones);
        summary.table(&["Similarity", "A", "B"], &rows);
    }

    if !delta.vocab.is_empty() {
        summary.heading(3, "Vocabulary findings");
        let rows = vocab_rows(&delta.vocab);
        summary.table(&["Change", "A", "B", "Overlap"], &rows);
    }

    if !delta.new_blocks.is_empty() {
        summary.heading(3, "New duplicated blocks");
        let rows = block_rows(&delta.new_blocks);
        summary.table(&["Size", "A", "B"], &rows);
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{Delta, VocabChange, VocabFinding};
    use crate::testutil::{vocab_pair, TempTree};
    use crate::token::ContentHash;

    // ------------------------------------------------------------- Severity

    #[test]
    fn severity_is_comparable_and_debuggable() {
        assert_eq!(Severity::Warning, Severity::Warning);
        assert_ne!(Severity::Warning, Severity::Error);
        let copy = Severity::Notice;
        assert_eq!(copy, Severity::Notice);
        assert!(format!("{:?}", Severity::Error).contains("Error"));
    }

    // ----------------------------------------------------------- Annotation

    #[test]
    fn warning_builds_a_bare_line_only_annotation() {
        let a = Annotation::warning("src/a.rs", Some(3), "hello");
        assert_eq!(
            a,
            Annotation {
                severity: Severity::Warning,
                file: "src/a.rs".to_string(),
                start_line: Some(3),
                end_line: None,
                title: None,
                message: "hello".to_string(),
            }
        );
    }

    #[test]
    fn annotation_is_cloneable_and_debuggable() {
        let a = Annotation::warning("f", None, "m");
        let cloned = a.clone();
        assert_eq!(a, cloned);
        assert!(format!("{a:?}").contains("Warning"));
    }

    #[test]
    fn renders_full_command_with_all_properties() {
        let a = Annotation {
            severity: Severity::Error,
            file: "src/a.rs".to_string(),
            start_line: Some(3),
            end_line: Some(5),
            title: Some("Something".to_string()),
            message: "the message here".to_string(),
        };
        assert_eq!(
            a.to_workflow_command(),
            "::error file=src/a.rs,line=3,endLine=5,title=Something::the message here"
        );
    }

    #[test]
    fn notice_and_warning_commands_use_their_own_names() {
        let notice = Annotation::warning("f", None, "m");
        let mut notice = notice;
        notice.severity = Severity::Notice;
        assert!(notice.to_workflow_command().starts_with("::notice "));
        let warning = Annotation::warning("f", None, "m");
        assert!(warning.to_workflow_command().starts_with("::warning "));
    }

    #[test]
    fn omits_absent_optional_properties() {
        let a = Annotation::warning("src/a.rs", None, "m");
        assert_eq!(a.to_workflow_command(), "::warning file=src/a.rs::m");
    }

    #[test]
    fn omits_end_line_when_only_start_line_is_present() {
        let a = Annotation::warning("src/a.rs", Some(3), "m");
        assert_eq!(a.to_workflow_command(), "::warning file=src/a.rs,line=3::m");
    }

    #[test]
    fn windows_line_endings_in_message_become_percent_0d_0a() {
        let a = Annotation::warning("f", None, "line one\r\nline two");
        assert_eq!(a.to_workflow_command(), "::warning file=f::line one%0D%0Aline two");
    }

    #[test]
    fn double_colon_in_message_is_not_escaped() {
        // `::` only matters at the start of a *line*; mid-message it is inert
        // to the runner's parser and escaping it would just be noise the
        // reader has to mentally strip back out.
        let a = Annotation::warning("f", None, "found dup::here");
        assert_eq!(a.to_workflow_command(), "::warning file=f::found dup::here");
    }

    #[test]
    fn colon_in_a_windows_path_property_is_escaped() {
        let a = Annotation::warning(r"C:\repo\src\a.rs", Some(1), "m");
        assert_eq!(a.to_workflow_command(), "::warning file=C%3A\\repo\\src\\a.rs,line=1::m");
    }

    #[test]
    fn comma_in_a_property_value_is_escaped() {
        let a = Annotation {
            severity: Severity::Warning,
            file: "f".to_string(),
            start_line: None,
            end_line: None,
            title: Some("a, b".to_string()),
            message: "m".to_string(),
        };
        assert_eq!(a.to_workflow_command(), "::warning file=f,title=a%2C b::m");
    }

    #[test]
    fn percent_is_escaped_first_so_the_escapes_are_not_doubly_escaped() {
        // If `\n` were escaped to `%0A` before `%` were escaped, the `%` that
        // `%0A` itself introduces would get caught by a later `%` pass and
        // become `%250A` -- corrupting the very escape sequence that was
        // just produced. Escaping `%` first rules that out.
        let a = Annotation::warning("f", None, "100%\ndone");
        assert_eq!(a.to_workflow_command(), "::warning file=f::100%25%0Adone");
    }

    #[test]
    fn percent_in_a_property_value_is_escaped() {
        let a = Annotation {
            severity: Severity::Warning,
            file: "f".to_string(),
            start_line: None,
            end_line: None,
            title: Some("100% dup".to_string()),
            message: "m".to_string(),
        };
        assert_eq!(a.to_workflow_command(), "::warning file=f,title=100%25 dup::m");
    }

    // --------------------------------------------------------------- Summary

    #[test]
    fn new_summary_renders_empty_and_reports_empty() {
        let s = Summary::new();
        assert!(s.is_empty());
        assert_eq!(s.render(), "");
    }

    #[test]
    fn default_summary_is_also_empty() {
        let s = Summary::default();
        assert!(s.is_empty());
    }

    #[test]
    fn summary_with_content_is_not_empty() {
        let mut s = Summary::new();
        s.paragraph("hi");
        assert!(!s.is_empty());
    }

    #[test]
    fn heading_renders_with_hashes_for_its_level() {
        let mut s = Summary::new();
        s.heading(2, "Duplication report");
        assert_eq!(s.render(), "## Duplication report\n");
    }

    #[test]
    fn heading_level_zero_panics() {
        let mut s = Summary::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.heading(0, "x");
        }));
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "heading level must be 1..=6, got 7")]
    fn heading_level_seven_panics() {
        Summary::new().heading(7, "x");
    }

    #[test]
    fn paragraph_and_bullets_chain_and_render_in_order() {
        let mut s = Summary::new();
        s.heading(1, "Title").paragraph("intro").bullet("first").bullet("second");
        assert_eq!(s.render(), "# Title\n\nintro\n\n- first\n- second\n");
    }

    #[test]
    fn table_renders_header_separator_and_rows() {
        let mut s = Summary::new();
        s.table(
            &["File", "Score"],
            &[vec!["a.rs".to_string(), "0.9".to_string()], vec!["b.rs".to_string(), "0.8".to_string()]],
        );
        assert_eq!(s.render(), "| File | Score |\n| --- | --- |\n| a.rs | 0.9 |\n| b.rs | 0.8 |\n");
    }

    #[test]
    fn table_escapes_pipe_in_a_cell() {
        let mut s = Summary::new();
        s.table(&["Expr"], &[vec!["a | b".to_string()]]);
        assert_eq!(s.render(), "| Expr |\n| --- |\n| a \\| b |\n");
    }

    #[test]
    fn table_with_empty_rows_still_renders_header() {
        let mut s = Summary::new();
        s.table(&["File"], &[]);
        assert_eq!(s.render(), "| File |\n| --- |\n");
    }

    #[test]
    #[should_panic(expected = "table row 1 has 1 cell(s), header has 2")]
    fn table_row_with_wrong_cell_count_panics() {
        Summary::new()
            .table(&["a", "b"], &[vec!["1".to_string(), "2".to_string()], vec!["only one".to_string()]]);
    }

    #[test]
    fn multiple_blocks_render_as_one_document_without_a_trailing_blank_line() {
        let mut s = Summary::new();
        s.heading(1, "T");
        s.table(&["a"], &[vec!["1".to_string()]]);
        let rendered = s.render();
        assert!(!rendered.ends_with("\n\n"));
        assert!(rendered.ends_with('\n'));
    }

    #[test]
    fn summary_is_cloneable_and_debuggable() {
        let mut s = Summary::new();
        s.paragraph("x");
        let cloned = s.clone();
        assert_eq!(cloned.render(), s.render());
        assert!(format!("{s:?}").contains("Paragraph"));
    }

    // ------------------------------------------------------------- fixtures
    //
    // Delta findings, built by hand rather than by scanning: rendering is the
    // subject under test here, so the findings themselves should be the
    // simplest possible input to it.

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

    fn block_ref(file: &str, start_line: usize, end_line: usize) -> BlockRef {
        BlockRef { file: file.to_string(), start_line, end_line }
    }

    fn block_pair(token: &str, tokens: usize, a: BlockRef, b: BlockRef) -> BlockPair {
        BlockPair { a, b, tokens, hash: ContentHash::of(&[token]) }
    }

    // ------------------------------------------- delta rendering: annotations

    #[test]
    fn a_new_clone_pair_annotates_both_sides_with_similarity_and_the_other_location() {
        let pair = clone_pair(0.87, unit("a.py", "f", "f", 3, 9), unit("b.py", "g", "g", 40, 46));
        let delta = Delta { new_clones: vec![pair], vocab: vec![], new_blocks: vec![], withheld: 0 };
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
        let delta = Delta { new_clones: vec![], vocab: vec![], new_blocks: vec![pair], withheld: 0 };
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
        let delta = Delta { new_clones: vec![], vocab: vec![finding], new_blocks: vec![], withheld: 0 };
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
        let delta = Delta {
            new_clones: vec![],
            vocab: vec![unreferenced, worsened],
            new_blocks: vec![],
            withheld: 0,
        };
        let annotations = delta.annotations();
        assert!(annotations[0].message.contains("inbound imports"));
        assert!(annotations[1].message.contains("30%"));
        assert!(annotations[1].message.contains("50%"));
    }

    #[test]
    fn annotations_are_empty_when_the_delta_is_empty() {
        assert!(Delta::default().annotations().is_empty());
    }

    // ---------------------------------------------- delta rendering: summary

    #[test]
    fn summary_renders_a_table_per_non_empty_category_and_omits_empty_ones() {
        let delta = Delta {
            new_clones: vec![clone_pair(0.9, unit("a.py", "f", "f", 1, 5), unit("b.py", "g", "g", 1, 5))],
            vocab: vec![],
            new_blocks: vec![],
            withheld: 0,
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
            withheld: 0,
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
            withheld: 0,
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
            withheld: 0,
        };
        let rendered = delta.summary().render();
        assert!(rendered.contains("New clone pairs"));
        assert!(rendered.contains("Vocabulary findings"));
        assert!(rendered.contains("New duplicated blocks"));
    }

    // ------------------------------------------------------------ append_to

    #[test]
    fn append_to_creates_the_file_when_absent() {
        let tmp = TempTree::new("annotate");
        let path = tmp.join("create");
        let mut s = Summary::new();
        s.paragraph("first");
        s.append_to(&path).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "first\n");
    }

    #[test]
    fn append_to_appends_rather_than_truncates() {
        let tmp = TempTree::new("annotate");
        let path = tmp.join("append");
        let mut first = Summary::new();
        first.paragraph("one");
        first.append_to(&path).unwrap();

        let mut second = Summary::new();
        second.paragraph("two");
        second.append_to(&path).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "one\ntwo\n");
    }

    #[test]
    fn append_to_an_unwritable_path_returns_err() {
        let mut s = Summary::new();
        s.paragraph("x");
        let bad = Path::new("/no/such/parent/dir/summary.md");
        assert!(s.append_to(bad).is_err());
    }
}
