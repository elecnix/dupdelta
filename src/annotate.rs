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

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::Path;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A unique scratch file under the system temp dir, removed on drop.
    ///
    /// `std::process::id()` disambiguates across parallel `cargo test`
    /// processes (there is only one per run); the counter disambiguates
    /// across the several such files a single test process's tests create
    /// concurrently.
    struct TempFile {
        path: std::path::PathBuf,
    }

    impl TempFile {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "dupdelta-annotate-test-{}-{}-{label}",
                std::process::id(),
                n
            ));
            TempFile { path }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

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

    // ------------------------------------------------------------ append_to

    #[test]
    fn append_to_creates_the_file_when_absent() {
        let tmp = TempFile::new("create");
        let mut s = Summary::new();
        s.paragraph("first");
        s.append_to(&tmp.path).unwrap();
        let contents = std::fs::read_to_string(&tmp.path).unwrap();
        assert_eq!(contents, "first\n");
    }

    #[test]
    fn append_to_appends_rather_than_truncates() {
        let tmp = TempFile::new("append");
        let mut first = Summary::new();
        first.paragraph("one");
        first.append_to(&tmp.path).unwrap();

        let mut second = Summary::new();
        second.paragraph("two");
        second.append_to(&tmp.path).unwrap();

        let contents = std::fs::read_to_string(&tmp.path).unwrap();
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
