//! The command line, and the pipeline behind it.
//!
//! Three commands, matching the three things anyone needs:
//!
//! - `scan` — reduce a tree to a report.
//! - `diff` — compare two reports and say what the first added.
//! - `ci` — do both: scan the working tree, scan its merge-base with some
//!   revision, and report the difference. This is the one CI runs.
//!
//! # Paths in a report are relative to the tree that was scanned
//!
//! `ci` scans two trees that live in different directories — the working tree,
//! and a detached worktree of the merge-base in a temporary directory. If
//! reports carried absolute paths, *every* finding in the base report would
//! have a path no head finding could match, vocabulary pairs would all look
//! new, and the tool would report the entire codebase as freshly duplicated.
//! So paths are made relative to the scan root, with `/` separators, before
//! anything else sees them.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use crate::annotate::Summary;
use crate::blocks::{self, BlockOptions};
use crate::config::{Config, ConfigError};
use crate::delta::{Delta, DeltaOptions};
use crate::extract::{Extractor, SourceFile};
use crate::git::{GitError, Repo};
use crate::lang;
use crate::report::{Report, ReportError};
use crate::scan;
use crate::token::Interner;
use crate::vocab::{self, VocabOptions};
use crate::walk::{self, WalkError, WalkOptions};

/// Report only the code duplication a change introduces.
#[derive(Debug, Parser)]
#[command(name = "dupdelta", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Scan a tree and write a report.
    Scan {
        /// Paths to scan. Defaults to the whole tree under `--root`.
        paths: Vec<PathBuf>,
        /// Tree root that recorded paths are made relative to.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Where to write the JSON report. Defaults to standard output.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Extra path substrings to skip, on top of the configured excludes.
        #[arg(long = "exclude")]
        excludes: Vec<String>,
        /// Configuration file. Defaults to discovery by walking up.
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Compare two reports and report what the first one added.
    Diff {
        /// Report for the tree under review.
        #[arg(long)]
        head: PathBuf,
        /// Report for the tree to compare against.
        #[arg(long)]
        base: PathBuf,
        #[command(flatten)]
        output: OutputArgs,
        /// Configuration file. Defaults to discovery by walking up.
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Scan the working tree and its merge-base, and report the difference.
    Ci {
        /// Revision to compare against.
        #[arg(long, default_value = "main")]
        base: String,
        /// Paths to scan. Defaults to the repository root.
        #[arg(long = "path")]
        paths: Vec<PathBuf>,
        /// Also write the head tree's full report here.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Extra path substrings to skip.
        #[arg(long = "exclude")]
        excludes: Vec<String>,
        #[command(flatten)]
        output: OutputArgs,
        /// Configuration file. Defaults to discovery by walking up.
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// List the languages this build can parse.
    Languages,
}

/// How findings are rendered. Shared by `diff` and `ci`.
#[derive(Debug, Clone, clap::Args)]
struct OutputArgs {
    /// Emit GitHub Actions annotations, which render inline on a pull request.
    #[arg(long)]
    github_annotations: bool,
    /// Append a markdown digest to this file, e.g. `$GITHUB_STEP_SUMMARY`.
    #[arg(long)]
    summary: Option<PathBuf>,
    /// Write the number of findings here, for a CI job to pick up.
    #[arg(long)]
    findings_out: Option<PathBuf>,
}

/// Anything that can stop a command.
#[derive(Debug)]
pub enum CliError {
    /// A tree could not be walked.
    Walk(WalkError),
    /// A report could not be read or written.
    Report(ReportError),
    /// The configuration was missing or invalid.
    Config(ConfigError),
    /// Git would not answer.
    Git(GitError),
    /// A file could not be read or written.
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying cause.
        source: std::io::Error,
    },
    /// A `ci` path lies outside the repository being compared.
    ///
    /// `ci` resolves each path inside *two* trees — the working tree and a
    /// worktree of the merge-base. A path not under the repository root cannot
    /// be resolved inside the second one; joining it there yields the working
    /// tree again, so the tool would compare that tree against itself and
    /// report every duplicate in it as new. Refused rather than guessed at.
    PathOutsideRepository {
        /// The offending path.
        path: PathBuf,
        /// The repository root it has to be under.
        root: PathBuf,
    },
    /// The base revision `ci` was asked to compare against resolves nowhere:
    /// not as given, and not as `origin/<base>` either.
    ///
    /// The most common cause is not a typo: the checkout dupdelta's own
    /// GitHub Action requires (`actions/checkout` with `fetch-depth: 0`)
    /// leaves `HEAD` detached and creates no local branches. `resolve_base`
    /// falls back to `origin/<base>` automatically; this error is only
    /// reached when that failed too, so the base genuinely does not exist in
    /// this checkout — the error says what was tried and what would fix it
    /// rather than leaving git's bare "unknown revision" as the whole story.
    UnresolvableBase {
        /// The base revision that failed to resolve, as given.
        base: String,
        /// git's own failure for the base as given, verbatim.
        source: GitError,
    },
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Walk(e) => write!(f, "{e}"),
            CliError::Report(e) => write!(f, "{e}"),
            CliError::Config(e) => write!(f, "{e}"),
            CliError::Git(e) => write!(f, "{e}"),
            CliError::Io { path, source } => write!(f, "could not access {}: {source}", path.display()),
            CliError::PathOutsideRepository { path, root } => write!(
                f,
                "{} is outside the repository at {}; `ci` can only scan paths within it",
                path.display(),
                root.display()
            ),
            CliError::UnresolvableBase { base, source } => write!(
                f,
                "could not resolve the base revision '{base}': {source}. Neither the name as given \
                 nor 'origin/{base}' resolves here; fetch the branch, or pass the full ref of the \
                 revision to compare against"
            ),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CliError::Walk(e) => Some(e),
            CliError::Report(e) => Some(e),
            CliError::Config(e) => Some(e),
            CliError::Git(e) => Some(e),
            CliError::Io { source, .. } => Some(source),
            CliError::PathOutsideRepository { .. } => None,
            CliError::UnresolvableBase { source, .. } => Some(source),
        }
    }
}

impl From<WalkError> for CliError {
    fn from(e: WalkError) -> Self {
        CliError::Walk(e)
    }
}
impl From<ReportError> for CliError {
    fn from(e: ReportError) -> Self {
        CliError::Report(e)
    }
}
impl From<ConfigError> for CliError {
    fn from(e: ConfigError) -> Self {
        CliError::Config(e)
    }
}
impl From<GitError> for CliError {
    fn from(e: GitError) -> Self {
        CliError::Git(e)
    }
}

/// Render a path the way a report records it: relative to the scan root, with
/// `/` separators, so two scans of the same code in different directories agree.
fn report_path(path: &Path, root: &Path) -> PathBuf {
    let relative = path.strip_prefix(root).unwrap_or(path);
    PathBuf::from(relative.to_string_lossy().replace('\\', "/"))
}

/// Read every supported file under `scan_roots`, with paths relativized to `tree_root`.
///
/// The two are separate because `ci` scans the same logical tree twice from two
/// different directories, and only paths relative to each tree's own root
/// compare equal between the resulting reports.
fn read_tree(
    tree_root: &Path,
    scan_roots: &[PathBuf],
    config: &Config,
    extra_excludes: &[String],
) -> Result<Vec<SourceFile>, CliError> {
    let mut excludes = config.excludes.clone();
    excludes.extend_from_slice(extra_excludes);

    let roots = if scan_roots.is_empty() { vec![tree_root.to_path_buf()] } else { scan_roots.to_vec() };
    let options = WalkOptions { roots, excludes, ..WalkOptions::default() };
    let discovered = walk::discover(&options, lang::is_supported)?;

    let mut files = Vec::with_capacity(discovered.len());
    for path in discovered {
        // A file that cannot be read is an error, not a file quietly missing
        // from the scan: a scan that skipped half a tree still reports "no new
        // duplication", which is indistinguishable from a clean result.
        let text =
            std::fs::read_to_string(&path).map_err(|source| CliError::Io { path: path.clone(), source })?;
        let language = lang::for_path(&path).expect("the walker only accepted supported paths");
        files.push(SourceFile { path: report_path(&path, tree_root), language, text });
    }
    Ok(files)
}

/// Scan a tree with every detector and assemble a report.
///
/// Recorded paths are relative to `tree_root`; `scan_roots` narrows *what* is
/// looked at without changing how it is named. An empty `scan_roots` means the
/// whole tree.
pub fn scan_tree(
    tree_root: &Path,
    scan_roots: &[PathBuf],
    config: &Config,
    extra_excludes: &[String],
) -> Result<Report, CliError> {
    let files = read_tree(tree_root, scan_roots, config, extra_excludes)?;

    let mut interner = Interner::new();
    let mut units = Vec::new();
    let mut broken = Vec::new();
    for file in &files {
        let mut extractor = Extractor::new(file.language);
        let extraction = extractor.extract(&file.text, &file.path, config.function.min_nodes, &mut interner);
        if extraction.had_syntax_errors {
            broken.push(file.path.to_string_lossy().to_string());
        }
        units.extend(extraction.units);
    }

    let vocab_options = VocabOptions {
        min_overlap: config.vocab.min_overlap,
        min_vocabulary: config.vocab.min_vocabulary,
        noise: config.vocab.noise.clone(),
        sample_size: VOCAB_SAMPLE_SIZE,
    };

    let mut report = Report {
        files_scanned: files.len(),
        units_considered: units.len(),
        files_with_syntax_errors: broken,
        clones: scan::find_clones(&units, config.function.min_similarity),
        vocab: vocab::find_vocab_pairs(&files, &vocab_options),
        blocks: blocks::find_blocks(&files, &BlockOptions { min_tokens: config.blocks.min_tokens }),
        ..Report::default()
    };
    report.sort();
    Ok(report)
}

/// How many shared identifiers to show per vocabulary finding.
const VOCAB_SAMPLE_SIZE: usize = 25;

/// Where a temporary worktree of the merge-base is checked out.
fn base_worktree_path(root: &Path) -> PathBuf {
    std::env::temp_dir().join(format!(
        "dupdelta-base-{}-{}",
        std::process::id(),
        root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
    ))
}

fn load_config(explicit: Option<&Path>, start: &Path) -> Result<Config, CliError> {
    match explicit {
        Some(path) => Ok(Config::load(path)?),
        None => match Config::discover(start) {
            Some(found) => Ok(Config::load(&found)?),
            None => Ok(Config::default()),
        },
    }
}

/// Express each path relative to the repository root.
///
/// An absolute path is accepted only if it is inside the repository, and is
/// then relativized; anything else is refused. Both sides are canonicalized
/// first so that a symlinked temp directory or a `./` prefix does not read as
/// "outside".
fn repo_relative_paths(paths: &[PathBuf], root: &Path) -> Result<Vec<PathBuf>, CliError> {
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut relative = Vec::with_capacity(paths.len());
    for path in paths {
        if path.is_relative() {
            relative.push(path.clone());
            continue;
        }
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        match canonical.strip_prefix(&canonical_root) {
            Ok(stripped) => relative.push(stripped.to_path_buf()),
            Err(_) => {
                return Err(CliError::PathOutsideRepository { path: path.clone(), root: canonical_root })
            }
        }
    }
    Ok(relative)
}

/// `ci`'s starting point for repository discovery: the first `--path`, or the
/// current directory when none was given.
fn default_scan_root(paths: &[PathBuf]) -> PathBuf {
    paths.first().cloned().unwrap_or_else(|| PathBuf::from("."))
}

/// Resolve the base revision `ci` compares against.
///
/// The base as given is tried first. When it does not resolve,
/// `origin/<base>` is tried in its place — on the checkout this tool's own
/// action produces, the base branch exists only as that remote-tracking ref,
/// and asking every user to know it is friction the tool can absorb. The
/// substitution is announced on stderr so the comparison is never silent
/// about what it compared against; when the fallback fails too, the error
/// carries git's verbatim failure for the base as given (the fallback's own
/// failure adds nothing — it is the same missing ref, one name over) and the
/// refs that match the name for the diagnostic.
fn resolve_base(repo: &Repo, base: &str) -> Result<String, CliError> {
    match repo.resolve(base) {
        Ok(commit) => Ok(commit),
        Err(source) => {
            let fallback = format!("origin/{base}");
            match repo.resolve(&fallback) {
                Ok(commit) => {
                    eprintln!(
                        "dupdelta: note: base '{base}' does not resolve; using remote-tracking ref '{fallback}'"
                    );
                    Ok(commit)
                }
                // The fallback's own git failure adds nothing to the story —
                // it is the same missing ref, one name over — so only the
                // original failure is carried.
                Err(_) => Err(CliError::UnresolvableBase { base: base.to_string(), source }),
            }
        }
    }
}

fn delta_options(config: &Config) -> DeltaOptions {
    DeltaOptions {
        min_similarity: config.function.min_similarity,
        worsened_delta: config.vocab.worsened_delta,
        max_findings: config.report.max_findings,
    }
}

/// Emit a delta through every requested channel, and return the finding count.
fn emit(delta: &Delta, output: &OutputArgs, out: &mut dyn std::io::Write) -> Result<usize, CliError> {
    if output.github_annotations {
        for annotation in delta.annotations() {
            writeln!(out, "{}", annotation.to_workflow_command())
                .map_err(|source| CliError::Io { path: PathBuf::from("<stdout>"), source })?;
        }
    }

    let summary: Summary = delta.summary();
    if let Some(path) = &output.summary {
        summary.append_to(path).map_err(|source| CliError::Io { path: path.clone(), source })?;
    }

    let count = delta.finding_count();
    if let Some(path) = &output.findings_out {
        std::fs::write(path, count.to_string())
            .map_err(|source| CliError::Io { path: path.clone(), source })?;
    }

    if !output.github_annotations {
        write!(out, "{}", summary.render())
            .map_err(|source| CliError::Io { path: PathBuf::from("<stdout>"), source })?;
    }
    Ok(count)
}

impl Cli {
    /// Parse arguments, failing with clap's own message on bad input.
    pub fn parse_from_args<I, T>(argv: I) -> Result<Cli, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        Cli::try_parse_from(argv)
    }

    /// Run the parsed command.
    ///
    /// Returns the number of findings, which is *not* an exit status: this tool
    /// warns and never fails a build, so the caller reports zero either way.
    pub fn run(self, out: &mut dyn std::io::Write) -> Result<usize, CliError> {
        match self.command {
            Command::Languages => {
                for language in lang::all() {
                    let extensions: Vec<&str> = language.extensions.to_vec();
                    writeln!(out, "{:<12} {}", language.name, extensions.join(" "))
                        .map_err(|source| CliError::Io { path: PathBuf::from("<stdout>"), source })?;
                }
                Ok(0)
            }

            Command::Scan { paths, root, out: destination, excludes, config } => {
                let config = load_config(config.as_deref(), &root)?;
                let report = scan_tree(&root, &paths, &config, &excludes)?;
                match destination {
                    Some(path) => report.write_to(&path)?,
                    None => write!(out, "{}", report.to_json())
                        .map_err(|source| CliError::Io { path: PathBuf::from("<stdout>"), source })?,
                }
                Ok(report.finding_count())
            }

            Command::Diff { head, base, output, config } => {
                let config = load_config(config.as_deref(), Path::new("."))?;
                let head_report = Report::read_from(&head)?;
                let base_report = Report::read_from(&base)?;
                let delta = Delta::compute(&head_report, &base_report, &delta_options(&config));
                emit(&delta, &output, out)
            }

            Command::Ci { base, paths, report, excludes, output, config } => {
                let start = default_scan_root(&paths);
                let repo = Repo::discover(&start)?;
                let config = load_config(config.as_deref(), repo.root())?;

                let head_commit = repo.resolve("HEAD")?;
                let base_commit = resolve_base(&repo, &base)?;
                let merge_base = repo.merge_base(&base_commit, &head_commit)?;

                // The same relative sub-paths, resolved inside each tree, so
                // the two reports name the same code the same way. An absolute
                // path would resolve to the working tree in *both* joins,
                // making the comparison a tree against itself.
                let relative = repo_relative_paths(&paths, repo.root())?;
                let head_roots: Vec<PathBuf> = relative.iter().map(|p| repo.root().join(p)).collect();
                let head_report = scan_tree(repo.root(), &head_roots, &config, &excludes)?;
                if let Some(path) = &report {
                    head_report.write_to(path)?;
                }

                let worktree = repo.add_detached_worktree(&base_worktree_path(repo.root()), &merge_base)?;
                let base_roots: Vec<PathBuf> = relative.iter().map(|p| worktree.path().join(p)).collect();
                let base_report = scan_tree(worktree.path(), &base_roots, &config, &excludes)?;
                worktree.remove()?;

                let delta = Delta::compute(&head_report, &base_report, &delta_options(&config));
                emit(&delta, &output, out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempTree;

    /// Two functions that are the same logic under different names.
    const TWINS: &str = "\
def total(rate, years, base):
    running = base
    for step in range(years):
        running = running * (1 + rate)
        if running > base:
            running = running - 1
    return running

def compute(pct, count, start):
    acc = start
    for i in range(count):
        acc = acc * (1 + pct)
        if acc > start:
            acc = acc - 1
    return acc
";

    fn run_cli(args: &[&str]) -> (Result<usize, CliError>, String) {
        let cli = Cli::parse_from_args(args).expect("arguments parse");
        let mut out: Vec<u8> = Vec::new();
        let result = cli.run(&mut out);
        (result, String::from_utf8(out).expect("output is utf-8"))
    }

    /// A writer that always fails, so `emit`'s and `run`'s stdout-write error
    /// paths — otherwise unreachable, since a `Vec<u8>` never fails to write
    /// — can be exercised honestly.
    struct FailingWriter;

    impl std::io::Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("destination refuses writes"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn run_cli_with(args: &[&str], out: &mut dyn std::io::Write) -> Result<usize, CliError> {
        Cli::parse_from_args(args).expect("arguments parse").run(out)
    }

    #[test]
    fn a_failing_writer_still_reports_flush_as_succeeding() {
        // None of `emit`'s or `run`'s write paths ever call `flush`, so
        // nothing above this line exercises it; it is proven directly.
        assert!(std::io::Write::flush(&mut FailingWriter).is_ok());
    }

    // ------------------------------------------------------------ report_path

    #[test]
    fn a_report_path_is_relative_to_the_scanned_root() {
        assert_eq!(report_path(Path::new("/tree/src/a.py"), Path::new("/tree")), PathBuf::from("src/a.py"));
    }

    #[test]
    fn a_path_outside_the_root_is_left_alone_rather_than_mangled() {
        assert_eq!(report_path(Path::new("/other/a.py"), Path::new("/tree")), PathBuf::from("/other/a.py"));
    }

    // ------------------------------------------------------------ argument parsing

    #[test]
    fn every_subcommand_parses() {
        let parsed = [
            Cli::parse_from_args(["dupdelta", "languages"]).is_ok(),
            Cli::parse_from_args(["dupdelta", "scan", "."]).is_ok(),
            Cli::parse_from_args(["dupdelta", "diff", "--head", "h.json", "--base", "b.json"]).is_ok(),
            Cli::parse_from_args(["dupdelta", "ci", "--base", "main"]).is_ok(),
            Cli::parse_from_args(["dupdelta", "nonsense"]).is_ok(),
        ];
        assert_eq!(parsed, [true, true, true, true, false]);
    }

    #[test]
    fn ci_defaults_its_base_to_main() {
        let cli = Cli::parse_from_args(["dupdelta", "ci"]).expect("parses");
        assert!(format!("{cli:?}").contains("main"));
    }

    // ---------------------------------------------------------------- languages

    #[test]
    fn the_languages_command_lists_every_registered_language() {
        let (count, output) = run_cli(&["dupdelta", "languages"]);
        assert_eq!(count.expect("succeeds"), 0);
        let listed = output.lines().count();
        assert_eq!(listed, lang::all().len());
        assert!(output.contains("python"));
    }

    #[test]
    fn languages_stops_rather_than_silently_dropping_lines_when_stdout_refuses_writes() {
        let result = run_cli_with(&["dupdelta", "languages"], &mut FailingWriter);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Io"));
    }

    // --------------------------------------------------------------------- scan

    #[test]
    fn scan_writes_a_report_naming_paths_relative_to_the_root() {
        let dir = TempTree::new("cli");
        dir.write("pkg/twins.py", TWINS);
        let out = dir.join("report.json");

        let (count, _) = run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ]);
        count.expect("scan succeeds");

        let report = Report::read_from(&out).expect("report is readable");
        assert_eq!(report.files_scanned, 1);
        assert!(report.clones.iter().all(|c| c.a.file == "pkg/twins.py"));
    }

    #[test]
    fn scan_finds_a_renamed_copy_of_a_function() {
        let dir = TempTree::new("cli");
        dir.write("twins.py", TWINS);
        let out = dir.join("report.json");
        run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .0
        .expect("scan succeeds");

        let report = Report::read_from(&out).expect("report is readable");
        let names: Vec<(String, String)> =
            report.clones.iter().map(|c| (c.a.qualname.clone(), c.b.qualname.clone())).collect();
        assert_eq!(names, vec![("total".to_string(), "compute".to_string())]);
    }

    #[test]
    fn scan_without_an_output_path_writes_the_report_to_stdout() {
        let dir = TempTree::new("cli");
        dir.write("a.py", "x = 1\n");
        let (count, output) = run_cli(&["dupdelta", "scan", "--root", dir.path().to_str().unwrap()]);
        assert_eq!(count.expect("succeeds"), 0);
        assert!(Report::from_json(&output).is_ok());
    }

    #[test]
    fn scan_stops_rather_than_silently_dropping_the_report_when_stdout_refuses_writes() {
        let dir = TempTree::new("cli");
        dir.write("a.py", "x = 1\n");
        let result =
            run_cli_with(&["dupdelta", "scan", "--root", dir.path().to_str().unwrap()], &mut FailingWriter);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Io"));
    }

    #[test]
    fn scan_writing_its_report_to_an_unwritable_destination_is_an_error() {
        let dir = TempTree::new("cli");
        dir.write("a.py", "x = 1\n");
        let (result, _) = run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--out",
            "/nonexistent/dupdelta/report.json",
        ]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Report"));
    }

    #[test]
    fn scan_narrows_to_the_given_paths_while_still_naming_them_from_the_root() {
        let dir = TempTree::new("cli");
        dir.write("kept/a.py", TWINS);
        dir.write("skipped/b.py", TWINS);
        let out = dir.join("report.json");
        let kept = dir.join("kept");

        run_cli(&[
            "dupdelta",
            "scan",
            kept.to_str().unwrap(),
            "--root",
            dir.path().to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .0
        .expect("scan succeeds");

        let report = Report::read_from(&out).expect("report is readable");
        assert_eq!(report.files_scanned, 1);
        assert!(report.clones.iter().all(|c| c.a.file.starts_with("kept/")));
    }

    #[test]
    fn scan_honours_an_extra_exclude() {
        let dir = TempTree::new("cli");
        dir.write("keep.py", TWINS);
        dir.write("vendor/skip.py", TWINS);
        let out = dir.join("report.json");

        run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--exclude",
            "/vendor/",
            "--out",
            out.to_str().unwrap(),
        ])
        .0
        .expect("scan succeeds");

        assert_eq!(Report::read_from(&out).expect("readable").files_scanned, 1);
    }

    #[test]
    fn scanning_a_root_that_does_not_exist_is_an_error_not_an_empty_report() {
        // The failure this whole tool exists to prevent: a scan that quietly
        // examines nothing and reports no duplication.
        let (result, _) = run_cli(&["dupdelta", "scan", "--root", "/nonexistent/dupdelta/tree"]);
        let error = result.expect_err("a missing root must fail");
        // Not `matches!`: its non-matching arm is user code compiled into
        // this crate and would sit permanently uncovered, since a passing
        // test never takes it (see CONTRIBUTING.md). `{:?}` on a `#[derive(Debug)]`
        // enum names the variant, so a substring check is an honest,
        // branch-free equivalent.
        assert!(format!("{error:?}").contains("Walk"));
    }

    #[test]
    fn scan_reports_a_file_it_could_not_parse_rather_than_dropping_it() {
        let dir = TempTree::new("cli");
        dir.write("broken.py", "def good(a):\n    return a\n\ndef !!! broken(\n");
        let out = dir.join("report.json");
        run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .0
        .expect("scan succeeds");

        let report = Report::read_from(&out).expect("readable");
        assert_eq!(report.files_with_syntax_errors, vec!["broken.py".to_string()]);
    }

    #[test]
    fn scan_reports_a_file_it_could_not_read_rather_than_dropping_it() {
        // The walker only checks that a path exists and matches a supported
        // extension; permission to open it is a separate fact, discovered
        // only when `read_to_string` actually tries. Losing that file here
        // would silently shrink the tree the same way a skipped directory
        // does — "no new duplication" indistinguishable from "half the tree
        // went unread".
        let dir = TempTree::new("cli");
        dir.write("unreadable.py", "value = 1\n");
        dir.make_unreadable("unreadable.py");

        let (result, _) = run_cli(&["dupdelta", "scan", "--root", dir.path().to_str().unwrap()]);

        assert!(format!("{:?}", result.expect_err("must fail")).contains("Io"));
    }

    // --------------------------------------------------------------------- diff

    #[test]
    fn diff_reports_only_what_the_head_report_added() {
        let dir = TempTree::new("cli");
        let base = dir.join("base.json");
        let head = dir.join("head.json");

        // Scan the same tree twice: nothing was added, so nothing is reported.
        let tree = TempTree::new("cli");
        tree.write("twins.py", TWINS);
        for out in [&base, &head] {
            run_cli(&[
                "dupdelta",
                "scan",
                "--root",
                tree.path().to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
            ])
            .0
            .expect("scan succeeds");
        }

        let (count, output) = run_cli(&[
            "dupdelta",
            "diff",
            "--head",
            head.to_str().unwrap(),
            "--base",
            base.to_str().unwrap(),
        ]);
        assert_eq!(count.expect("diff succeeds"), 0);
        assert!(output.contains("No new duplication"));
    }

    #[test]
    fn diff_reports_a_pair_the_head_tree_introduced() {
        let base_tree = TempTree::new("cli");
        base_tree.write("a.py", "def total(rate, years, base):\n    return rate\n");
        let head_tree = TempTree::new("cli");
        head_tree.write("a.py", TWINS);

        let reports = TempTree::new("cli");
        let base = reports.join("base.json");
        let head = reports.join("head.json");
        for (tree, out) in [(&base_tree, &base), (&head_tree, &head)] {
            run_cli(&[
                "dupdelta",
                "scan",
                "--root",
                tree.path().to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
            ])
            .0
            .expect("scan succeeds");
        }

        let (count, _) = run_cli(&[
            "dupdelta",
            "diff",
            "--head",
            head.to_str().unwrap(),
            "--base",
            base.to_str().unwrap(),
        ]);
        assert!(count.expect("diff succeeds") > 0);
    }

    #[test]
    fn diff_of_a_missing_report_fails_rather_than_comparing_against_nothing() {
        let (result, _) =
            run_cli(&["dupdelta", "diff", "--head", "/nonexistent/h.json", "--base", "/nonexistent/b.json"]);
        // See the comment on the `Walk` variant check above: `{:?}` substring
        // instead of `matches!`, to avoid a permanently-uncovered non-match arm.
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Report"));
    }

    #[test]
    fn diff_of_only_a_missing_base_report_fails_after_the_head_report_reads_fine() {
        let dir = TempTree::new("cli");
        let head = dir.join("head.json");
        Report::default().write_to(&head).expect("written");

        let (result, _) = run_cli(&[
            "dupdelta",
            "diff",
            "--head",
            head.to_str().unwrap(),
            "--base",
            "/nonexistent/dupdelta/base.json",
        ]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Report"));
    }

    #[test]
    fn diff_stops_rather_than_using_the_default_config_when_an_explicit_one_is_invalid() {
        let dir = TempTree::new("cli");
        let config = dir.write("bad.toml", "[function]\nmin_similarty = 0.9\n");
        let (result, _) = run_cli(&[
            "dupdelta",
            "diff",
            "--head",
            "/nonexistent/h.json",
            "--base",
            "/nonexistent/b.json",
            "--config",
            config.to_str().unwrap(),
        ]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Config"));
    }

    // ----------------------------------------------------------------- outputs

    #[test]
    fn annotations_and_a_summary_and_a_count_all_reach_their_destinations() {
        let base_tree = TempTree::new("cli");
        base_tree.write("a.py", "def total(rate, years, base):\n    return rate\n");
        let head_tree = TempTree::new("cli");
        head_tree.write("a.py", TWINS);

        let dir = TempTree::new("cli");
        let base = dir.join("base.json");
        let head = dir.join("head.json");
        for (tree, out) in [(&base_tree, &base), (&head_tree, &head)] {
            run_cli(&[
                "dupdelta",
                "scan",
                "--root",
                tree.path().to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
            ])
            .0
            .expect("scan succeeds");
        }

        let summary = dir.join("summary.md");
        let findings = dir.join("findings.txt");
        let (count, output) = run_cli(&[
            "dupdelta",
            "diff",
            "--head",
            head.to_str().unwrap(),
            "--base",
            base.to_str().unwrap(),
            "--github-annotations",
            "--summary",
            summary.to_str().unwrap(),
            "--findings-out",
            findings.to_str().unwrap(),
        ]);

        let count = count.expect("diff succeeds");
        assert!(output.contains("::warning file="));
        assert!(std::fs::read_to_string(&summary).expect("summary written").contains("|"));
        assert_eq!(std::fs::read_to_string(&findings).expect("count written"), count.to_string());
    }

    #[test]
    fn an_annotation_stops_rather_than_silently_dropping_itself_when_stdout_refuses_writes() {
        let base_tree = TempTree::new("cli");
        base_tree.write("a.py", "def total(rate, years, base):\n    return rate\n");
        let head_tree = TempTree::new("cli");
        head_tree.write("a.py", TWINS);

        let dir = TempTree::new("cli");
        let base = dir.join("base.json");
        let head = dir.join("head.json");
        for (tree, out) in [(&base_tree, &base), (&head_tree, &head)] {
            run_cli(&[
                "dupdelta",
                "scan",
                "--root",
                tree.path().to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
            ])
            .0
            .expect("scan succeeds");
        }

        let result = run_cli_with(
            &[
                "dupdelta",
                "diff",
                "--head",
                head.to_str().unwrap(),
                "--base",
                base.to_str().unwrap(),
                "--github-annotations",
            ],
            &mut FailingWriter,
        );
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Io"));
    }

    #[test]
    fn the_rendered_summary_stops_rather_than_silently_dropping_itself_when_stdout_refuses_writes() {
        let dir = TempTree::new("cli");
        let empty = Report::default();
        let base = dir.join("base.json");
        let head = dir.join("head.json");
        empty.write_to(&base).expect("written");
        empty.write_to(&head).expect("written");

        let result = run_cli_with(
            &["dupdelta", "diff", "--head", head.to_str().unwrap(), "--base", base.to_str().unwrap()],
            &mut FailingWriter,
        );
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Io"));
    }

    #[test]
    fn an_unwritable_summary_or_count_destination_is_an_error() {
        let dir = TempTree::new("cli");
        let empty = Report::default();
        let base = dir.join("base.json");
        let head = dir.join("head.json");
        empty.write_to(&base).expect("written");
        empty.write_to(&head).expect("written");

        let bad = "/nonexistent/dupdelta/out";
        let summary_failed = run_cli(&[
            "dupdelta",
            "diff",
            "--head",
            head.to_str().unwrap(),
            "--base",
            base.to_str().unwrap(),
            "--summary",
            bad,
        ])
        .0
        .is_err();
        let findings_failed = run_cli(&[
            "dupdelta",
            "diff",
            "--head",
            head.to_str().unwrap(),
            "--base",
            base.to_str().unwrap(),
            "--findings-out",
            bad,
        ])
        .0
        .is_err();
        assert_eq!([summary_failed, findings_failed], [true, true]);
    }

    // ------------------------------------------------------------------ config

    #[test]
    fn an_explicit_config_file_is_used() {
        let dir = TempTree::new("cli");
        dir.write("twins.py", TWINS);
        let config = dir.write("custom.toml", "[function]\nmin_similarity = 1.0\nmin_nodes = 9999\n");
        let out = dir.join("report.json");

        run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .0
        .expect("scan succeeds");

        // min_nodes of 9999 excludes every unit, so nothing can pair.
        let report = Report::read_from(&out).expect("readable");
        assert_eq!((report.units_considered, report.clones.len()), (0, 0));
    }

    #[test]
    fn a_discovered_config_file_is_used() {
        let dir = TempTree::new("cli");
        dir.write("twins.py", TWINS);
        dir.write(".dupdelta.toml", "[function]\nmin_nodes = 9999\n");
        let out = dir.join("report.json");

        run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .0
        .expect("scan succeeds");
        assert_eq!(Report::read_from(&out).expect("readable").units_considered, 0);
    }

    #[test]
    fn a_discovered_config_file_that_is_invalid_stops_the_run() {
        let dir = TempTree::new("cli");
        dir.write("twins.py", TWINS);
        dir.write(".dupdelta.toml", "[function]\nmin_similarty = 0.9\n");

        let (result, _) = run_cli(&["dupdelta", "scan", "--root", dir.path().to_str().unwrap()]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Config"));
    }

    #[test]
    fn an_invalid_config_file_stops_the_run() {
        let dir = TempTree::new("cli");
        let config = dir.write("bad.toml", "[function]\nmin_similarty = 0.9\n");
        let (result, _) = run_cli(&[
            "dupdelta",
            "scan",
            "--root",
            dir.path().to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
        ]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Config"));
    }

    // ------------------------------------------------------------- ci internals

    #[test]
    fn default_scan_root_falls_back_to_the_current_directory_when_no_path_was_given() {
        assert_eq!(default_scan_root(&[]), PathBuf::from("."));
    }

    #[test]
    fn default_scan_root_uses_the_first_given_path() {
        assert_eq!(default_scan_root(&[PathBuf::from("a"), PathBuf::from("b")]), PathBuf::from("a"));
    }

    #[test]
    fn a_relative_path_is_kept_relative_rather_than_resolved_against_the_repository() {
        // A relative `--path` is passed through untouched: it is later joined
        // onto each tree's own root, so resolving it here (against whichever
        // root happens to be current) would silently pick the wrong tree.
        let root = TempTree::new("cli");
        let relative =
            repo_relative_paths(&[PathBuf::from("src/a.py")], root.path()).expect("stays relative");
        assert_eq!(relative, vec![PathBuf::from("src/a.py")]);
    }

    #[test]
    fn a_root_that_cannot_be_canonicalized_is_used_as_given() {
        // If the repository root itself cannot be canonicalized (e.g. it no
        // longer exists), falling back to the given root — rather than
        // failing outright — still lets a relative `--path` (the common
        // case) resolve correctly.
        let root = Path::new("/nonexistent/dupdelta/repo-root");
        let relative =
            repo_relative_paths(&[PathBuf::from("a.py")], root).expect("relative path needs no root");
        assert_eq!(relative, vec![PathBuf::from("a.py")]);
    }

    #[test]
    fn an_absolute_path_that_cannot_be_canonicalized_is_refused_rather_than_guessed_at() {
        // The path itself doesn't exist, so it cannot be canonicalized either;
        // the fallback keeps it as given, and it is then correctly refused
        // for not being under the (real, existing) repository root.
        let root = TempTree::new("cli");
        let missing = PathBuf::from("/nonexistent/dupdelta/does-not-exist");
        let result = repo_relative_paths(&[missing], root.path());
        assert!(format!("{:?}", result.expect_err("must fail")).contains("PathOutsideRepository"));
    }

    // ----------------------------------------------------------------------- ci

    #[test]
    fn ci_compares_the_working_tree_against_its_merge_base() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "def total(rate, years, base):\n    return rate\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base"]);
        repo.git(&["checkout", "-qb", "feature"]);
        repo.write("a.py", TWINS);
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "add a twin"]);

        let findings = repo.join("findings.txt");
        let head_report = repo.join("head.json");
        let (count, _) = run_cli(&[
            "dupdelta",
            "ci",
            "--base",
            "main",
            "--path",
            repo.path().to_str().unwrap(),
            "--report",
            head_report.to_str().unwrap(),
            "--findings-out",
            findings.to_str().unwrap(),
        ]);

        let count = count.expect("ci succeeds");
        // Bare assert!: the branch introduced a renamed copy, so this must be
        // positive; a custom message here would itself be a failure-only
        // branch (see CONTRIBUTING.md).
        assert!(count > 0);
        assert!(Report::read_from(&head_report).is_ok());
    }

    #[test]
    fn ci_is_silent_when_a_branch_adds_no_duplication() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", TWINS);
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base already has the duplication"]);
        repo.git(&["checkout", "-qb", "feature"]);
        repo.write("note.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "an unrelated change"]);

        // The pre-existing duplicate is in the merge-base, so it stays silent.
        let (count, output) =
            run_cli(&["dupdelta", "ci", "--base", "main", "--path", repo.path().to_str().unwrap()]);
        assert_eq!(count.expect("ci succeeds"), 0);
        assert!(output.contains("No new duplication"));
    }

    #[test]
    fn ci_outside_a_repository_fails_rather_than_scanning_one_tree() {
        let (result, _) = run_cli(&["dupdelta", "ci", "--path", "/nonexistent/dupdelta/tree"]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Git"));
    }

    #[test]
    fn an_unresolvable_base_falls_back_to_its_remote_tracking_ref() {
        // actions/checkout with fetch-depth: 0 — the checkout dupdelta's own
        // action requires — leaves HEAD detached and creates no local
        // branches, so the default base `main` resolves to nothing even
        // though `origin/main` is right there. Reproduce that exact layout
        // with a real remote: the comparison must go ahead against the
        // remote-tracking ref, not fail asking the user to know better.
        let upstream = TempTree::new("cli");
        upstream.git(&["init", "-q", "."]);
        upstream.write("a.py", "value = 1\n");
        upstream.git(&["add", "-A"]);
        upstream.git(&["commit", "-qm", "base"]);

        let repo = TempTree::new("cli");
        repo.git(&["clone", "-q", upstream.path().to_str().unwrap(), "."]);
        repo.git(&["checkout", "-q", "--detach", "origin/main"]);
        repo.git(&["branch", "-qD", "main"]);
        repo.write("b.py", "value = 2\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "head"]);

        let (result, output) =
            run_cli(&["dupdelta", "ci", "--base", "main", "--path", repo.path().to_str().unwrap()]);
        // Bare assert!: the branch added a file with no twin in the base, so
        // a positive count is the whole point; a custom message here would
        // itself be a failure-only branch (see CONTRIBUTING.md).
        assert_eq!(result.expect("ci succeeds via the remote-tracking ref"), 0);
        assert!(output.contains("No new duplication"));
    }

    #[test]
    fn ci_refuses_a_path_outside_the_repository() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base"]);

        let outside = TempTree::new("cli");
        outside.write("b.py", "value = 2\n");

        // The first `--path` is inside the repository (and doubles as the
        // start point for repository discovery); the second is genuinely
        // outside it. Without the refusal, the outside path would resolve
        // inside *both* trees `ci` compares — the working tree, twice — and
        // every duplicate already in it would be reported as newly introduced.
        let (result, _) = run_cli(&[
            "dupdelta",
            "ci",
            "--path",
            repo.path().to_str().unwrap(),
            "--path",
            outside.path().to_str().unwrap(),
        ]);

        use std::error::Error;
        let error = result.expect_err("a path outside the repository must be refused");
        assert!(error.to_string().contains("outside the repository"));
        assert!(error.source().is_none());
    }

    #[test]
    fn ci_with_an_invalid_config_file_stops_the_run() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base"]);
        let config = repo.write("bad.toml", "[function]\nmin_similarty = 0.9\n");

        let (result, _) = run_cli(&[
            "dupdelta",
            "ci",
            "--path",
            repo.path().to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
        ]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Config"));
    }

    #[test]
    fn ci_on_a_repository_with_no_commits_cannot_resolve_head() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);

        let (result, _) = run_cli(&["dupdelta", "ci", "--path", repo.path().to_str().unwrap()]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Git"));
    }

    #[test]
    fn ci_with_an_unresolvable_base_revision_stops_the_run() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base"]);

        let (result, _) =
            run_cli(&["dupdelta", "ci", "--base", "nosuch", "--path", repo.path().to_str().unwrap()]);
        let error = result.expect_err("must fail");
        use std::error::Error as _;
        // git's own stderr is passed through verbatim (and is localized, so
        // only the revision name is safe to match on); the message names both
        // refs that were tried and what would fix it.
        assert!(error.to_string().contains("could not resolve the base revision 'nosuch'"));
        assert!(error.to_string().contains("origin/nosuch"));
        assert!(error.source().is_some());
    }

    #[test]
    fn ci_between_unrelated_histories_has_no_merge_base() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "on main"]);
        repo.git(&["checkout", "-q", "--orphan", "unrelated"]);
        repo.write("b.py", "value = 2\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "on unrelated"]);
        repo.git(&["checkout", "-q", "main"]);

        let (result, _) =
            run_cli(&["dupdelta", "ci", "--base", "unrelated", "--path", repo.path().to_str().unwrap()]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Git"));
    }

    #[test]
    fn ci_scanning_a_path_that_does_not_exist_in_the_head_tree_is_an_error() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base"]);
        let missing = repo.join("missing");

        let (result, _) = run_cli(&[
            "dupdelta",
            "ci",
            "--base",
            "main",
            "--path",
            repo.path().to_str().unwrap(),
            "--path",
            missing.to_str().unwrap(),
        ]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Walk"));
    }

    #[test]
    fn ci_writing_its_head_report_to_an_unwritable_destination_is_an_error() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base"]);

        let (result, _) = run_cli(&[
            "dupdelta",
            "ci",
            "--base",
            "main",
            "--path",
            repo.path().to_str().unwrap(),
            "--report",
            "/nonexistent/dupdelta/head.json",
        ]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Report"));
    }

    #[test]
    fn ci_refuses_a_stale_non_worktree_directory_occupying_the_merge_base_checkout_path() {
        let repo_dir = TempTree::new("cli");
        repo_dir.git(&["init", "-q", "."]);
        repo_dir.write("a.py", "value = 1\n");
        repo_dir.git(&["add", "-A"]);
        repo_dir.git(&["commit", "-qm", "base"]);

        // `add_detached_worktree` only clears a *registered* worktree left at
        // this path by a killed prior run; anything else there is left alone
        // and the checkout fails loudly on it instead (see its doc comment).
        let repo = Repo::discover(repo_dir.path()).expect("repo discoverable");
        let occupied = base_worktree_path(repo.root());
        std::fs::create_dir_all(&occupied).expect("scratch dir is creatable");
        std::fs::write(occupied.join("occupied.txt"), "not a worktree\n").expect("scratch file is writable");

        let (result, _) =
            run_cli(&["dupdelta", "ci", "--base", "main", "--path", repo_dir.path().to_str().unwrap()]);

        let _ = std::fs::remove_dir_all(&occupied);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Git"));
    }

    #[test]
    fn ci_scanning_a_path_missing_from_the_merge_base_tree_is_an_error() {
        let repo = TempTree::new("cli");
        repo.git(&["init", "-q", "."]);
        repo.write("a.py", "value = 1\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "base"]);
        let base_commit = repo.git(&["rev-parse", "HEAD"]);
        repo.write("newdir/b.py", "value = 2\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-qm", "add newdir"]);
        let newdir = repo.join("newdir");

        let (result, _) =
            run_cli(&["dupdelta", "ci", "--base", &base_commit, "--path", newdir.to_str().unwrap()]);
        assert!(format!("{:?}", result.expect_err("must fail")).contains("Walk"));
    }

    // ------------------------------------------------------------------ errors

    #[test]
    fn every_error_variant_displays_and_exposes_its_cause() {
        use std::error::Error;
        let io = CliError::Io {
            path: PathBuf::from("/some/file"),
            source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        };
        let walk: CliError = walk::discover(
            &WalkOptions { roots: vec![PathBuf::from("/nonexistent/x")], ..WalkOptions::default() },
            |_| true,
        )
        .unwrap_err()
        .into();
        let report: CliError = Report::read_from(Path::new("/nonexistent/r.json")).unwrap_err().into();
        let config: CliError = Config::load(Path::new("/nonexistent/c.toml")).unwrap_err().into();
        let git: CliError = Repo::discover(Path::new("/nonexistent/repo")).unwrap_err().into();
        let path_outside = CliError::PathOutsideRepository {
            path: PathBuf::from("/scan/outside/a.py"),
            root: PathBuf::from("/scan/repo"),
        };

        let all = [&io, &walk, &report, &config, &git, &path_outside];
        assert_eq!(all.iter().filter(|e| e.source().is_some()).count(), 5);
        assert_eq!(all.iter().filter(|e| !e.to_string().is_empty()).count(), 6);
        assert!(io.to_string().contains("/some/file"));
        assert!(format!("{io:?}").contains("Io"));
        assert!(path_outside.source().is_none());
        assert!(path_outside.to_string().contains("/scan/outside/a.py"));
        assert!(path_outside.to_string().contains("/scan/repo"));
    }
}
