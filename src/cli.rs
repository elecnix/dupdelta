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
        /// Paths to scan. Defaults to the working directory.
        paths: Vec<PathBuf>,
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
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Walk(e) => write!(f, "{e}"),
            CliError::Report(e) => write!(f, "{e}"),
            CliError::Config(e) => write!(f, "{e}"),
            CliError::Git(e) => write!(f, "{e}"),
            CliError::Io { path, source } => write!(f, "could not access {}: {source}", path.display()),
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

/// Read every supported file under `root`, with paths already relativized.
fn read_tree(root: &Path, config: &Config, extra_excludes: &[String]) -> Result<Vec<SourceFile>, CliError> {
    let mut excludes = config.excludes.clone();
    excludes.extend_from_slice(extra_excludes);

    let options = WalkOptions { roots: vec![root.to_path_buf()], excludes, ..WalkOptions::default() };
    let discovered = walk::discover(&options, |p| lang::is_supported(p))?;

    let mut files = Vec::with_capacity(discovered.len());
    for path in discovered {
        // A file that cannot be read is an error, not a file quietly missing
        // from the scan: a scan that skipped half a tree still reports "no new
        // duplication", which is indistinguishable from a clean result.
        let text = std::fs::read_to_string(&path)
            .map_err(|source| CliError::Io { path: path.clone(), source })?;
        let language = lang::for_path(&path).expect("the walker only accepted supported paths");
        files.push(SourceFile { path: report_path(&path, root), language, text });
    }
    Ok(files)
}

/// Scan a tree with every detector and assemble a report.
pub fn scan_tree(root: &Path, config: &Config, extra_excludes: &[String]) -> Result<Report, CliError> {
    let files = read_tree(root, config, extra_excludes)?;

    let mut interner = Interner::new();
    let mut units = Vec::new();
    let mut broken = Vec::new();
    for file in &files {
        let mut extractor = Extractor::new(file.language);
        let extraction =
            extractor.extract(&file.text, &file.path, config.function.min_nodes, &mut interner);
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

            Command::Scan { paths, out: destination, excludes, config } => {
                let root = paths.first().cloned().unwrap_or_else(|| PathBuf::from("."));
                let config = load_config(config.as_deref(), &root)?;
                let report = scan_tree(&root, &config, &excludes)?;
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
                let start = paths.first().cloned().unwrap_or_else(|| PathBuf::from("."));
                let repo = Repo::discover(&start)?;
                let config = load_config(config.as_deref(), repo.root())?;

                let head_commit = repo.resolve("HEAD")?;
                let base_commit = repo.resolve(&base)?;
                let merge_base = repo.merge_base(&base_commit, &head_commit)?;

                let head_report = scan_tree(repo.root(), &config, &excludes)?;
                if let Some(path) = &report {
                    head_report.write_to(path)?;
                }

                let worktree = repo.add_detached_worktree(&base_worktree_path(repo.root()), &merge_base)?;
                let base_report = scan_tree(worktree.path(), &config, &excludes)?;
                worktree.remove()?;

                let delta = Delta::compute(&head_report, &base_report, &delta_options(&config));
                emit(&delta, &output, out)
            }
        }
    }
}
