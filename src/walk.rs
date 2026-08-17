//! Source-file discovery for a scan.
//!
//! Everything downstream — tokenizing, similarity, delta reporting — only
//! ever sees the files this module decided were worth looking at. That makes
//! it the one place where "the tool quietly scanned nothing" can originate,
//! so its failure modes are deliberately narrow: a root that does not exist,
//! or a directory that cannot be read, is an [`Err`], never an empty [`Vec`].
//! An empty result is only ever produced by a real, readable tree that
//! genuinely contains nothing [`discover`]'s caller asked for — that is a
//! legitimate zero, not a swallowed error.
//!
//! # Why `ignore` and not a hand-rolled walk
//!
//! The [`ignore`] crate is what ripgrep is built on: it knows `.gitignore` /
//! `.ignore` semantics, follows-or-not symlinks without looping, and can
//! prune a directory from the walk before descending into it (rather than
//! filtering its contents out afterwards, which would still pay the cost of
//! — and still fail on — reading an unreadable subtree we never wanted).
//!
//! Two of its defaults are deliberately overridden here, unconditionally:
//!
//! - `git_global(false)` — the crate's default reads the *invoking user's*
//!   global `~/.gitconfig` excludes. A scan whose result depends on who
//!   happened to run it, on which machine, is exactly the kind of silent
//!   non-determinism this tool exists to avoid.
//! - `require_git(false)` — the crate's default only honours `.gitignore`
//!   files inside an actual git repository (a `.git` directory somewhere
//!   above the root). `respect_gitignore` in [`WalkOptions`] promises to
//!   honour `.gitignore` files, full stop; a tree that has one but no `.git`
//!   (e.g. an extracted archive) should not be treated differently.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

/// Which files a scan should consider.
#[derive(Debug, Clone)]
pub struct WalkOptions {
    /// Directories — or individual files — to discover from.
    pub roots: Vec<PathBuf>,
    /// Substring patterns. A file whose full path contains any of them is
    /// skipped, and a *directory* whose full path contains any of them is
    /// never descended into.
    pub excludes: Vec<String>,
    /// Honour `.gitignore` / `.ignore` files.
    pub respect_gitignore: bool,
    /// Follow symlinks.
    pub follow_symlinks: bool,
    /// Skip files larger than this. `None` means no limit.
    pub max_file_bytes: Option<u64>,
}

impl Default for WalkOptions {
    fn default() -> Self {
        WalkOptions {
            roots: Vec::new(),
            excludes: Vec::new(),
            respect_gitignore: true,
            follow_symlinks: false,
            max_file_bytes: None,
        }
    }
}

/// Why [`discover`] could not produce a file list.
///
/// There is no "root not found, treated as empty" variant: that shape is
/// exactly the silent zero this crate refuses to produce. See the module
/// docs.
#[derive(Debug)]
pub enum WalkError {
    /// A configured root does not exist.
    MissingRoot(PathBuf),
    /// Reading a directory entry failed — permissions, a vanished file
    /// between listing and stat, a symlink loop, and so on.
    Io {
        /// The path the failing operation was on, when the underlying error
        /// identifies one; otherwise the root being walked.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
}

impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WalkError::MissingRoot(path) => {
                write!(f, "walk root does not exist: {}", path.display())
            }
            WalkError::Io { path, source } => {
                write!(f, "failed to read {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for WalkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WalkError::MissingRoot(_) => None,
            WalkError::Io { source, .. } => Some(source),
        }
    }
}

/// `true` if `path`'s full path string contains any of `excludes`.
fn excluded(path: &Path, excludes: &[String]) -> bool {
    let text = path.to_string_lossy();
    excludes.iter().any(|pattern| text.contains(pattern.as_str()))
}

/// `true` if `bytes` is within `max`. `None` means no limit.
fn within_limit(bytes: u64, max: Option<u64>) -> bool {
    match max {
        Some(limit) => bytes <= limit,
        None => true,
    }
}

/// Best-effort path attribution for an [`ignore::Error`].
///
/// Errors from a failed directory read come pre-wrapped by the crate as
/// [`ignore::Error::WithPath`], and that is the only shape this bothers to
/// unwrap. A symlink-loop error is reported differently (`WithDepth` around
/// `Loop`, no `WithPath`) — that case, and any other shape, falls back to the
/// root currently being walked at the call site rather than digging further;
/// the path is diagnostic context, not something the caller branches on, so
/// guessing wrong costs nothing and a bare `None` is the honest answer.
fn ignore_error_path(err: &ignore::Error) -> Option<PathBuf> {
    if let ignore::Error::WithPath { path, .. } = err {
        Some(path.clone())
    } else {
        None
    }
}

/// Converts an [`ignore::Error`] to the [`std::io::Error`] it wraps, or a
/// synthesized one (e.g. for a symlink loop, which carries no `io::Error`)
/// that preserves the original message.
fn to_io_error(err: &ignore::Error) -> std::io::Error {
    match err.io_error() {
        Some(io_err) => std::io::Error::new(io_err.kind(), err.to_string()),
        None => std::io::Error::other(err.to_string()),
    }
}

/// Discover files under the configured roots, deterministically sorted and
/// deduplicated.
///
/// `accept` decides whether a discovered path is interesting; in production
/// the language registry supplies it (e.g. "is this a `.py` file"). It never
/// influences directory pruning — that is `excludes`' job alone — so a
/// directory containing nothing `accept` wants is still walked, and an
/// `accept` that rejects everything is a legitimate empty result, not an
/// error.
pub fn discover<F>(options: &WalkOptions, accept: F) -> Result<Vec<PathBuf>, WalkError>
where
    F: Fn(&Path) -> bool,
{
    let mut found: HashSet<PathBuf> = HashSet::new();

    for root in &options.roots {
        let metadata = std::fs::metadata(root).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                WalkError::MissingRoot(root.clone())
            } else {
                WalkError::Io { path: root.clone(), source }
            }
        })?;

        if metadata.is_file() {
            if !excluded(root, &options.excludes)
                && within_limit(metadata.len(), options.max_file_bytes)
                && accept(root)
            {
                found.insert(root.clone());
            }
            continue;
        }

        walk_directory(root, options, &accept, &mut found)?;
    }

    let mut files: Vec<PathBuf> = found.into_iter().collect();
    files.sort();
    Ok(files)
}

/// Walks a single directory root, adding accepted files to `found`.
fn walk_directory<F>(
    root: &Path,
    options: &WalkOptions,
    accept: &F,
    found: &mut HashSet<PathBuf>,
) -> Result<(), WalkError>
where
    F: Fn(&Path) -> bool,
{
    let excludes = options.excludes.clone();
    let mut builder = WalkBuilder::new(root);
    builder
        .standard_filters(options.respect_gitignore)
        // See the module docs: always off, regardless of respect_gitignore.
        .git_global(false)
        .require_git(false)
        .follow_links(options.follow_symlinks)
        .max_filesize(options.max_file_bytes)
        .filter_entry(move |entry| !excluded(entry.path(), &excludes));

    for entry in builder.build() {
        match entry {
            Ok(entry) => {
                let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
                if is_file && accept(entry.path()) {
                    found.insert(entry.path().to_path_buf());
                }
            }
            Err(err) => {
                let path = ignore_error_path(&err).unwrap_or_else(|| root.to_path_buf());
                return Err(WalkError::Io { path, source: to_io_error(&err) });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempTree;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    fn options(roots: Vec<PathBuf>) -> WalkOptions {
        WalkOptions { roots, ..WalkOptions::default() }
    }

    fn py_files(path: &Path) -> bool {
        path.extension().is_some_and(|ext| ext == "py")
    }

    // -------------------------------------------------------------- defaults

    #[test]
    fn default_options_respect_gitignore_and_do_not_follow_symlinks() {
        let opts = WalkOptions::default();
        assert!(format!("{opts:?}").contains("respect_gitignore: true"));
        assert!(opts.respect_gitignore);
        assert!(!opts.follow_symlinks);
        assert_eq!(opts.max_file_bytes, None);
        assert!(opts.roots.is_empty());
        assert!(opts.excludes.is_empty());
    }

    // ---------------------------------------------------------- missing root

    #[test]
    fn a_root_that_does_not_exist_is_an_error_not_an_empty_result() {
        let tree = TempTree::new("walk");
        let missing = tree.path().join("does-not-exist");
        let result = discover(&options(vec![missing.clone()]), py_files);
        let is_missing_root = matches!(result, Err(WalkError::MissingRoot(p)) if p == missing);
        assert!(is_missing_root);
    }

    #[test]
    fn a_root_that_cannot_be_read_is_an_io_error() {
        let tree = TempTree::new("walk");
        let root = tree.path().to_path_buf();
        // `stat`ing `root` still succeeds (that only needs search permission
        // on *its parent*), so this reaches the walk itself, which then
        // fails to list `root`'s contents — unlike a root that does not
        // exist at all, caught earlier by `discover`'s own existence check.
        fs::set_permissions(&root, fs::Permissions::from_mode(0o000)).expect("chmod 000");
        let result = discover(&options(vec![root]), py_files);
        let is_io_error = matches!(result, Err(WalkError::Io { .. }));
        assert!(is_io_error);
    }

    #[test]
    fn a_root_whose_own_existence_check_is_denied_is_an_io_error() {
        // Distinct from the previous test: here `stat` itself fails, because
        // the *parent* lacks search permission — so `discover` never even
        // reaches the walk. Both are "cannot be read", surfaced at different
        // points, and both must produce `WalkError::Io`, never `MissingRoot`.
        let tree = TempTree::new("walk");
        tree.dir("locked-parent/child");
        let child = tree.path().join("locked-parent/child");
        tree.make_unreadable("locked-parent");

        let result = discover(&options(vec![child]), py_files);
        let is_io_error = matches!(result, Err(WalkError::Io { .. }));
        assert!(is_io_error);
    }

    // -------------------------------------------------------------- file root

    #[test]
    fn a_file_root_is_included_directly_without_being_walked() {
        let tree = TempTree::new("walk");
        let file = tree.write("solo.py", "x = 1\n");
        let files = discover(&options(vec![file.clone()]), py_files).unwrap();
        assert_eq!(files, vec![file]);
    }

    #[test]
    fn a_file_root_rejected_by_accept_yields_no_files() {
        let tree = TempTree::new("walk");
        let file = tree.write("solo.txt", "not python\n");
        let files = discover(&options(vec![file]), py_files).unwrap();
        assert!(files.is_empty());
    }

    // --------------------------------------------------- sorted / deduplicated

    #[test]
    fn overlapping_roots_yield_each_file_once_sorted() {
        let tree = TempTree::new("walk");
        tree.write("pkg/a.py", "a\n");
        tree.write("pkg/b.py", "b\n");
        let pkg = tree.path().join("pkg");
        let files = discover(&options(vec![tree.path().to_path_buf(), pkg]), py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("pkg/a.py"), tree.path().join("pkg/b.py")]);
    }

    // ------------------------------------------------------------- excludes

    #[test]
    fn an_excluded_directory_is_never_descended_into() {
        let tree = TempTree::new("walk");
        tree.write("keep/keep.py", "keep\n");
        tree.dir("vendor");
        // `vendor` itself — not just something inside it — is unreadable. A
        // walk that merely filtered `vendor`'s contents out afterwards would
        // still have to open `vendor` to enumerate them, and that would fail
        // here. A clean `Ok` is only possible if the walker never attempted
        // to read `vendor` at all, i.e. pruned it before descending.
        tree.make_unreadable("vendor");

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.excludes = vec!["vendor".to_string()];
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("keep/keep.py")]);
    }

    #[test]
    fn excludes_also_filter_individual_files() {
        let tree = TempTree::new("walk");
        tree.write("keep.py", "keep\n");
        tree.write("generated.py", "generated\n");

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.excludes = vec!["generated".to_string()];
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("keep.py")]);
    }

    // -------------------------------------------------------- .gitignore

    #[test]
    fn respect_gitignore_true_hides_what_gitignore_hides() {
        let tree = TempTree::new("walk");
        tree.write(".gitignore", "hidden.py\n");
        tree.write("hidden.py", "hidden\n");
        tree.write("visible.py", "visible\n");

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.respect_gitignore = true;
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("visible.py")]);
    }

    #[test]
    fn respect_gitignore_false_finds_what_gitignore_would_hide() {
        let tree = TempTree::new("walk");
        tree.write(".gitignore", "hidden.py\n");
        tree.write("hidden.py", "hidden\n");
        tree.write("visible.py", "visible\n");

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.respect_gitignore = false;
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("hidden.py"), tree.path().join("visible.py")]);
    }

    // ------------------------------------------------------- max_file_bytes

    #[test]
    fn max_file_bytes_skips_files_over_the_limit() {
        let tree = TempTree::new("walk");
        tree.write("small.py", "x\n");
        tree.write("big.py", "0123456789");

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.max_file_bytes = Some(3);
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("small.py")]);
    }

    #[test]
    fn max_file_bytes_none_skips_nothing() {
        let tree = TempTree::new("walk");
        tree.write("big.py", "0123456789");

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.max_file_bytes = None;
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("big.py")]);
    }

    #[test]
    fn max_file_bytes_boundary_is_inclusive() {
        let tree = TempTree::new("walk");
        let file = tree.write("exact.py", "12345");
        assert_eq!(fs::metadata(&file).unwrap().len(), 5);

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.max_file_bytes = Some(5);
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![file]);
    }

    #[test]
    fn max_file_bytes_applies_to_a_file_root_too() {
        let tree = TempTree::new("walk");
        let file = tree.write("big.py", "0123456789");

        let mut opts = options(vec![file]);
        opts.max_file_bytes = Some(3);
        let files = discover(&opts, py_files).unwrap();
        assert!(files.is_empty());
    }

    // ------------------------------------------------------------- symlinks

    #[test]
    fn follow_symlinks_false_does_not_hang_on_a_symlink_loop() {
        let tree = TempTree::new("walk");
        tree.write("real.py", "real\n");
        tree.symlink("loop", tree.path());

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.follow_symlinks = false;
        let files = discover(&opts, py_files).unwrap();
        assert_eq!(files, vec![tree.path().join("real.py")]);
    }

    #[test]
    fn follow_symlinks_true_reports_a_symlink_loop_as_an_error_not_a_hang() {
        let tree = TempTree::new("walk");
        tree.symlink("loop", tree.path());

        let mut opts = options(vec![tree.path().to_path_buf()]);
        opts.follow_symlinks = true;
        let result = discover(&opts, py_files);
        let is_io_error = matches!(result, Err(WalkError::Io { .. }));
        assert!(is_io_error);
    }

    // ---------------------------------------------------------------- accept

    #[test]
    fn accept_rejecting_everything_is_an_empty_result_not_an_error() {
        let tree = TempTree::new("walk");
        tree.write("a.py", "a\n");
        tree.write("b.py", "b\n");

        let files = discover(&options(vec![tree.path().to_path_buf()]), |_| false).unwrap();
        assert!(files.is_empty());
    }

    // ---------------------------------------------------------------- errors

    #[test]
    fn walk_error_display_and_source() {
        let missing = WalkError::MissingRoot(PathBuf::from("/nowhere"));
        let io =
            WalkError::Io { path: PathBuf::from("/nowhere/child"), source: std::io::Error::other("denied") };

        let displays: Vec<String> = vec![missing.to_string(), io.to_string()];
        assert_eq!(
            displays,
            vec![
                "walk root does not exist: /nowhere".to_string(),
                "failed to read /nowhere/child: denied".to_string(),
            ]
        );

        let sources: Vec<bool> =
            vec![std::error::Error::source(&missing).is_some(), std::error::Error::source(&io).is_some()];
        assert_eq!(sources, vec![false, true]);

        assert!(format!("{missing:?}").contains("MissingRoot"));
    }
}
