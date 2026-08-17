//! Shared test-only scratch-directory fixture.
//!
//! Six modules used to each hand-roll their own version of "a unique
//! directory under the OS temp root, named from the process id plus a
//! counter, removed on drop." This is the single copy. Per `CONTRIBUTING.md`,
//! an unused method here is an uncovered line and fails the coverage gate,
//! so this exposes exactly the union of what real call sites need — no more.

use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A unique scratch directory tree, removed on drop.
///
/// `std::process::id()` disambiguates across parallel `cargo test`
/// processes (there is only one per run); the counter disambiguates across
/// the several such trees a single test process's tests create
/// concurrently. `label` is folded into the directory name purely so a tree
/// left behind by a killed process is traceable to the module that created
/// it.
pub(crate) struct TempTree {
    root: PathBuf,
}

impl TempTree {
    /// Creates the tree's root directory and returns a handle to it.
    pub(crate) fn new(label: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("dupdelta-test-{}-{n}-{label}", std::process::id()));
        fs::create_dir_all(&root).expect("create temp tree root");
        TempTree { root }
    }

    /// The tree's root directory.
    pub(crate) fn path(&self) -> &Path {
        &self.root
    }

    /// `rel`, resolved against the tree root. Does not touch the filesystem,
    /// so it is also how a caller names a path it wants to create itself.
    pub(crate) fn join(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    /// Writes `rel` with `contents`, creating any parent directories, and
    /// returns its absolute path.
    pub(crate) fn write(&self, rel: &str, contents: &str) -> PathBuf {
        let path = self.root.join(rel);
        fs::create_dir_all(path.parent().expect("a joined path always has a parent"))
            .expect("create parent dir");
        fs::write(&path, contents).expect("write file");
        path
    }

    /// Creates a directory at `rel` (and any missing parents) and returns
    /// its absolute path.
    pub(crate) fn dir(&self, rel: &str) -> PathBuf {
        let path = self.root.join(rel);
        fs::create_dir_all(&path).expect("create dir");
        path
    }

    /// Creates a symlink at `rel` pointing at `target`, and returns the
    /// symlink's absolute path.
    pub(crate) fn symlink(&self, rel: &str, target: &Path) -> PathBuf {
        let path = self.root.join(rel);
        fs::create_dir_all(path.parent().expect("a joined path always has a parent"))
            .expect("create parent dir");
        symlink(target, &path).expect("create symlink");
        path
    }

    /// Strips all permissions from `rel`, so it can be seen (`stat`) but not
    /// opened or listed. Returns its absolute path.
    pub(crate) fn make_unreadable(&self, rel: &str) -> PathBuf {
        let path = self.root.join(rel);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).expect("chmod 000");
        path
    }

    /// Runs `git` inside the tree, isolated from the machine's own config
    /// (identity, default branch, and both config-file locations pointed at
    /// `/dev/null`, so a test depends on nothing about the machine it runs
    /// on), and returns trimmed stdout.
    ///
    /// Every fixture invocation is expected to succeed, so success is
    /// asserted here rather than left to each call site: a bare `assert!`,
    /// not one with a formatted message, because the message would be a
    /// branch built only on failure and so permanently uncovered while
    /// tests pass (see `CONTRIBUTING.md`). A real `git` failure belongs to
    /// the module under test, exercised through production code, never
    /// through this fixture.
    pub(crate) fn git(&self, args: &[&str]) -> String {
        let base: &[&str] = &[
            "-c",
            "user.name=dupdelta-test",
            "-c",
            "user.email=dupdelta-test@example.invalid",
            "-c",
            "init.defaultBranch=main",
        ];
        let output = Command::new("git")
            .args(base)
            .args(args)
            .current_dir(&self.root)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .output()
            .expect("run git in test fixture");
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }
}

impl Drop for TempTree {
    fn drop(&mut self) {
        restore_permissions(&self.root);
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Widens permissions top-down so [`TempTree::drop`] can always remove the
/// tree, even one containing a directory or file a test locked down. Only
/// recurses through real directories (`DirEntry::file_type` does not follow
/// symlinks), so a symlink loop a test built cannot make this recurse
/// forever.
fn restore_permissions(path: &Path) {
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o755));
    // Combinators rather than `if let Ok(entries) = ...`: the widen-then-
    // list sequencing above means `read_dir` here never actually fails in
    // practice, and an `if let` would leave its untaken `Err` arm as a
    // permanently-zero region the coverage gate flags. `.flatten()` folds
    // "couldn't list" and "couldn't stat one entry" into the same silent
    // skip a failed `if let` would have produced, with no branch to leave
    // uncovered.
    for entry in fs::read_dir(path).into_iter().flatten().flatten() {
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            restore_permissions(&entry.path());
        }
    }
}

/// A [`SourceFile`] built from in-memory text, for detector tests.
///
/// `blocks` and `vocab` each had their own verbatim copy of this; dupdelta
/// flagged the pair, so it lives here once. Both languages are exposed
/// because both are used — an unused one would be an uncovered line.
fn source_file(path: &str, language: &str, text: &str) -> crate::extract::SourceFile {
    crate::extract::SourceFile {
        path: std::path::PathBuf::from(path),
        language: crate::lang::by_name(language).expect("language is registered"),
        text: text.to_string(),
    }
}

/// A Python [`SourceFile`] built from in-memory text.
pub(crate) fn python_file(path: &str, text: &str) -> crate::extract::SourceFile {
    source_file(path, "python", text)
}

/// A JavaScript [`SourceFile`] built from in-memory text.
pub(crate) fn javascript_file(path: &str, text: &str) -> crate::extract::SourceFile {
    source_file(path, "javascript", text)
}

/// Every unit a Python source string yields, at the most permissive threshold.
///
/// `extract` and `scan` held byte-identical copies of this; dupdelta flagged
/// the pair at similarity 1.00, so it lives here once. Both import it under
/// their old local name, which keeps their call sites unchanged.
pub(crate) fn python_units(source: &str) -> Vec<crate::extract::Unit> {
    let mut interner = crate::token::Interner::new();
    crate::extract::Extractor::new(crate::lang::by_name("python").expect("python is registered"))
        .extract(source, std::path::Path::new("sample.py"), 1, &mut interner)
        .units
}

/// A [`crate::report::VocabPair`] with plausible filler, for tests that care
/// about only a field or two of it.
pub(crate) fn vocab_pair(a: &str, b: &str, overlap: f64, zero_inbound: bool) -> crate::report::VocabPair {
    crate::report::VocabPair {
        a: a.to_string(),
        b: b.to_string(),
        overlap,
        shared: 12,
        a_vocabulary: 40,
        b_vocabulary: 30,
        a_inbound_imports: 0,
        b_inbound_imports: 3,
        zero_inbound,
        sample_shared: vec!["rate".to_string()],
    }
}
