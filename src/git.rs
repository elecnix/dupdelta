//! Everything the tool needs from `git`, so no other module shells out.
//!
//! `dupdelta` answers "did this change make it worse?" by scanning two
//! trees — a branch and its merge-base — and diffing the findings. Every
//! fact that comparison depends on (what the merge-base *is*, which files
//! actually changed, what a commit's tree really contained) comes from
//! `git` itself, run as a subprocess. There is no fallback path: a `git`
//! command that fails is surfaced as an [`Err`], never mapped to an empty
//! result. An empty [`Repo::changed_files`] must mean "git compared these
//! two commits and found no difference," never "the command failed and we
//! moved on" — the latter would make every pre-existing duplicate look new
//! by comparing the branch against nothing.
//!
//! # Why shell out instead of a `git2`/`libgit2` binding
//!
//! The tool's own contract (see the crate docs) is to report only what a
//! diff *introduces*; the ground truth for "what changed between these two
//! commits" is whatever the user's own `git` would say, on their own
//! checkout, with their own config. Shelling out to the same `git` a
//! reviewer would run by hand keeps that ground truth singular.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A git repository on disk.
#[derive(Debug, Clone)]
pub struct Repo {
    root: PathBuf,
}

/// A detached worktree that removes itself when dropped.
///
/// Holds the repository root alongside the worktree path because every
/// worktree operation (`git worktree remove`) has to be run *from inside
/// the repository*, not from inside the worktree being torn down.
#[derive(Debug)]
pub struct Worktree {
    path: PathBuf,
    repo_root: PathBuf,
}

/// Why a `git` operation could not produce a result.
///
/// There is no "command failed, treated as empty" variant: that shape is
/// exactly the silent zero this module exists to refuse. See the module
/// docs.
#[derive(Debug)]
pub enum GitError {
    /// The `git` binary could not be run at all (not found, not executable,
    /// or the process could not be spawned for some other OS-level reason).
    NotAvailable(std::io::Error),
    /// No git repository was found at or above the given path.
    NotARepository(PathBuf),
    /// A git command ran but exited non-zero.
    CommandFailed {
        /// The arguments `git` was invoked with (not including `git` itself).
        args: Vec<String>,
        /// The process exit code, when the process actually exited (`None`
        /// if it was terminated by a signal).
        status: Option<i32>,
        /// `git`'s own stderr, verbatim. This is the detail that matters:
        /// swallowing it in favour of a generic "git failed" would throw
        /// away the one thing that tells a caller *why*.
        stderr: String,
    },
}

impl fmt::Display for GitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GitError::NotAvailable(source) => write!(f, "could not run `git`: {source}"),
            GitError::NotARepository(path) => {
                write!(f, "no git repository at or above {}", path.display())
            }
            GitError::CommandFailed { args, status, stderr } => {
                let status_text = match status {
                    Some(code) => format!("exit status {code}"),
                    None => "terminated by signal".to_string(),
                };
                write!(f, "git {} failed ({status_text}): {}", args.join(" "), stderr.trim())
            }
        }
    }
}

impl std::error::Error for GitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GitError::NotAvailable(source) => Some(source),
            GitError::NotARepository(_) | GitError::CommandFailed { .. } => None,
        }
    }
}

/// Runs `git` with `args` in `dir`, returning trimmed stdout.
///
/// Any non-zero exit becomes [`GitError::CommandFailed`] carrying git's own
/// stderr verbatim — never mapped away to an empty string or an empty
/// collection by a caller further up.
fn run(dir: &Path, args: &[&str]) -> Result<String, GitError> {
    run_named("git", dir, args)
}

/// Runs `program` with `args` in `dir`. Split out from [`run`] purely so a
/// test can exercise [`GitError::NotAvailable`] honestly — by naming a
/// program that really does not exist — rather than constructing that
/// variant by hand and hoping it matches what a real spawn failure looks
/// like.
fn run_named(program: &str, dir: &Path, args: &[&str]) -> Result<String, GitError> {
    let output =
        Command::new(program).args(args).current_dir(dir).output().map_err(GitError::NotAvailable)?;

    if !output.status.success() {
        return Err(GitError::CommandFailed {
            args: args.iter().map(|s| s.to_string()).collect(),
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// [`Repo::discover`]'s implementation, parameterized on the program name
/// for the same reason [`run_named`] is split from [`run`]: so a test can
/// exercise the "`git` itself could not be run" path honestly.
///
/// A failure here almost always means "no repository", but it is still
/// `git`'s own words that decide that, not an assumption made here: only
/// [`GitError::CommandFailed`] — an actual non-zero exit from `git` — is
/// folded into [`GitError::NotARepository`]. Any other failure (`git`
/// missing entirely) propagates as whatever [`run_named`] produced, so a
/// broken `git` install is never misreported as "you're not in a
/// repository".
fn discover_with(program: &str, start: &Path) -> Result<Repo, GitError> {
    match run_named(program, start, &["rev-parse", "--show-toplevel"]) {
        Ok(top) => Ok(Repo { root: PathBuf::from(top) }),
        Err(GitError::CommandFailed { .. }) => Err(GitError::NotARepository(start.to_path_buf())),
        Err(other) => Err(other),
    }
}

impl Repo {
    /// Finds the repository containing `start`.
    pub fn discover(start: &Path) -> Result<Repo, GitError> {
        discover_with("git", start)
    }

    /// The repository's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The best common ancestor of two commits.
    pub fn merge_base(&self, a: &str, b: &str) -> Result<String, GitError> {
        run(&self.root, &["merge-base", a, b])
    }

    /// Resolves a revision (branch, tag, `HEAD`, short hash, …) to a full
    /// commit id.
    pub fn resolve(&self, rev: &str) -> Result<String, GitError> {
        run(&self.root, &["rev-parse", rev])
    }

    /// Remote-tracking branch short names (`origin/main`), as git lists them.
    ///
    /// Exists to power the base-resolution hint in `ci`: on the checkout
    /// dupdelta's own GitHub Action requires (`actions/checkout` with
    /// `fetch-depth: 0`), the base branch exists only as a remote-tracking
    /// ref, and naming it turns git's bare "unknown revision" into the
    /// actual fix.
    pub fn remote_branches(&self) -> Result<Vec<String>, GitError> {
        let out = run(&self.root, &["for-each-ref", "--format=%(refname)", "refs/remotes/"])?;
        // refs/remotes/<remote>/HEAD is a symref newer git records when it
        // learns the remote's default branch. It is a pointer, not a branch,
        // and its short name ("origin") would only pollute the hint.
        Ok(out
            .lines()
            .filter(|refname| !refname.ends_with("/HEAD"))
            .map(|refname| refname.strip_prefix("refs/remotes/").unwrap_or(refname).to_string())
            .collect())
    }

    /// Files added, changed, renamed or copied between two commits,
    /// repo-relative, `/`-separated, sorted.
    ///
    /// Deliberately excludes deletions (`--diff-filter=ACMR` omits `D`): the
    /// tool cannot scan a file that no longer exists at `to`, so listing it
    /// as "changed" would just make every downstream lookup fail on a path
    /// nothing put there on purpose.
    pub fn changed_files(&self, from: &str, to: &str) -> Result<Vec<String>, GitError> {
        let out = run(&self.root, &["diff", "--name-only", "--diff-filter=ACMR", from, to])?;
        let mut files: Vec<String> =
            if out.is_empty() { Vec::new() } else { out.lines().map(str::to_string).collect() };
        files.sort();
        Ok(files)
    }

    /// Checks `commit` out into a fresh detached worktree at `path`.
    ///
    /// If `path` already exists — a leftover from a run that was killed
    /// before it could call [`Worktree::remove`] — `git worktree add`
    /// refuses with "already exists". Rather than deleting an arbitrary
    /// directory on disk (which would be wrong if `path` were reused for
    /// something that was never a worktree at all), this only clears
    /// *registered* worktrees: `worktree remove --force` plus `worktree
    /// prune` clean up exactly what a previous `add_detached_worktree` at
    /// this same path could have left behind. Anything else at `path` is
    /// left alone, and the `add` below fails loudly on it instead.
    pub fn add_detached_worktree(&self, path: &Path, commit: &str) -> Result<Worktree, GitError> {
        let path_str = path.to_string_lossy().into_owned();

        if path.exists() {
            let _ = run(&self.root, &["worktree", "remove", "--force", &path_str]);
            let _ = run(&self.root, &["worktree", "prune"]);
        }

        run(&self.root, &["worktree", "add", "--detach", &path_str, commit])?;
        Ok(Worktree { path: path.to_path_buf(), repo_root: self.root.clone() })
    }
}

impl Worktree {
    /// The worktree's path on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Removes the worktree now, surfacing any failure.
    pub fn remove(self) -> Result<(), GitError> {
        let path_str = self.path.to_string_lossy().into_owned();
        run(&self.repo_root, &["worktree", "remove", "--force", &path_str])?;
        Ok(())
    }
}

impl Drop for Worktree {
    /// Best-effort cleanup. Unlike [`Worktree::remove`], a failure here has
    /// nowhere to go — `drop` cannot return a `Result` — so it is swallowed
    /// deliberately, on the one path where swallowing is honest: the
    /// caller who wants to *know* about a removal failure has
    /// [`Worktree::remove`] to call instead.
    fn drop(&mut self) {
        let path_str = self.path.to_string_lossy().into_owned();
        let _ = run(&self.repo_root, &["worktree", "remove", "--force", &path_str]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempTree;
    use std::fs;

    /// A throwaway git repository under a [`TempTree`], isolated from the
    /// machine's own git config per the project's testing notes.
    ///
    /// Kept as a thin wrapper rather than folded into [`TempTree`] itself:
    /// `commit_file`/`delete_file`/`branch`/`checkout` are git-repository
    /// vocabulary specific to this module, not something the other five
    /// call sites the shared fixture serves have any use for.
    struct TempRepo {
        tree: TempTree,
    }

    impl TempRepo {
        fn new() -> Self {
            let tree = TempTree::new("git-repo");
            tree.git(&["init", "--quiet"]);
            TempRepo { tree }
        }

        fn path(&self) -> &Path {
            self.tree.path()
        }

        fn repo(&self) -> Repo {
            Repo::discover(self.path()).expect("discover fixture repo")
        }

        /// Writes `rel` with `contents`, stages it, commits, and returns the
        /// new commit's full id.
        fn commit_file(&self, rel: &str, contents: &str, message: &str) -> String {
            self.tree.write(rel, contents);
            self.tree.git(&["add", rel]);
            self.tree.git(&["commit", "--quiet", "-m", message]);
            self.tree.git(&["rev-parse", "HEAD"])
        }

        /// Stages the removal of `rel` and commits it.
        fn delete_file(&self, rel: &str, message: &str) -> String {
            fs::remove_file(self.tree.join(rel)).expect("remove fixture file");
            self.tree.git(&["add", rel]);
            self.tree.git(&["commit", "--quiet", "-m", message]);
            self.tree.git(&["rev-parse", "HEAD"])
        }

        fn branch(&self, name: &str) {
            self.tree.git(&["branch", name]);
        }

        fn checkout(&self, rev: &str) {
            self.tree.git(&["checkout", "--quiet", rev]);
        }
    }

    // -------------------------------------------------------- command failure

    #[test]
    fn a_failing_git_command_surfaces_gits_own_stderr() {
        let fixture = TempRepo::new();
        fixture.commit_file("a.txt", "a\n", "initial");
        let repo = fixture.repo();

        // Also drive it through `changed_files`, whose whole job is to list
        // what differs between two commits: on a `git` failure it must
        // propagate the error, not report "nothing differs" — that would
        // be exactly the silent-zero this module exists to refuse.
        fixture.commit_file("b.txt", "b\n", "second");
        let changed_result = repo.changed_files("no-such-rev-a", "HEAD");
        assert!(changed_result.is_err());

        let err = repo.merge_base("no-such-rev-a", "no-such-rev-b").unwrap_err();
        let display = err.to_string();
        assert!(display.contains("no-such-rev-a"));
    }

    #[test]
    fn a_missing_git_binary_is_not_available_not_a_silent_empty_result() {
        let fixture = TempRepo::new();
        let err =
            run_named("dupdelta-definitely-not-a-real-binary", fixture.path(), &["--version"]).unwrap_err();
        assert!(err.to_string().starts_with("could not run `git`"));
    }

    #[test]
    fn discover_propagates_a_broken_git_install_instead_of_calling_it_not_a_repository() {
        let fixture = TempRepo::new();
        let err = discover_with("dupdelta-definitely-not-a-real-binary", fixture.path()).unwrap_err();
        assert!(err.to_string().starts_with("could not run `git`"));
    }

    #[test]
    fn add_detached_worktree_with_an_unknown_commit_fails_loudly() {
        let fixture = TempRepo::new();
        fixture.commit_file("a.txt", "a\n", "initial");
        let repo = fixture.repo();
        let dest = TempTree::new("bad-commit");
        let wt_path = dest.join("wt");

        let result = repo.add_detached_worktree(&wt_path, "no-such-commit");

        assert!(result.is_err());
    }

    // -------------------------------------------------------------- discover

    #[test]
    fn discover_finds_the_repo_from_a_subdirectory() {
        let fixture = TempRepo::new();
        fixture.commit_file("nested/dir/file.txt", "x\n", "initial");
        let sub = fixture.path().join("nested/dir");

        let repo = Repo::discover(&sub).expect("discover from subdirectory");

        // `canonicalize` on both sides: the OS temp dir can itself be a
        // symlink (e.g. macOS `TMPDIR`), and git reports the resolved path.
        let want = fs::canonicalize(fixture.path()).expect("canonicalize fixture root");
        let got = fs::canonicalize(repo.root()).expect("canonicalize discovered root");
        assert_eq!(got, want);
    }

    #[test]
    fn discover_returns_not_a_repository_for_a_path_outside_any_repo() {
        let dir = TempTree::new("bare");
        let dir_path = dir.path().to_path_buf();

        // Honesty check, per the task: don't assume the ambient temp
        // directory tree is free of any git repository above it — walk
        // every real ancestor, up to the filesystem root, and confirm none
        // of them carries a `.git` before trusting the negative result
        // below. If this ever fails, the fixture location is the thing to
        // fix, not this assertion.
        let mut ancestors_checked = 0usize;
        for ancestor in dir_path.ancestors() {
            assert!(!ancestor.join(".git").exists());
            ancestors_checked += 1;
        }
        assert!(ancestors_checked > 1);

        let result = Repo::discover(&dir_path);
        let is_not_a_repository = matches!(result, Err(GitError::NotARepository(ref p)) if *p == dir_path);
        assert!(is_not_a_repository);
    }

    #[test]
    fn repo_root_returns_the_configured_root() {
        let fixture = TempRepo::new();
        fixture.commit_file("a.txt", "a\n", "initial");
        let repo = fixture.repo();
        let want = fs::canonicalize(fixture.path()).expect("canonicalize fixture root");
        let got = fs::canonicalize(repo.root()).expect("canonicalize repo root");
        assert_eq!(got, want);
    }

    // ------------------------------------------------------------ merge_base

    #[test]
    fn merge_base_of_two_branches_returns_their_actual_common_ancestor() {
        let fixture = TempRepo::new();
        let base = fixture.commit_file("a.txt", "a\n", "base");
        fixture.branch("feature");
        fixture.commit_file("a.txt", "a on main\n", "main-only");
        fixture.checkout("feature");
        fixture.commit_file("b.txt", "b\n", "feature-only");
        let repo = fixture.repo();

        let got = repo.merge_base("feature", "main").expect("merge base");

        assert_eq!(got, base);
    }

    // ------------------------------------------------------- remote_branches

    #[test]
    fn remote_branches_lists_remote_tracking_refs_by_short_name() {
        let upstream = TempRepo::new();
        upstream.commit_file("a.txt", "a\n", "base");
        let fixture = TempRepo::new();
        fixture.commit_file("b.txt", "b\n", "head");
        fixture.tree.git(&["remote", "add", "origin", upstream.path().to_str().unwrap()]);
        fixture.tree.git(&["fetch", "--quiet", "origin"]);
        // Newer git records the remote's default branch as a HEAD symref at
        // fetch time; pin one explicitly so the skip is exercised on every
        // git version, not only where the fetch happens to create it.
        fixture.tree.git(&["symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"]);
        let repo = fixture.repo();

        assert_eq!(repo.remote_branches().expect("remote branches"), vec!["origin/main".to_string()]);
    }

    // --------------------------------------------------------- changed_files

    #[test]
    fn changed_files_is_exact_repo_relative_sorted_and_excludes_deletions() {
        let fixture = TempRepo::new();
        let from = fixture.commit_file("keep.txt", "keep\n", "add keep");
        fixture.commit_file("doomed.txt", "doomed\n", "add doomed");
        fixture.commit_file("z/new.txt", "z\n", "add z/new");
        fixture.commit_file("keep.txt", "keep, changed\n", "modify keep");
        let to = fixture.delete_file("doomed.txt", "delete doomed");
        let repo = fixture.repo();

        let got = repo.changed_files(&from, &to).expect("changed files");

        assert_eq!(got, vec!["keep.txt".to_string(), "z/new.txt".to_string()]);
    }

    #[test]
    fn changed_files_between_identical_commits_is_a_real_empty_result() {
        let fixture = TempRepo::new();
        let commit = fixture.commit_file("a.txt", "a\n", "initial");
        let repo = fixture.repo();

        let got = repo.changed_files(&commit, &commit).expect("changed files");

        assert!(got.is_empty());
    }

    // ------------------------------------------------------- detached worktree

    #[test]
    fn add_detached_worktree_produces_the_tree_as_of_that_commit() {
        let fixture = TempRepo::new();
        let first = fixture.commit_file("file.txt", "version one\n", "v1");
        fixture.commit_file("file.txt", "version two\n", "v2");
        let repo = fixture.repo();
        let dest = TempTree::new("dest");
        let wt_path = dest.join("wt");

        let worktree = repo.add_detached_worktree(&wt_path, &first).expect("add worktree");

        let got = fs::read_to_string(worktree.path().join("file.txt")).expect("read checked-out file");
        assert_eq!(got, "version one\n");
    }

    #[test]
    fn a_stale_leftover_worktree_directory_does_not_break_a_new_one() {
        let fixture = TempRepo::new();
        let commit = fixture.commit_file("file.txt", "hello\n", "initial");
        let repo = fixture.repo();
        let dest = TempTree::new("stale");
        let wt_path = dest.join("wt");

        let first = repo.add_detached_worktree(&wt_path, &commit).expect("first add");
        // Simulate a run that was killed before it could clean up: skip
        // `Drop`, leaving both the registered worktree and its directory
        // behind at the same path a fresh run will reuse.
        std::mem::forget(first);
        assert!(wt_path.join("file.txt").exists());

        let second = repo.add_detached_worktree(&wt_path, &commit).expect("second add over the leftover");

        let got = fs::read_to_string(second.path().join("file.txt")).expect("read checked-out file");
        assert_eq!(got, "hello\n");
    }

    // ----------------------------------------------------------- worktree drop

    #[test]
    fn worktree_remove_deletes_it_and_the_directory_is_gone() {
        let fixture = TempRepo::new();
        let commit = fixture.commit_file("file.txt", "hello\n", "initial");
        let repo = fixture.repo();
        let dest = TempTree::new("remove");
        let wt_path = dest.join("wt");
        let worktree = repo.add_detached_worktree(&wt_path, &commit).expect("add worktree");

        worktree.remove().expect("remove worktree");

        assert!(!wt_path.exists());
    }

    #[test]
    fn worktree_remove_surfaces_a_failure_instead_of_swallowing_it() {
        let fixture = TempRepo::new();
        let commit = fixture.commit_file("file.txt", "hello\n", "initial");
        let repo = fixture.repo();
        let dest = TempTree::new("remove-fail");
        let wt_path = dest.join("wt");
        let worktree = repo.add_detached_worktree(&wt_path, &commit).expect("add worktree");

        // Corrupt git's own bookkeeping for this worktree from underneath
        // it, so the later `remove()` call fails for a real reason instead
        // of a fabricated one: git can no longer find the administrative
        // files it needs to tear the worktree down cleanly.
        let admin_root = fixture.path().join(".git/worktrees");
        let admin_entries: Vec<PathBuf> = fs::read_dir(&admin_root)
            .expect("list worktree admin dir")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .collect();
        assert_eq!(admin_entries.len(), 1);
        fs::remove_dir_all(&admin_entries[0]).expect("corrupt worktree admin state");

        let err = worktree.remove().unwrap_err();

        assert!(err.to_string().contains("worktree remove"));
    }

    #[test]
    fn dropping_a_worktree_without_calling_remove_cleans_up_on_a_best_effort_basis() {
        let fixture = TempRepo::new();
        let commit = fixture.commit_file("file.txt", "hello\n", "initial");
        let repo = fixture.repo();
        let dest = TempTree::new("auto-drop");
        let wt_path = dest.join("wt");

        {
            let worktree = repo.add_detached_worktree(&wt_path, &commit).expect("add worktree");
            assert!(worktree.path().exists());
        }

        assert!(!wt_path.exists());
    }

    // ---------------------------------------------------------------- resolve

    #[test]
    fn resolve_of_a_branch_name_and_head_agree_on_the_same_commit() {
        let fixture = TempRepo::new();
        let commit = fixture.commit_file("a.txt", "a\n", "initial");
        let repo = fixture.repo();

        let by_branch = repo.resolve("main").expect("resolve main");
        let by_head = repo.resolve("HEAD").expect("resolve HEAD");

        assert_eq!(by_branch, commit);
        assert_eq!(by_head, commit);
    }

    // ------------------------------------------------------------------ errors

    #[test]
    fn git_error_display_and_source() {
        let not_available = GitError::NotAvailable(std::io::Error::other("no such program"));
        let not_a_repository = GitError::NotARepository(PathBuf::from("/nowhere"));
        let command_failed_exit = GitError::CommandFailed {
            args: vec!["merge-base".to_string(), "a".to_string(), "b".to_string()],
            status: Some(128),
            stderr: "fatal: not a valid object name a\n".to_string(),
        };
        let command_failed_signal = GitError::CommandFailed {
            args: vec!["diff".to_string()],
            status: None,
            stderr: "killed".to_string(),
        };

        let displays: Vec<String> = vec![
            not_available.to_string(),
            not_a_repository.to_string(),
            command_failed_exit.to_string(),
            command_failed_signal.to_string(),
        ];
        assert_eq!(
            displays,
            vec![
                "could not run `git`: no such program".to_string(),
                "no git repository at or above /nowhere".to_string(),
                "git merge-base a b failed (exit status 128): fatal: not a valid object name a".to_string(),
                "git diff failed (terminated by signal): killed".to_string(),
            ]
        );

        let sources: Vec<bool> = vec![
            std::error::Error::source(&not_available).is_some(),
            std::error::Error::source(&not_a_repository).is_some(),
            std::error::Error::source(&command_failed_exit).is_some(),
        ];
        assert_eq!(sources, vec![true, false, false]);

        assert!(format!("{not_available:?}").contains("NotAvailable"));
        assert!(format!("{:?}", Repo { root: PathBuf::from("/r") }.clone()).contains("/r"));
    }
}
