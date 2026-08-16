//! `.dupdelta.toml` — every tuning knob, in one reviewable file.
//!
//! Thresholds that decide whether a pull request fails CI belong in a file a
//! reviewer can read in a diff, not scattered across CI command-line flags
//! where a change is invisible unless someone happens to open the workflow
//! YAML. This module is the schema for that file, plus the two things that
//! make it trustworthy:
//!
//! - **Unknown keys are rejected.** Every struct is `#[serde(deny_unknown_fields)]`.
//!   Silently ignoring a typo'd key (`min_similarty`) would leave the user
//!   believing they configured a threshold that is still the default — a
//!   config-file version of the zero-fallback bug this whole tool exists to
//!   catch in *other* people's code.
//! - **Absence and zero are different facts.** [`ReportConfig::max_findings`]
//!   is `Option<usize>`, not `usize`, because "the key is missing" (no cap)
//!   and "the key is `0`" (report nothing) are both legitimate and must not
//!   collapse onto each other.
//!
//! [`Config::validate`] rejects values that parse fine as TOML but are
//! meaningless as thresholds (a similarity outside `0.0..=1.0`, a NaN, a
//! minimum-nodes of `0`). [`Config::parse`] always validates, so a
//! syntactically valid but semantically broken file never produces a `Config`
//! for a caller to act on.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// Name of the config file [`Config::discover`] looks for.
pub const CONFIG_FILE_NAME: &str = ".dupdelta.toml";

/// Top-level configuration, loaded from `.dupdelta.toml`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Substring patterns; a path containing any is not scanned.
    pub excludes: Vec<String>,
    /// Function-pair (structural clone) detection thresholds.
    pub function: FunctionConfig,
    /// Module-vocabulary (identifier overlap) detection thresholds.
    pub vocab: VocabConfig,
    /// Output shaping.
    pub report: ReportConfig,
}

/// Thresholds for structural (token-similarity) function-pair detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FunctionConfig {
    /// Minimum similarity for a function pair to be a finding. Default 0.85.
    pub min_similarity: f64,
    /// Ignore units smaller than this many syntax nodes. Default 30.
    pub min_nodes: usize,
}

impl Default for FunctionConfig {
    fn default() -> Self {
        FunctionConfig { min_similarity: 0.85, min_nodes: 30 }
    }
}

/// Thresholds for identifier-vocabulary overlap detection between modules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct VocabConfig {
    /// Minimum identifier-vocabulary overlap for a module pair. Default 0.30.
    pub min_overlap: f64,
    /// Ignore modules with fewer distinct identifiers than this. Default 15.
    pub min_vocabulary: usize,
    /// How much an existing pair's overlap must grow to count as worsened. Default 0.05.
    pub worsened_delta: f64,
    /// Identifiers too common to carry domain signal, keyed by language name.
    pub noise: BTreeMap<String, Vec<String>>,
}

impl Default for VocabConfig {
    fn default() -> Self {
        VocabConfig { min_overlap: 0.30, min_vocabulary: 15, worsened_delta: 0.05, noise: BTreeMap::new() }
    }
}

/// Output shaping for the final report.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReportConfig {
    /// Cap on findings reported. `None` means no cap.
    ///
    /// Deliberately `Option<usize>` rather than `usize` with `0` doubling as
    /// "no cap": that would make the key's *absence* and the user writing
    /// `max_findings = 0` indistinguishable, and `0` is a legitimate request
    /// ("report nothing, just tell me pass/fail").
    pub max_findings: Option<usize>,
}

/// Everything that can go wrong loading or validating a config.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read.
    Io {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The text was not valid config TOML (syntax error or unknown key).
    Parse {
        /// The file the text came from, if it came from a file.
        path: Option<PathBuf>,
        /// The underlying TOML deserialization error.
        source: toml::de::Error,
    },
    /// The TOML parsed, but a value is out of its legal range.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io { path, source } => {
                write!(f, "reading {}: {source}", path.display())
            }
            ConfigError::Parse { path: Some(path), source } => {
                write!(f, "parsing {}: {source}", path.display())
            }
            ConfigError::Parse { path: None, source } => {
                write!(f, "parsing config: {source}")
            }
            ConfigError::Invalid(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io { source, .. } => Some(source),
            ConfigError::Parse { source, .. } => Some(source),
            ConfigError::Invalid(_) => None,
        }
    }
}

/// Checks `value` is finite and within `range`, producing a [`ConfigError::Invalid`]
/// naming `key` and `value` when it is not.
///
/// Shared by every `0.0..=1.0` threshold so the message wording — and the NaN
/// check, which `RangeInclusive::contains` alone would not catch consistently
/// across all callers if each rolled its own — cannot drift between them.
fn check_unit_range(key: &str, value: f64) -> Result<(), ConfigError> {
    if value.is_nan() || !(0.0..=1.0).contains(&value) {
        return Err(ConfigError::Invalid(format!("{key} must be between 0.0 and 1.0, got {value}")));
    }
    Ok(())
}

/// Checks `value` is at least 1, producing a [`ConfigError::Invalid`] naming
/// `key` and `value` when it is not.
fn check_at_least_one(key: &str, value: usize) -> Result<(), ConfigError> {
    if value < 1 {
        return Err(ConfigError::Invalid(format!("{key} must be at least 1, got {value}")));
    }
    Ok(())
}

impl Config {
    /// Parses and validates config text. Never returns an invalid `Config`.
    pub fn parse(text: &str) -> Result<Config, ConfigError> {
        let config: Config =
            toml::from_str(text).map_err(|source| ConfigError::Parse { path: None, source })?;
        config.validate()?;
        Ok(config)
    }

    /// Reads, parses and validates the config file at `path`.
    pub fn load(path: &Path) -> Result<Config, ConfigError> {
        let text = fs::read_to_string(path)
            .map_err(|source| ConfigError::Io { path: path.to_path_buf(), source })?;
        let config: Config = toml::from_str(&text)
            .map_err(|source| ConfigError::Parse { path: Some(path.to_path_buf()), source })?;
        config.validate()?;
        Ok(config)
    }

    /// Walks `start` and its ancestors looking for [`CONFIG_FILE_NAME`].
    ///
    /// Returns the first ancestor (nearest first) that has one, or `None` if
    /// no ancestor does. Does not read or validate the file it finds — that
    /// is [`Config::load`]'s job, so a caller that only wants the path is not
    /// forced to also parse.
    pub fn discover(start: &Path) -> Option<PathBuf> {
        let mut dir = Some(start);
        while let Some(candidate) = dir {
            let file = candidate.join(CONFIG_FILE_NAME);
            if file.is_file() {
                return Some(file);
            }
            dir = candidate.parent();
        }
        None
    }

    /// Rejects values that are outside their legal range.
    ///
    /// Called by [`Config::parse`] (and transitively [`Config::load`]), so a
    /// syntactically valid file with a nonsensical threshold never reaches a
    /// caller as a usable `Config`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        check_unit_range("function.min_similarity", self.function.min_similarity)?;
        check_unit_range("vocab.min_overlap", self.vocab.min_overlap)?;
        check_unit_range("vocab.worsened_delta", self.vocab.worsened_delta)?;
        check_at_least_one("function.min_nodes", self.function.min_nodes)?;
        check_at_least_one("vocab.min_vocabulary", self.vocab.min_vocabulary)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A private, self-cleaning directory for file-based tests.
    ///
    /// Named from the process id plus a per-process counter so concurrent
    /// test threads (and concurrent `cargo test` runs) never collide, and
    /// removed on drop so a panicking test does not leak a directory into
    /// the shared temp root.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("dupdelta-config-test-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            TempDir { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    // ------------------------------------------------------------- defaults

    #[test]
    fn empty_toml_yields_default_config() {
        assert_eq!(Config::parse("").unwrap(), Config::default());
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let config = Config::default();
        assert_eq!(config.excludes, Vec::<String>::new());
        assert_eq!(config.function.min_similarity, 0.85);
        assert_eq!(config.function.min_nodes, 30);
        assert_eq!(config.vocab.min_overlap, 0.30);
        assert_eq!(config.vocab.min_vocabulary, 15);
        assert_eq!(config.vocab.worsened_delta, 0.05);
        assert_eq!(config.vocab.noise, BTreeMap::new());
        assert_eq!(config.report.max_findings, None);
    }

    // ------------------------------------------------------- partial overlay

    #[test]
    fn partial_file_overrides_only_the_keys_it_sets() {
        let config = Config::parse("[function]\nmin_similarity = 0.9\n").unwrap();
        let mut want = Config::default();
        want.function.min_similarity = 0.9;
        assert_eq!(config, want);
    }

    // ------------------------------------------------------ unknown fields

    #[test]
    fn unknown_top_level_key_is_rejected() {
        assert!(Config::parse("bogus = true\n").is_err());
    }

    #[test]
    fn unknown_section_key_is_rejected() {
        // The typo the docs warn about: `min_similarty` instead of
        // `min_similarity`. Must error, not silently keep the default while
        // the user believes they configured it.
        assert!(Config::parse("[function]\nmin_similarty = 0.9\n").is_err());
    }

    // ------------------------------------------------------ max_findings=0

    #[test]
    fn max_findings_absent_means_no_cap() {
        assert_eq!(Config::parse("").unwrap().report.max_findings, None);
    }

    #[test]
    fn max_findings_zero_means_report_nothing() {
        let config = Config::parse("[report]\nmax_findings = 0\n").unwrap();
        assert_eq!(config.report.max_findings, Some(0));
    }

    // ---------------------------------------------------------- round-trip

    #[test]
    fn config_round_trips_through_toml() {
        let mut noise = BTreeMap::new();
        noise.insert("python".to_string(), vec!["self".to_string(), "cls".to_string()]);
        let original = Config {
            excludes: vec!["vendor/".to_string(), "generated/".to_string()],
            function: FunctionConfig { min_similarity: 0.9, min_nodes: 40 },
            vocab: VocabConfig { min_overlap: 0.4, min_vocabulary: 20, worsened_delta: 0.1, noise },
            report: ReportConfig { max_findings: Some(50) },
        };

        let text = toml::to_string(&original).unwrap();
        let round_tripped = Config::parse(&text).unwrap();
        assert_eq!(round_tripped, original);
    }

    #[test]
    fn config_with_no_cap_round_trips_through_toml() {
        let original = Config::default();
        let text = toml::to_string(&original).unwrap();
        let round_tripped = Config::parse(&text).unwrap();
        assert_eq!(round_tripped, original);
    }

    // -------------------------------------------------------------- validate

    #[test]
    fn min_similarity_boundaries() {
        let below = Config::parse("[function]\nmin_similarity = -0.1\n");
        let low_edge = Config::parse("[function]\nmin_similarity = 0.0\n");
        let high_edge = Config::parse("[function]\nmin_similarity = 1.0\n");
        let above = Config::parse("[function]\nmin_similarity = 1.1\n");
        assert_eq!(
            [below.is_err(), low_edge.is_ok(), high_edge.is_ok(), above.is_err()],
            [true, true, true, true]
        );
    }

    #[test]
    fn min_overlap_boundaries() {
        let below = Config::parse("[vocab]\nmin_overlap = -0.1\n");
        let low_edge = Config::parse("[vocab]\nmin_overlap = 0.0\n");
        let high_edge = Config::parse("[vocab]\nmin_overlap = 1.0\n");
        let above = Config::parse("[vocab]\nmin_overlap = 1.1\n");
        assert_eq!(
            [below.is_err(), low_edge.is_ok(), high_edge.is_ok(), above.is_err()],
            [true, true, true, true]
        );
    }

    #[test]
    fn worsened_delta_boundaries() {
        let below = Config::parse("[vocab]\nworsened_delta = -0.1\n");
        let low_edge = Config::parse("[vocab]\nworsened_delta = 0.0\n");
        let high_edge = Config::parse("[vocab]\nworsened_delta = 1.0\n");
        let above = Config::parse("[vocab]\nworsened_delta = 1.1\n");
        assert_eq!(
            [below.is_err(), low_edge.is_ok(), high_edge.is_ok(), above.is_err()],
            [true, true, true, true]
        );
    }

    #[test]
    fn min_nodes_must_be_at_least_one() {
        let zero = Config::parse("[function]\nmin_nodes = 0\n");
        let one = Config::parse("[function]\nmin_nodes = 1\n");
        assert_eq!([zero.is_err(), one.is_ok()], [true, true]);
    }

    #[test]
    fn min_vocabulary_must_be_at_least_one() {
        let zero = Config::parse("[vocab]\nmin_vocabulary = 0\n");
        let one = Config::parse("[vocab]\nmin_vocabulary = 1\n");
        assert_eq!([zero.is_err(), one.is_ok()], [true, true]);
    }

    #[test]
    fn nan_threshold_is_rejected() {
        assert!(Config::parse("[function]\nmin_similarity = nan\n").is_err());
    }

    #[test]
    fn invalid_error_names_the_key_and_the_value() {
        let err = Config::parse("[function]\nmin_similarity = 1.5\n").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("function.min_similarity") && message.contains("1.5"));
    }

    // ------------------------------------------------------------ ConfigError

    #[test]
    fn config_error_is_debuggable() {
        let err = Config::parse("bogus = true\n").unwrap_err();
        assert!(format!("{err:?}").contains("Parse"));
    }

    #[test]
    fn parse_error_display_includes_the_underlying_message() {
        let err = Config::parse("not valid toml =====").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("parsing config"));
    }

    #[test]
    fn invalid_error_display_is_the_bare_message() {
        let err = ConfigError::Invalid("custom message".to_string());
        assert_eq!(err.to_string(), "custom message");
    }

    #[test]
    fn io_error_display_names_the_path() {
        let dir = TempDir::new();
        let missing = dir.path().join("nowhere.toml");
        let err = Config::load(&missing).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("nowhere.toml"));
    }

    #[test]
    fn config_error_source_is_present_for_io_and_parse_and_absent_for_invalid() {
        let dir = TempDir::new();
        let missing = dir.path().join("nowhere.toml");
        let io_err = Config::load(&missing).unwrap_err();
        let parse_err = Config::parse("bogus = true\n").unwrap_err();
        let invalid_err = ConfigError::Invalid("x".to_string());
        assert_eq!(
            [
                std::error::Error::source(&io_err).is_some(),
                std::error::Error::source(&parse_err).is_some(),
                std::error::Error::source(&invalid_err).is_some(),
            ],
            [true, true, false]
        );
    }

    // -------------------------------------------------------------------- load

    #[test]
    fn load_reads_and_validates_a_file_on_disk() {
        let dir = TempDir::new();
        let path = dir.path().join(CONFIG_FILE_NAME);
        fs::write(&path, "[report]\nmax_findings = 5\n").unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.report.max_findings, Some(5));
    }

    #[test]
    fn load_of_missing_path_is_an_io_error_naming_the_path() {
        let dir = TempDir::new();
        let missing = dir.path().join(CONFIG_FILE_NAME);
        let err = Config::load(&missing).unwrap_err();
        let matches_path = matches!(&err, ConfigError::Io { path, .. } if path == &missing);
        assert!(matches_path);
    }

    #[test]
    fn load_rejects_an_invalid_file() {
        let dir = TempDir::new();
        let path = dir.path().join(CONFIG_FILE_NAME);
        fs::write(&path, "[function]\nmin_similarity = 2.0\n").unwrap();
        assert!(Config::load(&path).is_err());
    }

    #[test]
    fn load_of_a_syntactically_malformed_file_names_the_path_in_the_message() {
        // Distinct from `load_rejects_an_invalid_file`: that file is valid
        // TOML with an out-of-range value (`ConfigError::Invalid`), this one
        // is not valid TOML at all, which is the only way to reach
        // `ConfigError::Parse { path: Some(_), .. }` through `Config::load`.
        let dir = TempDir::new();
        let path = dir.path().join(CONFIG_FILE_NAME);
        fs::write(&path, "not valid toml =====").unwrap();
        let err = Config::load(&path).unwrap_err();
        let matches_path = matches!(&err, ConfigError::Parse { path: Some(p), .. } if p == &path);
        assert!(matches_path);
        let message = err.to_string();
        assert!(message.starts_with("parsing ") && message.contains(CONFIG_FILE_NAME));
    }

    // ---------------------------------------------------------------- discover

    #[test]
    fn discover_finds_the_file_in_a_parent_directory() {
        // Built entirely inside a fresh temp tree, so a stray
        // `.dupdelta.toml` somewhere above the OS temp root (e.g. a real repo
        // checkout the test happens to run under) cannot make this pass for
        // the wrong reason: the file this test finds is one it wrote itself,
        // at a path under `dir` that only this test's ancestors chain reaches.
        let dir = TempDir::new();
        fs::write(dir.path().join(CONFIG_FILE_NAME), "").unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        fs::create_dir_all(&nested).unwrap();

        let found = Config::discover(&nested).unwrap();
        assert_eq!(found, dir.path().join(CONFIG_FILE_NAME));
    }

    #[test]
    fn discover_returns_none_when_no_ancestor_has_one() {
        // `discover` walks real filesystem ancestors, which this test cannot
        // sandbox: it can only guarantee no `.dupdelta.toml` inside the tree
        // it builds, not above the OS temp root. So rather than assume the
        // environment is clean and assert `None` on that unverified premise,
        // walk the identical ancestor chain first and check for a stray file
        // at every level; a panic there means the test's precondition does
        // not hold on this machine, which is the loud failure this project
        // prefers over a silently vacuous pass. This walk does not call
        // `Config::discover` or assume its return value, so it cannot make
        // the real assertion below pass for the wrong reason -- it only
        // establishes that the fixture is what the test believes it is.
        let dir = TempDir::new();
        let nested = dir.path().join("x").join("y");
        fs::create_dir_all(&nested).unwrap();

        let mut probe = Some(nested.as_path());
        while let Some(candidate) = probe {
            assert!(!candidate.join(CONFIG_FILE_NAME).is_file());
            probe = candidate.parent();
        }

        assert_eq!(Config::discover(&nested), None);
    }
}
