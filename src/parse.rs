//! Parsing a file once and sharing that one tree with every detector.
//!
//! # Why this module exists
//!
//! A scan used to parse every file three times: once in [`crate::extract`] for
//! function-level units, once in [`crate::blocks`] for placed tokens, and once
//! in [`crate::vocab`] for identifiers. Each of those built its own
//! [`tree_sitter::Parser`], installed the grammar again, and produced a tree
//! from the same bytes. Three identical syntax trees, thrown away two times
//! over, per file.
//!
//! The three descents already shared their rules — [`crate::normalize`] owns
//! the single walk and every detector calls into it — so nothing about the
//! result depended on which detector asked first. Only the *production* of the
//! tree was duplicated. This module is where that production now happens, once.
//!
//! # One parser per language, not one per file
//!
//! Installing a grammar is not free, so [`Parsers`] keeps one parser per
//! language for the whole scan and reuses it for every file of that language.
//! [`crate::extract::Extractor`] was already documented this way; this is
//! simply where the parser now lives.
//!
//! # What this does not do
//!
//! It does not cache across scans. A `ci` run scans the working tree and then
//! a detached worktree of the merge-base, and re-parsing the base tree is a
//! separate problem with a separate cost; see the tracking issue. Everything
//! here is per-scan, in-process, and stateless between scans.

use std::collections::HashMap;

use tree_sitter::{Parser, Tree};

use crate::extract::SourceFile;
use crate::lang::Language;

// Number of parses performed on the current thread, counted only in tests.
//
// Thread-local, not global: Rust's test harness runs each `#[test]` on its own
// thread, so a count taken around one test's own work cannot be perturbed by
// another test parsing concurrently. That is what lets the regression test
// assert an exact number rather than a relative one.
#[cfg(test)]
thread_local! {
    static PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Parses performed on the current thread since the test thread started.
#[cfg(test)]
pub(crate) fn parse_count() -> usize {
    PARSES.with(|count| count.get())
}

/// One tree-sitter parser per language, created on first use and reused.
///
/// Holding one of these for the duration of a scan is what turns "parse every
/// file" from "build a parser for every file" into "build at most one parser
/// per language in the scan". Keyed by registry name, which is what
/// identifies a language in every map in this crate — `Language` itself is not
/// `Hash`, since a grammar is a function pointer that cannot be compared.
#[derive(Default)]
pub struct Parsers {
    by_language: HashMap<&'static str, Parser>,
}

/// Hand-written because `tree_sitter::Parser` is not `Debug`: a scan holds
/// twelve of these, and "which languages does this scan have a parser for" is
/// the only thing about them worth printing.
impl std::fmt::Debug for Parsers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut languages: Vec<&str> = self.by_language.keys().copied().collect();
        languages.sort_unstable();
        f.debug_struct("Parsers").field("languages", &languages).finish()
    }
}

impl Parsers {
    /// A set with no parsers yet; each language's is built on its first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many parsers have actually been built so far.
    ///
    /// One per language seen, never one per file — which is the property the
    /// per-file extraction path depends on.
    pub fn parser_count(&self) -> usize {
        self.by_language.len()
    }

    /// Parse one file, building its language's parser if this is the first
    /// file of that language in this scan.
    pub fn parse(&mut self, file: &SourceFile) -> Tree {
        self.parse_loose(file.language, &file.text)
    }

    /// Parse every file exactly once, in order.
    ///
    /// The result is index-aligned with `files`: tree `i` is the tree of
    /// `files[i]`, so a caller cannot pair a tree with the wrong file.
    pub fn parse_all(&mut self, files: &[SourceFile]) -> ParsedFiles {
        let trees = files.iter().map(|file| self.parse(file)).collect();
        ParsedFiles { trees }
    }

    /// Parse loose text with the parser for `language`.
    ///
    /// For callers holding a path and a string rather than a [`SourceFile`]
    /// — chiefly [`crate::extract::Extractor::extract`], which predates this
    /// module. A scan never goes through here; it parses files once and hands
    /// the trees to the detectors.
    ///
    /// # Panics
    /// If the grammar cannot be installed, which would mean a tree-sitter ABI
    /// mismatch; or if the parser returns no tree, which happens only when a
    /// parse is cancelled or times out. Neither is configured here. The first
    /// is pinned by
    /// `lang::tests::every_registered_language_builds_its_grammar_and_declares_kinds_it_really_has`,
    /// the second is the same condition [`crate::extract::Extractor::extract`]
    /// has always panicked on.
    pub fn parse_loose(&mut self, language: &'static Language, text: &str) -> Tree {
        let parser = self.by_language.entry(language.name).or_insert_with(|| new_parser(language));

        #[cfg(test)]
        PARSES.with(|count| count.set(count.get() + 1));

        parser.parse(text, None).expect("no timeout or cancellation flag is set")
    }
}

/// Build a parser with `language`'s grammar installed.
fn new_parser(language: &'static Language) -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(&language.grammar())
        .expect("registered grammars are ABI-compatible; see lang::tests");
    parser
}

/// The trees of one scan: exactly one per source file, in file order.
///
/// Detectors take a [`tree_sitter::Node`] from here rather than parsing, which
/// is what makes a file's syntax tree a shared input rather than three private
/// ones.
pub struct ParsedFiles {
    trees: Vec<Tree>,
}

impl ParsedFiles {
    /// Number of files parsed.
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// Whether the scan had no files at all.
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// The trees, in file order, paired by index with the files given to
    /// [`Parsers::parse_all`].
    pub fn iter(&self) -> impl Iterator<Item = &Tree> {
        self.trees.iter()
    }

    /// Each file with the tree that was parsed from it.
    ///
    /// [`Parsers::parse_all`] returns one tree per file, in order, so the two
    /// line up index for index -- but that is a promise about the past, and a
    /// detector that pairs them with `zip` is restating it every time. This is
    /// where the promise is cashed in.
    ///
    /// Cashing it in means checking it, because `zip` stops at the shorter of
    /// the two: handed a `files` slice longer than the one this was built
    /// from, it would drop the trailing files from the scan and say nothing.
    /// A missing file is not a finding, so it would surface as duplication
    /// that was quietly never looked for.
    pub fn each<'a>(&'a self, files: &'a [SourceFile]) -> impl Iterator<Item = (&'a SourceFile, &'a Tree)> {
        assert_eq!(
            files.len(),
            self.trees.len(),
            "ParsedFiles has {} trees but was handed {} files; it can only be paired \
             with the same list it was parsed from",
            self.trees.len(),
            files.len()
        );
        files.iter().zip(self.trees.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::blocks::{find_blocks, find_blocks_parsed, placed_tokens, placed_tokens_of, BlockOptions};
    use crate::extract::Extractor;
    use crate::lang;
    use crate::testutil::{javascript_file, python_file};
    use crate::token::TokenScope;
    use crate::vocab::{find_vocab_pairs, find_vocab_pairs_parsed, vocabulary_of, VocabOptions};

    /// Two files per language, so a per-file parser would build four and a
    /// per-language one builds two.
    fn corpus() -> Vec<SourceFile> {
        vec![
            python_file("a.py", "def alpha(beta):\n    return beta + 1\n"),
            python_file("b.py", "def gamma(delta):\n    return delta + 1\n"),
            javascript_file("c.js", "function zeta() { return 1; }\n"),
            javascript_file("d.js", "function eta() { return 2; }\n"),
        ]
    }

    #[test]
    #[should_panic(expected = "it can only be paired with the same list it was parsed from")]
    fn each_refuses_a_file_list_it_was_not_parsed_from() {
        // `zip` stops at the shorter of the two, so a longer `files` would
        // drop its tail from the scan without a word. The tail is files no
        // detector ever looked at, which surfaces only as duplication that
        // was quietly never searched for -- so this refuses rather than
        // truncating.
        let files = corpus();
        let parsed = Parsers::new().parse_all(&files[..2]);
        let _ = parsed.each(&files);
    }

    fn vocab_options() -> VocabOptions {
        VocabOptions { min_overlap: 0.0, min_vocabulary: 1, noise: BTreeMap::new(), sample_size: 5 }
    }

    #[test]
    fn a_scan_parses_each_file_exactly_once_and_the_detectors_parse_none() {
        // The regression this module exists for. It fails against the old
        // three-parser shape: the same four files parsed once per detector
        // would take this counter to 12, not 4.
        let files = corpus();
        let mut parsers = Parsers::new();
        let trees = parsers.parse_all(&files);
        assert_eq!(parse_count(), files.len(), "one parse per file, and nothing more");

        // Now run all three detectors over those same trees.
        let mut scope = TokenScope::for_a_scan();
        let mut units = 0;
        for (file, tree) in trees.each(&files) {
            let extraction = Extractor::new(file.language).extract_tree(
                &file.text,
                &file.path,
                tree.root_node(),
                1,
                &mut scope,
            );
            units += extraction.units.len();
        }
        let _ = find_blocks_parsed(&files, &trees, &BlockOptions { min_tokens: 5 });
        let _ = find_vocab_pairs_parsed(&files, &trees, &vocab_options());

        assert!(units > 0, "the fixture must actually produce units");
        assert_eq!(parse_count(), files.len(), "a detector re-parsed a file it was handed a tree for");
    }

    #[test]
    fn one_parser_is_built_per_language_and_reused_across_files() {
        let files = corpus();
        let mut parsers = Parsers::new();
        assert_eq!(parsers.parser_count(), 0, "no parser exists before anything is parsed");

        let trees = parsers.parse_all(&files);
        assert_eq!((parsers.parser_count(), trees.len()), (2, files.len()));
    }

    #[test]
    fn the_trees_come_back_in_file_order() {
        let files = corpus();
        let trees = Parsers::new().parse_all(&files);

        let errors: Vec<bool> = trees.iter().map(|tree| tree.root_node().has_error()).collect();
        assert_eq!(errors, vec![false; files.len()]);
    }

    #[test]
    fn a_broken_file_yields_a_tree_that_reports_the_error() {
        // The tree is what carries `had_syntax_errors` up to the scanner, so
        // parsing must not fail on bad input -- it must return a tree with an
        // error in it.
        let files = vec![python_file("broken.py", "def good(a):\n    return a\n\ndef !!! broken(\n")];
        let trees = Parsers::new().parse_all(&files);
        assert!(trees.iter().next().expect("one tree").root_node().has_error());
    }

    #[test]
    fn scanning_nothing_parses_nothing() {
        let trees = Parsers::new().parse_all(&[]);
        assert!(trees.is_empty());
        assert_eq!(trees.len(), 0);
        assert_eq!(trees.iter().count(), 0);
    }

    #[test]
    fn loose_text_is_parsed_with_the_parser_for_its_language() {
        // The path `Extractor::extract` takes, since it holds a path and a
        // string rather than a `SourceFile`.
        let mut parsers = Parsers::new();
        let python = parsers.parse_loose(lang::by_name("python").expect("python is registered"), "x = 1\n");
        assert_eq!(parsers.parser_count(), 1);

        let javascript = parsers
            .parse_loose(lang::by_name("javascript").expect("javascript is registered"), "let x = 1;\n");
        assert_eq!(parsers.parser_count(), 2);
        assert_eq!((python.root_node().kind(), javascript.root_node().kind()), ("module", "program"));
    }

    #[test]
    fn detectors_agree_whether_they_are_handed_a_tree_or_asked_for_one() {
        // The output-identity pin: sharing a tree must not change a single
        // finding, so the tree-taking variants and the file-taking ones that
        // parse for themselves have to agree exactly.
        let files = corpus();
        let trees = Parsers::new().parse_all(&files);

        let blocks = find_blocks_parsed(&files, &trees, &BlockOptions { min_tokens: 5 });
        let vocab = find_vocab_pairs_parsed(&files, &trees, &vocab_options());
        // Not vacuous: an agreement between two empty results would pin
        // nothing at all.
        assert!(!blocks.is_empty() && !vocab.is_empty());

        assert_eq!(blocks, find_blocks(&files, &BlockOptions { min_tokens: 5 }));
        assert_eq!(vocab, find_vocab_pairs(&files, &vocab_options()));

        for (file, tree) in trees.each(&files) {
            assert_eq!(placed_tokens_of(file, tree.root_node()), placed_tokens(file));
            assert_eq!(
                vocabulary_of(file, tree.root_node(), &BTreeMap::new()),
                crate::vocab::vocabulary(file, &BTreeMap::new())
            );
        }
    }

    #[test]
    fn a_parser_set_reports_and_debugs_its_languages() {
        let mut parsers = Parsers::new();
        assert_eq!(format!("{parsers:?}"), "Parsers { languages: [] }");

        parsers.parse(&python_file("a.py", "x = 1\n"));
        assert!(format!("{parsers:?}").contains("python"));
    }
}
