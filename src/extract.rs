//! Pulling comparable units — functions and methods — out of a source file.
//!
//! A *unit* is the granularity at which duplication is reported: one function,
//! with its normalized token stream, its location, and its size. Everything
//! downstream compares units, never files.
//!
//! # Nested units are extracted too, and also stay inside their parent
//!
//! A method inside a class is a unit; so is a closure inside that method. The
//! enclosing unit's stream still contains the nested one. That is deliberate:
//! two functions that both wrap the same helper are similar *because* of it,
//! and hiding the nested code from the parent's stream would lose that.
//!
//! # Syntax errors are reported, never swallowed
//!
//! tree-sitter is error-tolerant: given nonsense it returns a tree containing
//! `ERROR` nodes rather than refusing. That is useful — a file with one bad
//! line still yields its other functions — but it is also how a scan can
//! quietly degrade to finding nothing. So [`Extraction::had_syntax_errors`]
//! carries the fact upward, and the scanner counts it.

use std::path::{Path, PathBuf};

use tree_sitter::{Node, Parser};

use crate::lang::Language;
use crate::normalize::{count_nodes, normalize};
use crate::token::{Interner, TokenStream};

/// Separator between qualified-name segments, in every language.
///
/// Uniform on purpose: qualified names are for a human reading a report, and a
/// reader scanning findings across a polyglot repository should not have to
/// switch between `::`, `.`, and `#` to parse them.
pub const QUALNAME_SEPARATOR: &str = ".";

/// Name given to a unit whose grammar node carries no name — a JavaScript
/// arrow function, say. The line number keeps it identifiable in a report.
fn anonymous_name(line: usize) -> String {
    format!("<anonymous@{line}>")
}

/// A file that has been read, together with the language claiming it.
///
/// Every detector works from this rather than reading files itself, so a scan
/// touches the disk exactly once no matter how many detectors run over it.
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// Path as the walker reported it.
    pub path: PathBuf,
    /// The language whose grammar parses this file.
    pub language: &'static Language,
    /// Full contents of the file.
    pub text: String,
}

/// One comparable piece of code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// File the unit was found in, exactly as the walker reported it.
    pub path: PathBuf,
    /// Dotted path to the unit within the file, e.g. `Account.grow`.
    pub qualname: String,
    /// First line of the unit, 1-based.
    pub start_line: usize,
    /// Last line of the unit, 1-based and inclusive.
    pub end_line: usize,
    /// Named syntax nodes in the unit, the measure `min_nodes` applies to.
    pub node_count: usize,
    /// The normalized token stream and its content hash.
    pub stream: TokenStream,
}

/// What extracting one file produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extraction {
    /// Units at or above the size threshold, in source order.
    pub units: Vec<Unit>,
    /// Whether the grammar reported a syntax error anywhere in the file.
    ///
    /// Not an error in itself — the units that did parse are still returned —
    /// but a scan over a tree where this is often true is a scan that is not
    /// seeing the code, and the user needs to be told.
    pub had_syntax_errors: bool,
}

/// A parser bound to one language, reusable across files.
///
/// Creating a tree-sitter parser and installing a grammar is not free; a scan
/// creates one extractor per language and feeds every file of that language
/// through it.
pub struct Extractor {
    parser: Parser,
    language: &'static Language,
}

impl Extractor {
    /// Build an extractor for a registered language.
    ///
    /// # Panics
    /// If the grammar cannot be installed, which would mean a tree-sitter ABI
    /// mismatch. Every registered language is checked against its grammar by
    /// `lang::tests::every_registered_language_builds_its_grammar_and_declares_kinds_it_really_has`,
    /// so reaching this panic means the registry and the linked grammars have
    /// diverged — a build-level fault, not a runtime condition to recover from.
    pub fn new(language: &'static Language) -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&language.grammar())
            .expect("registered grammars are ABI-compatible; see lang::tests");
        Extractor { parser, language }
    }

    /// The language this extractor parses.
    pub fn language(&self) -> &'static Language {
        self.language
    }

    /// Extract every unit of at least `min_nodes` named syntax nodes.
    ///
    /// # Panics
    /// If the parser returns no tree, which happens only when a parse is
    /// cancelled or times out. Neither is configured here.
    pub fn extract(
        &mut self,
        source: &str,
        path: &Path,
        min_nodes: usize,
        interner: &mut Interner,
    ) -> Extraction {
        let tree = self.parser.parse(source, None).expect("no timeout or cancellation flag is set");
        let root = tree.root_node();

        let mut units = Vec::new();
        self.visit(root, source, path, "", min_nodes, interner, &mut units);
        units.sort_by_key(|u| (u.start_line, u.end_line));

        Extraction { units, had_syntax_errors: root.has_error() }
    }

    #[allow(clippy::too_many_arguments)]
    fn visit(
        &self,
        node: Node<'_>,
        source: &str,
        path: &Path,
        prefix: &str,
        min_nodes: usize,
        interner: &mut Interner,
        units: &mut Vec<Unit>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let kind = child.kind();
            let start_line = child.start_position().row + 1;

            let declared_name = |field: &str| {
                child
                    .child_by_field_name(field)
                    .map(|n| source[n.byte_range()].to_string())
                    .unwrap_or_else(|| anonymous_name(start_line))
            };

            if let Some(field) = self.language.unit_name_field(kind) {
                let qualname = join(prefix, &declared_name(field));
                let node_count = count_nodes(child);
                if node_count >= min_nodes {
                    units.push(Unit {
                        path: path.to_path_buf(),
                        qualname: qualname.clone(),
                        start_line,
                        end_line: child.end_position().row + 1,
                        node_count,
                        stream: TokenStream::intern(&normalize(child, self.language), interner),
                    });
                }
            }

            // A unit is normally also a scope, so a nested function reads as
            // `outer.inner`. Recursion is unconditional, never gated on the
            // size filter -- though as it happens the filter could not hide a
            // nested unit anyway, since an enclosing unit always contains at
            // least as many nodes as anything inside it. Pinned by
            // `an_enclosing_unit_never_counts_fewer_nodes_than_one_nested_in_it`.
            let nested_prefix = match self.language.scope_name_field(kind) {
                Some(field) => join(prefix, &declared_name(field)),
                None => prefix.to_string(),
            };
            self.visit(child, source, path, &nested_prefix, min_nodes, interner, units);
        }
    }
}

impl std::fmt::Debug for Extractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extractor").field("language", &self.language.name).finish_non_exhaustive()
    }
}

fn join(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_string()
    } else {
        format!("{prefix}{QUALNAME_SEPARATOR}{segment}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang;
    use crate::testutil::python_units as units_of;

    fn python() -> Extractor {
        Extractor::new(lang::by_name("python").expect("python is registered"))
    }

    fn javascript() -> Extractor {
        Extractor::new(lang::by_name("javascript").expect("javascript is registered"))
    }

    /// Qualified names a language's extractor finds in a source string.
    fn js_qualnames(source: &str) -> Vec<String> {
        let mut interner = Interner::new();
        javascript()
            .extract(source, Path::new("sample.js"), 1, &mut interner)
            .units
            .into_iter()
            .map(|u| u.qualname)
            .collect()
    }

    /// Extract from Python source with a permissive size threshold.
    fn qualnames(source: &str) -> Vec<String> {
        units_of(source).into_iter().map(|u| u.qualname).collect()
    }

    // ------------------------------------------------------------- extraction

    #[test]
    fn a_module_level_function_is_extracted_with_its_name() {
        assert_eq!(qualnames("def total(a):\n    return a\n"), vec!["total"]);
    }

    #[test]
    fn a_file_with_no_functions_yields_no_units() {
        // A legitimate zero: the file parsed fine and simply has no functions.
        assert!(units_of("x = 1\ny = 2\n").is_empty());
    }

    #[test]
    fn a_method_is_qualified_by_its_class() {
        assert_eq!(
            qualnames("class Account:\n    def grow(self):\n        return 1\n"),
            vec!["Account.grow"]
        );
    }

    #[test]
    fn a_nested_function_is_qualified_by_its_enclosing_function() {
        let source = "def outer():\n    def inner():\n        return 1\n    return inner\n";
        assert_eq!(qualnames(source), vec!["outer", "outer.inner"]);
    }

    #[test]
    fn deeply_nested_scopes_accumulate_every_segment() {
        let source =
            "class A:\n    class B:\n        def f(self):\n            def g():\n                return 1\n";
        assert_eq!(qualnames(source), vec!["A.B.f", "A.B.f.g"]);
    }

    #[test]
    fn units_are_returned_in_source_order() {
        let source = "def b():\n    return 1\n\ndef a():\n    return 2\n";
        assert_eq!(qualnames(source), vec!["b", "a"]);
    }

    #[test]
    fn a_unit_records_its_one_based_inclusive_line_span() {
        // Line 1 is blank, so the function occupies lines 2 through 3.
        let units = units_of("\ndef f():\n    return 1\n");
        assert_eq!(units.iter().map(|u| (u.start_line, u.end_line)).collect::<Vec<_>>(), vec![(2, 3)]);
    }

    #[test]
    fn a_unit_records_the_file_it_came_from() {
        let mut interner = Interner::new();
        let extraction =
            python().extract("def f():\n    return 1\n", Path::new("a/b/c.py"), 1, &mut interner);
        assert_eq!(extraction.units[0].path, PathBuf::from("a/b/c.py"));
    }

    // ------------------------------------------------------------- size filter

    #[test]
    fn units_below_the_size_threshold_are_not_reported() {
        let mut interner = Interner::new();
        let source = "def tiny():\n    pass\n";
        let permissive = python().extract(source, Path::new("s.py"), 1, &mut interner).units.len();
        let strict = python().extract(source, Path::new("s.py"), 1000, &mut interner).units.len();
        assert_eq!((permissive, strict), (1, 0));
    }

    #[test]
    fn an_enclosing_unit_never_counts_fewer_nodes_than_one_nested_in_it() {
        // This is what makes the size filter safe: raising `min_nodes` can
        // never drop a wrapper while keeping the function inside it, so no
        // finding can hide behind a thin outer function. It holds because a
        // nested unit is a subtree of its parent -- assert it rather than
        // assume it, since the filter's correctness rests on it.
        let source = "def wrapper():\n    def real(a, b, c):\n        total = a + b * c\n        if total > a:\n            total = total - c\n        return total\n";
        let units = units_of(source);
        let by_name: Vec<(&str, usize)> = units.iter().map(|u| (u.qualname.as_str(), u.node_count)).collect();
        assert_eq!(by_name.len(), 2);
        assert_eq!(by_name[0].0, "wrapper");
        assert_eq!(by_name[1].0, "wrapper.real");
        assert!(by_name[0].1 >= by_name[1].1);
    }

    #[test]
    fn raising_the_threshold_drops_units_from_the_smallest_upward() {
        let source = "def small():\n    return 1\n\ndef larger(a, b, c):\n    total = a + b * c\n    if total > a:\n        total = total - c\n    return total\n";
        let mut interner = Interner::new();
        let kept: Vec<Vec<String>> = [1usize, 12, 1000]
            .iter()
            .map(|min| {
                python()
                    .extract(source, Path::new("s.py"), *min, &mut interner)
                    .units
                    .into_iter()
                    .map(|u| u.qualname)
                    .collect()
            })
            .collect();
        assert_eq!(
            kept,
            vec![
                vec!["small".to_string(), "larger".to_string()],
                vec!["larger".to_string()],
                Vec::<String>::new(),
            ]
        );
    }

    #[test]
    fn node_count_reflects_the_size_of_the_unit() {
        let units = units_of("def f():\n    return 1\n\ndef g(a, b):\n    return a + b * 2\n");
        let counts: Vec<bool> = units.windows(2).map(|w| w[0].node_count < w[1].node_count).collect();
        assert_eq!(counts, vec![true]);
    }

    // ------------------------------------------------------------- normalizing

    #[test]
    fn two_functions_differing_only_in_names_share_a_content_hash() {
        // The property the delta engine rests on: identity follows content, not
        // naming or location.
        let units = units_of(
            "def total(rate, years):\n    return rate * years\n\ndef compute(x, n):\n    return x * n\n",
        );
        assert_eq!(units[0].stream.hash(), units[1].stream.hash());
    }

    #[test]
    fn two_functions_with_different_logic_do_not_share_a_content_hash() {
        let units = units_of("def f(a, b):\n    return a + b\n\ndef g(a, b):\n    return a - b\n");
        assert_ne!(units[0].stream.hash(), units[1].stream.hash());
    }

    #[test]
    fn a_content_hash_does_not_depend_on_the_file_or_the_name() {
        let mut one = Interner::new();
        let mut two = Interner::new();
        let body = "def f(a):\n    return a * 2\n";
        let renamed = "def entirely_different(z):\n    return z * 9\n";
        let a = python().extract(body, Path::new("x/one.py"), 1, &mut one);
        let b = python().extract(renamed, Path::new("y/two.py"), 1, &mut two);
        assert_eq!(a.units[0].stream.hash(), b.units[0].stream.hash());
    }

    // ---------------------------------------------------------- syntax errors

    #[test]
    fn a_clean_file_reports_no_syntax_errors() {
        let mut interner = Interner::new();
        let extraction = python().extract("def f():\n    return 1\n", Path::new("s.py"), 1, &mut interner);
        assert!(!extraction.had_syntax_errors);
    }

    #[test]
    fn a_broken_file_reports_syntax_errors_and_still_yields_what_parsed() {
        // Silence here is the dangerous outcome: a scan that parses nothing and
        // reports "no duplication" looks exactly like a clean tree.
        let mut interner = Interner::new();
        let source = "def good(a):\n    return a\n\ndef !!! broken(\n";
        let extraction = python().extract(source, Path::new("s.py"), 1, &mut interner);
        assert!(extraction.had_syntax_errors);
        assert!(extraction.units.iter().any(|u| u.qualname == "good"));
    }

    // ------------------------------------------------------------- qualnames

    // --------------------------------------------------- more than one language

    #[test]
    fn a_javascript_method_is_qualified_by_its_class() {
        let source = "class Account {\n  grow(years) { return this.balance * years; }\n}\n";
        assert_eq!(js_qualnames(source), vec!["Account.grow"]);
    }

    #[test]
    fn an_arrow_function_has_no_name_of_its_own_and_is_identified_by_its_line() {
        // No grammar field can name an arrow function. Rather than drop the
        // unit -- which would make a whole idiom of modern JavaScript invisible
        // to the detector -- it is reported at its line, under the variable it
        // was assigned to.
        let source = "const add = (x, y) => {\n  return x + y;\n};\n";
        assert_eq!(js_qualnames(source), vec!["add.<anonymous@1>"]);
    }

    #[test]
    fn an_anonymous_function_expression_is_also_reported() {
        let source = "const f = function (q) {\n  return q;\n};\n";
        assert_eq!(js_qualnames(source), vec!["f.<anonymous@1>"]);
    }

    #[test]
    fn a_named_function_expression_uses_its_own_name() {
        let source = "const f = function inner(q) {\n  return q;\n};\n";
        assert_eq!(js_qualnames(source), vec!["f.inner"]);
    }

    #[test]
    fn a_renamed_copy_is_detected_across_a_second_language_too() {
        // The blind rename is a property of normalization, not of one grammar.
        let mut interner = Interner::new();
        let a = javascript().extract(
            "function total(rate, years) { return rate * years; }\n",
            Path::new("a.js"),
            1,
            &mut interner,
        );
        let b = javascript().extract(
            "function compute(x, n) { return x * n; }\n",
            Path::new("b.js"),
            1,
            &mut interner,
        );
        assert_eq!(a.units[0].stream.hash(), b.units[0].stream.hash());
    }

    #[test]
    fn the_same_logic_in_two_languages_does_not_share_a_hash() {
        // Node kinds are grammar-specific, so cross-language identity is not
        // claimed. Asserted so nobody later reads equality into it.
        let mut interner = Interner::new();
        let py = python().extract("def f(a, b):\n    return a + b\n", Path::new("x.py"), 1, &mut interner);
        let js =
            javascript().extract("function f(a, b) { return a + b; }\n", Path::new("x.js"), 1, &mut interner);
        assert_ne!(py.units[0].stream.hash(), js.units[0].stream.hash());
    }

    #[test]
    fn a_source_file_carries_its_path_language_and_text() {
        let file = SourceFile {
            path: PathBuf::from("a.py"),
            language: lang::by_name("python").expect("python is registered"),
            text: "x = 1\n".to_string(),
        };
        let copy = file.clone();
        assert_eq!((copy.path, copy.language.name, copy.text), (file.path, "python", file.text));
    }

    #[test]
    fn join_omits_the_separator_at_the_top_level() {
        assert_eq!(join("", "f"), "f");
        assert_eq!(join("A", "f"), format!("A{QUALNAME_SEPARATOR}f"));
    }

    #[test]
    fn an_unnamed_unit_is_identified_by_its_line() {
        assert_eq!(anonymous_name(42), "<anonymous@42>");
    }

    // ------------------------------------------------------------ plumbing

    #[test]
    fn an_extractor_reports_and_debugs_its_language() {
        let extractor = python();
        assert_eq!(extractor.language().name, "python");
        assert!(format!("{extractor:?}").contains("python"));
    }

    #[test]
    fn units_and_extractions_clone_compare_and_debug() {
        let mut interner = Interner::new();
        let extraction = python().extract("def f():\n    return 1\n", Path::new("s.py"), 1, &mut interner);
        let copy = extraction.clone();
        assert_eq!(extraction, copy);
        assert!(format!("{extraction:?}").contains("Extraction"));
        assert!(format!("{:?}", extraction.units[0]).contains("qualname"));
    }
}
