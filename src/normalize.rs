//! Turning syntax into a normalized token stream.
//!
//! Normalization is what makes a *renamed* copy of a function still look like a
//! copy. It performs two abstractions, both deliberately aggressive:
//!
//! - **Blind rename.** Every identifier becomes one placeholder. `interest`,
//!   `rate`, and `x` are all `ID`. Two functions that differ only in naming
//!   normalize to identical streams.
//! - **Literal abstraction.** Every literal becomes a type tag. `0.05` and
//!   `0.055` are both `NUM`. This is the point: the same rule copied with a
//!   tweaked constant is exactly the duplication worth finding, and a detector
//!   that treated the constants as a difference would miss it.
//!
//! Structure — node kinds, nesting, operators, arity — survives untouched.
//!
//! # Why every structural node gets a closing token
//!
//! A bare preorder walk is ambiguous: `X(Y, Z)` and `X(Y(Z))` both linearize to
//! `X Y Z`. Two structurally different functions would then produce identical
//! streams and be reported as a perfect clone — a false positive with nothing
//! in the output to reveal it was one. Emitting [`NODE_END`] when a structural
//! node closes makes the encoding a balanced-parenthesis serialization, which
//! is unique per tree. It costs one extra token per structural node.
//!
//! # This module owns the descent
//!
//! [`placed_tokens`] is the only function in this crate that walks a
//! tree-sitter tree by calling [`Language::role`] on each node and
//! recursing into its children. [`normalize`] and [`identifiers`] are both
//! views over that one walk, not separate walks with their own copy of the
//! same rules — a rule change here (say, how a literal's children are
//! skipped) used to require updating three independent implementations in
//! lockstep, and nothing failed when someone changed one and missed the
//! others. [`crate::blocks`] and [`crate::vocab`] call into this module
//! rather than walking trees themselves.

use std::collections::BTreeSet;

use tree_sitter::Node;

use crate::lang::{Language, Role, IDENTIFIER_PLACEHOLDER};

/// Token emitted when a structural node's children are exhausted.
///
/// Chosen so no grammar can produce it as a node kind: `#` opens a comment in
/// most languages, and no operator table contains it. If it ever collided with
/// a real kind, distinct trees could serialize identically — see the module
/// docs for why that matters.
pub const NODE_END: &str = "#END";

/// A normalized token plus the 1-based source line it came from.
///
/// The line is what turns a flat token match back into a location a human can
/// jump to. An identifier or literal is placed at its own start line; a
/// structural node at *its* start line; its closing [`NODE_END`] at the line
/// the node's last child actually ends on, so a block that closes deep inside
/// a multi-line construct reports the line it truly closes on rather than the
/// line the construct opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedToken {
    /// The normalized token name — see the module docs.
    pub name: String,
    /// 1-based source line the token originated from.
    pub line: usize,
}

/// Linearize a syntax subtree into normalized token names, in preorder.
///
/// Identifier and literal nodes are leaves: they emit one token and are not
/// descended into, so that a string's quotes and contents cannot leak its value
/// into the stream.
///
/// This is [`placed_tokens`] with the line numbers dropped, rather than a
/// second, independent walk — this crate found that keeping "the same walk,
/// twice" as two separate functions is exactly how a rule change (say, how a
/// literal's children are skipped) stops being applied uniformly: the two
/// walks silently disagree, and block findings ([`crate::blocks`]) stop being
/// comparable with function-level ones. There is exactly one traversal in
/// this crate; every derived view is a projection of it. Pinned by
/// `normalize_matches_placed_tokens_with_lines_dropped`.
pub fn normalize(node: Node<'_>, language: &Language) -> Vec<String> {
    placed_tokens(node, language).into_iter().map(|t| t.name).collect()
}

/// Linearize a syntax subtree into normalized tokens, each carrying the
/// 1-based source line it came from.
///
/// The one descent implementation in this crate. [`normalize`] and
/// [`identifiers`] are both derived from it rather than walking the tree
/// again themselves — see [`normalize`]'s docs for why that matters.
pub fn placed_tokens(node: Node<'_>, language: &Language) -> Vec<PlacedToken> {
    let mut tokens = Vec::new();
    push_placed(node, language, &mut tokens);
    tokens
}

fn push_placed(node: Node<'_>, language: &Language, tokens: &mut Vec<PlacedToken>) {
    match language.role(node.kind(), node.is_named()) {
        Role::Ignored => {}
        Role::Identifier => {
            tokens.push(PlacedToken {
                name: IDENTIFIER_PLACEHOLDER.to_string(),
                line: node.start_position().row + 1,
            });
        }
        Role::Literal(tag) => {
            tokens
                .push(PlacedToken { name: tag.as_token().to_string(), line: node.start_position().row + 1 });
        }
        Role::Structural => {
            tokens.push(PlacedToken { name: node.kind().to_string(), line: node.start_position().row + 1 });
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                push_placed(child, language, tokens);
            }
            tokens.push(PlacedToken { name: NODE_END.to_string(), line: node.end_position().row + 1 });
        }
    }
}

/// Every distinct identifier's un-renamed *text* in a subtree.
///
/// Same descent rules as [`placed_tokens`], different payload: where
/// normalization blind-renames every identifier to one placeholder,
/// [`crate::vocab`] needs the opposite — the actual name a human chose —
/// which is why this collects text instead of deriving from
/// [`placed_tokens`]'s already-placeholdered output. The *rules* for what
/// counts as an identifier, a literal, or ignored are still the single ones
/// [`Language::role`] defines; only the payload collected at an identifier
/// differs. Callers filter noise on top of this — see [`crate::vocab`].
pub fn identifiers(node: Node<'_>, language: &Language, source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    push_identifiers(node, language, source, &mut names);
    names
}

fn push_identifiers(node: Node<'_>, language: &Language, source: &str, out: &mut BTreeSet<String>) {
    match language.role(node.kind(), node.is_named()) {
        Role::Identifier => {
            out.insert(source[node.byte_range()].to_string());
        }
        Role::Structural => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                push_identifiers(child, language, source, out);
            }
        }
        Role::Literal(_) | Role::Ignored => {}
    }
}

/// Count the named syntax nodes in a subtree, including the root.
///
/// This is the size measure the `min_nodes` threshold applies to. Named nodes
/// are counted rather than normalized tokens because the number is meant to
/// answer "how much code is this", and should not shift when the normalizer's
/// encoding changes.
pub fn count_nodes(node: Node<'_>) -> usize {
    let mut count = usize::from(node.is_named());
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        count += count_nodes(child);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang;
    use tree_sitter::Parser;

    /// Tokens the normalizer produces for a whole Python file.
    fn tokens_of(source: &str) -> Vec<String> {
        let language = lang::by_name("python").expect("python is registered");
        let mut parser = Parser::new();
        parser.set_language(&language.grammar()).expect("grammar loads");
        let tree = parser.parse(source, None).expect("parser is not cancelled");
        normalize(tree.root_node(), language)
    }

    fn named_node_count(source: &str) -> usize {
        let language = lang::by_name("python").expect("python is registered");
        let mut parser = Parser::new();
        parser.set_language(&language.grammar()).expect("grammar loads");
        let tree = parser.parse(source, None).expect("parser is not cancelled");
        count_nodes(tree.root_node())
    }

    const LITERAL_TAGS: &[&str] = &["NUM", "STR", "BOOL", "NIL", "LIT"];

    // ----------------------------------------------------------- blind rename

    #[test]
    fn functions_differing_only_in_identifiers_normalize_identically() {
        let a = tokens_of("def total(rate, years):\n    return rate * years\n");
        let b = tokens_of("def compute(x, n):\n    return x * n\n");
        assert_eq!(a, b);
    }

    #[test]
    fn every_identifier_collapses_to_one_placeholder() {
        let tokens = tokens_of("alpha = beta\n");
        let placeholders = tokens.iter().filter(|t| t.as_str() == IDENTIFIER_PLACEHOLDER).count();
        assert_eq!(placeholders, 2);
    }

    // ----------------------------------------------------- literal abstraction

    #[test]
    fn numeric_literals_of_different_values_normalize_identically() {
        // The whole point: the same rule copied with a tweaked constant is the
        // duplication worth finding.
        assert_eq!(tokens_of("r = 0.05\n"), tokens_of("r = 0.055\n"));
        assert_eq!(tokens_of("n = 1\n"), tokens_of("n = 999999\n"));
    }

    #[test]
    fn a_literal_is_a_leaf_so_its_contents_cannot_leak() {
        // A Python string is a node with string_start/string_content/string_end
        // children. Descending into it would put the text into the stream.
        let short = tokens_of("s = 'a'\n");
        let long = tokens_of("s = 'a much longer piece of text'\n");
        assert_eq!(short, long);
        assert!(short.contains(&"STR".to_string()));
    }

    #[test]
    fn literals_of_different_types_stay_distinguishable() {
        let sources = ["n = 1", "s = 'x'", "b = True", "v = None"];
        let tags: Vec<Vec<String>> = sources
            .iter()
            .map(|src| tokens_of(src).into_iter().filter(|t| LITERAL_TAGS.contains(&t.as_str())).collect())
            .collect();
        assert_eq!(tags, vec![vec!["NUM"], vec!["STR"], vec!["BOOL"], vec!["NIL"]]);
    }

    // -------------------------------------------------- structure is preserved

    #[test]
    fn different_operators_produce_different_streams() {
        assert_ne!(tokens_of("c = a + b\n"), tokens_of("c = a - b\n"));
    }

    #[test]
    fn different_word_operators_produce_different_streams() {
        assert_ne!(tokens_of("c = a and b\n"), tokens_of("c = a or b\n"));
    }

    #[test]
    fn different_control_flow_produces_different_streams() {
        assert_ne!(tokens_of("if c:\n    f()\n"), tokens_of("while c:\n    f()\n"));
    }

    #[test]
    fn call_arity_is_preserved() {
        assert_ne!(tokens_of("f(a)\n"), tokens_of("f(a, b)\n"));
    }

    #[test]
    fn comments_contribute_nothing() {
        assert_eq!(tokens_of("x = 1\n"), tokens_of("# an explanation\nx = 1  # trailing\n"));
    }

    #[test]
    fn keywords_contribute_nothing_beyond_their_parent_node_kind() {
        // `async def` and `def` differ only by an anonymous keyword; the parent
        // is `function_definition` either way, so the streams match.
        assert_eq!(tokens_of("def f():\n    pass\n"), tokens_of("async def f():\n    pass\n"));
    }

    // ------------------------------------------------------ encoding soundness

    #[test]
    fn nesting_and_sibling_order_are_distinguishable() {
        // The ambiguity a bare preorder walk would have: `X(Y, Z)` versus
        // `X(Y(Z))`. Without the closing token these collide, and two
        // structurally different functions get reported as a perfect clone.
        assert_ne!(tokens_of("f(a, g)\n"), tokens_of("f(a(g))\n"));
    }

    #[test]
    fn every_structural_node_is_closed_exactly_once() {
        // Balanced-parenthesis property: the stream must never close a node it
        // did not open, and must close every node it opened.
        let tokens = tokens_of("def f(a):\n    if a:\n        return [1, 2]\n    return None\n");
        let mut depth: i64 = 0;
        let mut deepest_underflow = 0i64;
        for token in &tokens {
            if token == NODE_END {
                depth -= 1;
            } else if token != IDENTIFIER_PLACEHOLDER && !LITERAL_TAGS.contains(&token.as_str()) {
                depth += 1;
            }
            deepest_underflow = deepest_underflow.min(depth);
        }
        assert_eq!((depth, deepest_underflow), (0, 0));
    }

    #[test]
    fn an_empty_source_normalizes_to_just_the_root() {
        assert_eq!(tokens_of(""), vec!["module".to_string(), NODE_END.to_string()]);
    }

    #[test]
    fn source_that_fails_to_parse_still_yields_a_stream() {
        // tree-sitter is error-tolerant and produces ERROR nodes rather than
        // refusing. Normalization must not panic on them; whether a file had
        // syntax errors is reported separately, by the extractor.
        assert!(!tokens_of("def (:::\n").is_empty());
    }

    // ------------------------------------------------------------- count_nodes

    #[test]
    fn node_count_grows_with_the_amount_of_code() {
        let small = named_node_count("x = 1\n");
        let large = named_node_count("def f(a, b):\n    return a + b * 2\n");
        assert!(large > small);
    }

    #[test]
    fn node_count_of_an_empty_file_counts_only_the_root() {
        assert_eq!(named_node_count(""), 1);
    }

    #[test]
    fn node_count_ignores_anonymous_nodes() {
        // `(a)` adds two anonymous parentheses plus one named
        // parenthesized_expression around the same identifier.
        assert_eq!(named_node_count("(a)\n") - named_node_count("a\n"), 1);
    }

    // ------------------------------------------------- single descent, pinned

    /// The full [`PlacedToken`] stream for a whole source file in the named
    /// language.
    fn placed_tokens_of(language_name: &str, source: &str) -> Vec<PlacedToken> {
        let language = lang::by_name(language_name).expect("language is registered");
        let mut parser = Parser::new();
        parser.set_language(&language.grammar()).expect("grammar loads");
        let tree = parser.parse(source, None).expect("parser is not cancelled");
        placed_tokens(tree.root_node(), language)
    }

    #[test]
    fn normalize_matches_placed_tokens_with_lines_dropped() {
        // This is the property the whole refactor exists to protect. Before
        // this module owned the descent, `normalize::normalize`,
        // `blocks::placed_tokens`, and `vocab::collect_identifiers` were
        // three hand-written walks that had to be kept in lockstep by
        // convention alone -- `blocks.rs`'s own docs admitted "must not
        // diverge" with nothing that would fail if they did. Now
        // `normalize` is *defined* as `placed_tokens` with the lines
        // dropped, so they cannot diverge -- but that guarantee is only as
        // good as this test, which is what would have caught the old
        // three-way drift. Checked in two unrelated grammars so it cannot
        // pass by coincidentally matching one language's node shapes.
        let python = "def total(rate, years):\n    if rate > 0:\n        return rate * years\n    return 0\n";
        let javascript = "function total(rate, years) {\n  if (rate > 0) {\n    return rate * years;\n  }\n  return 0;\n}\n";

        let via_normalize: Vec<Vec<String>> = [("python", python), ("javascript", javascript)]
            .iter()
            .map(|&(l, s)| tokens_of_lang(l, s))
            .collect();
        let via_placed_tokens: Vec<Vec<String>> = [("python", python), ("javascript", javascript)]
            .iter()
            .map(|&(l, s)| placed_tokens_of(l, s).into_iter().map(|t| t.name).collect())
            .collect();

        assert_eq!(via_normalize, via_placed_tokens);
    }

    /// [`normalize`] for an arbitrary language, not just Python -- [`tokens_of`]
    /// stays Python-only because most tests above only care about one grammar.
    fn tokens_of_lang(language_name: &str, source: &str) -> Vec<String> {
        let language = lang::by_name(language_name).expect("language is registered");
        let mut parser = Parser::new();
        parser.set_language(&language.grammar()).expect("grammar loads");
        let tree = parser.parse(source, None).expect("parser is not cancelled");
        normalize(tree.root_node(), language)
    }
}
