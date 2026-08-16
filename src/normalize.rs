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

use tree_sitter::Node;

use crate::lang::{Language, Role, IDENTIFIER_PLACEHOLDER};

/// Token emitted when a structural node's children are exhausted.
///
/// Chosen so no grammar can produce it as a node kind: `#` opens a comment in
/// most languages, and no operator table contains it. If it ever collided with
/// a real kind, distinct trees could serialize identically — see the module
/// docs for why that matters.
pub const NODE_END: &str = "#END";

/// Linearize a syntax subtree into normalized token names, in preorder.
///
/// Identifier and literal nodes are leaves: they emit one token and are not
/// descended into, so that a string's quotes and contents cannot leak its value
/// into the stream.
pub fn normalize(node: Node<'_>, language: &Language) -> Vec<String> {
    let mut tokens = Vec::new();
    push_tokens(node, language, &mut tokens);
    tokens
}

fn push_tokens(node: Node<'_>, language: &Language, tokens: &mut Vec<String>) {
    match language.role(node.kind(), node.is_named()) {
        Role::Ignored => {}
        Role::Identifier => tokens.push(IDENTIFIER_PLACEHOLDER.to_string()),
        Role::Literal(tag) => tokens.push(tag.as_token().to_string()),
        Role::Structural => {
            tokens.push(node.kind().to_string());
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                push_tokens(child, language, tokens);
            }
            tokens.push(NODE_END.to_string());
        }
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
}
