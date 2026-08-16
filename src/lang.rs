//! The language registry: which grammars are supported, and what their syntax
//! nodes mean for normalization.
//!
//! Everything language-specific in `dupdelta` lives here. The detectors above
//! this layer never mention a language; they work on [`crate::token::TokenStream`]s,
//! which is why adding a language is a data change rather than a code change.
//!
//! # What a language has to declare
//!
//! Normalization needs to know, for each syntax node, whether it is an
//! identifier (blind-renamed to one placeholder), a literal (abstracted to a
//! type tag), or structure (kept). Extraction additionally needs to know which
//! nodes are *units* — the functions and methods that get compared — and which
//! nodes contribute a segment to a qualified name.
//!
//! # Why keywords are dropped and operators are kept
//!
//! In a tree-sitter grammar, keywords are *anonymous* nodes: `if` is a child of
//! `if_statement`, `def` a child of `function_definition`. The parent's kind
//! already carries that information, so emitting the keyword too is pure
//! duplication — it inflates every token stream without separating anything.
//!
//! Operators are different. `a + b` and `a - b` are both `binary_operator` with
//! two identifiers; the *only* thing distinguishing them is the anonymous `+`
//! or `-` node. Drop those and a function that adds becomes indistinguishable
//! from one that subtracts. So anonymous nodes are dropped unless they appear
//! in [`OPERATOR_TOKENS`], which includes word-operators (`and`, `not`, `in`,
//! `is`) because several languages spell operators as words.

use std::path::Path;

use tree_sitter::Language as Grammar;

/// Placeholder emitted for every identifier, whatever it was called.
///
/// This is the "blind rename": two functions that differ only in their variable
/// names normalize to the same stream, which is what makes a renamed copy of a
/// function detectable as a copy.
pub const IDENTIFIER_PLACEHOLDER: &str = "ID";

/// Anonymous nodes kept as structure. Everything else anonymous is dropped.
///
/// Word-operators are included: Python spells conjunction `and`, Ruby has `or`
/// and `not`, and losing them would make `a and b` equal to `a or b`.
pub const OPERATOR_TOKENS: &[&str] = &[
    // arithmetic
    "+", "-", "*", "/", "%", "**", "//", //
    // comparison
    "==", "!=", "<", ">", "<=", ">=", "<=>", "===", "!==", //
    // logical
    "&&", "||", "!", "and", "or", "not", //
    // membership and identity
    "in", "is", //
    // bitwise
    "&", "|", "^", "~", "<<", ">>", ">>>", //
    // assignment
    "=", "+=", "-=", "*=", "/=", "%=", "**=", "//=", "&=", "|=", "^=", "<<=", ">>=", "&&=", "||=", //
    // null handling, ranges, spreads
    "??", "??=", "?.", "?", "..", "...", "..=", "=>",
];

/// What a syntax node contributes to a normalized token stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// An identifier. Emitted as [`IDENTIFIER_PLACEHOLDER`]; not descended into.
    Identifier,
    /// A literal. Emitted as its type tag; not descended into, so that the
    /// internals of a string (its quotes, its content) cannot leak the value.
    Literal(LiteralTag),
    /// Structure. Emitted as its own node kind, and descended into.
    Structural,
    /// Contributes nothing and is not descended into.
    Ignored,
}

/// The type of a literal, which is all that survives abstraction.
///
/// The *value* is deliberately discarded: a rate written `0.05` in one place
/// and `0.055` in another is the same logic duplicated, and a detector that
/// treats them as different misses exactly the copies worth finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralTag {
    /// Integer, float, or other numeric literal.
    Number,
    /// String or character literal.
    String,
    /// Boolean literal.
    Boolean,
    /// The language's null/nil/none literal.
    Nil,
    /// A literal that is none of the above.
    Other,
}

impl LiteralTag {
    /// The token emitted in place of the literal.
    pub fn as_token(self) -> &'static str {
        match self {
            LiteralTag::Number => "NUM",
            LiteralTag::String => "STR",
            LiteralTag::Boolean => "BOOL",
            LiteralTag::Nil => "NIL",
            LiteralTag::Other => "LIT",
        }
    }
}

/// A node kind paired with the field holding its name.
///
/// Most grammars name a definition through a `name` field, but not all: a Rust
/// `impl_item` carries its subject in `type`. Declaring the field per kind
/// avoids a special case in the extractor.
pub type NamedKind = (&'static str, &'static str);

/// A supported language: its grammar, plus what its node kinds mean.
pub struct Language {
    /// Registry name, e.g. `"python"`.
    pub name: &'static str,
    /// File extensions, without the dot.
    pub extensions: &'static [&'static str],
    /// The tree-sitter grammar. A function pointer because a grammar cannot be
    /// built in a `const`.
    grammar: fn() -> Grammar,
    /// Node kinds that form a comparable unit, with the field naming each.
    pub unit_kinds: &'static [NamedKind],
    /// Node kinds contributing a segment to a qualified name, with their name
    /// field. Units are usually also scopes, so that a nested function reads as
    /// `outer.inner`.
    pub scope_kinds: &'static [NamedKind],
    /// Node kinds whose text is an identifier.
    pub identifier_kinds: &'static [&'static str],
    /// Node kinds that are literals, and the tag replacing them.
    pub literal_kinds: &'static [(&'static str, LiteralTag)],
    /// Named node kinds that contribute nothing — comments, chiefly.
    pub ignored_kinds: &'static [&'static str],
}

impl Language {
    /// Build the tree-sitter grammar for this language.
    pub fn grammar(&self) -> Grammar {
        (self.grammar)()
    }

    /// Classify a node kind.
    ///
    /// `named` distinguishes a grammar's named nodes from its anonymous ones
    /// (keywords, punctuation, operators).
    pub fn role(&self, kind: &str, named: bool) -> Role {
        if self.ignored_kinds.contains(&kind) {
            return Role::Ignored;
        }
        if self.identifier_kinds.contains(&kind) {
            return Role::Identifier;
        }
        if let Some(&(_, tag)) = self.literal_kinds.iter().find(|(k, _)| *k == kind) {
            return Role::Literal(tag);
        }
        if named || OPERATOR_TOKENS.contains(&kind) {
            Role::Structural
        } else {
            Role::Ignored
        }
    }

    /// The name field for `kind` if it is a unit kind.
    pub fn unit_name_field(&self, kind: &str) -> Option<&'static str> {
        self.unit_kinds.iter().find(|(k, _)| *k == kind).map(|(_, f)| *f)
    }

    /// The name field for `kind` if it contributes a qualified-name segment.
    pub fn scope_name_field(&self, kind: &str) -> Option<&'static str> {
        self.scope_kinds.iter().find(|(k, _)| *k == kind).map(|(_, f)| *f)
    }
}

impl std::fmt::Debug for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Language")
            .field("name", &self.name)
            .field("extensions", &self.extensions)
            .finish_non_exhaustive()
    }
}

/// Python.
static PYTHON: Language = Language {
    name: "python",
    extensions: &["py", "pyi"],
    grammar: || tree_sitter_python::LANGUAGE.into(),
    unit_kinds: &[("function_definition", "name")],
    scope_kinds: &[("class_definition", "name"), ("function_definition", "name")],
    identifier_kinds: &["identifier"],
    literal_kinds: &[
        ("integer", LiteralTag::Number),
        ("float", LiteralTag::Number),
        ("string", LiteralTag::String),
        ("concatenated_string", LiteralTag::String),
        ("true", LiteralTag::Boolean),
        ("false", LiteralTag::Boolean),
        ("none", LiteralTag::Nil),
        ("ellipsis", LiteralTag::Other),
    ],
    ignored_kinds: &["comment"],
};

/// JavaScript.
///
/// Note the two unit kinds with no name of their own: an `arrow_function` and
/// an anonymous `function_expression` are real, comparable units that no
/// grammar field can name. `variable_declarator` is registered as a scope so
/// that `const add = (x, y) => …` reports as `add.<anonymous@…>` rather than
/// stranding the reader with a bare line number.
static JAVASCRIPT: Language = Language {
    name: "javascript",
    extensions: &["js", "mjs", "cjs", "jsx"],
    grammar: || tree_sitter_javascript::LANGUAGE.into(),
    unit_kinds: &[
        ("function_declaration", "name"),
        ("generator_function_declaration", "name"),
        ("function_expression", "name"),
        ("method_definition", "name"),
        ("arrow_function", "name"),
    ],
    scope_kinds: &[
        ("class_declaration", "name"),
        ("variable_declarator", "name"),
        ("function_declaration", "name"),
        ("generator_function_declaration", "name"),
        ("method_definition", "name"),
    ],
    identifier_kinds: &[
        "identifier",
        "property_identifier",
        "private_property_identifier",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
    ],
    literal_kinds: &[
        ("number", LiteralTag::Number),
        ("string", LiteralTag::String),
        ("template_string", LiteralTag::String),
        ("regex", LiteralTag::Other),
        ("true", LiteralTag::Boolean),
        ("false", LiteralTag::Boolean),
        ("null", LiteralTag::Nil),
        ("undefined", LiteralTag::Nil),
    ],
    ignored_kinds: &["comment"],
};

/// Every language the registry knows, ordered by name.
///
/// Adding a language means adding a `static` above and one entry here. No code
/// outside this module changes.
static ALL: &[&Language] = &[&JAVASCRIPT, &PYTHON];

/// Every language the registry knows, ordered by name.
pub fn all() -> &'static [&'static Language] {
    ALL
}

/// Look a language up by registry name.
pub fn by_name(name: &str) -> Option<&'static Language> {
    all().iter().copied().find(|l| l.name == name)
}

/// The language for a path, by file extension.
///
/// Matching is case-insensitive, because `.PY` is still Python.
pub fn for_path(path: &Path) -> Option<&'static Language> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    all().iter().copied().find(|l| l.extensions.contains(&ext.as_str()))
}

/// Whether any registered language claims this path.
pub fn is_supported(path: &Path) -> bool {
    for_path(path).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ------------------------------------------------------------ LiteralTag

    #[test]
    fn every_literal_tag_has_a_distinct_token() {
        let tags =
            [LiteralTag::Number, LiteralTag::String, LiteralTag::Boolean, LiteralTag::Nil, LiteralTag::Other];
        let tokens: Vec<&str> = tags.iter().map(|t| t.as_token()).collect();
        let mut unique = tokens.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(tokens.len(), unique.len());
        assert_eq!(tokens, vec!["NUM", "STR", "BOOL", "NIL", "LIT"]);
    }

    #[test]
    fn literal_tag_is_copyable_and_debuggable() {
        let tag = LiteralTag::Number;
        let copy = tag;
        assert_eq!(tag, copy);
        assert!(format!("{tag:?}").contains("Number"));
    }

    // ------------------------------------------------------------------ Role

    #[test]
    fn role_is_copyable_and_debuggable() {
        let role = Role::Literal(LiteralTag::String);
        let copy = role;
        assert_eq!(role, copy);
        assert!(format!("{role:?}").contains("Literal"));
        assert_ne!(role, Role::Structural);
        assert_ne!(Role::Identifier, Role::Ignored);
    }

    // -------------------------------------------------------- classification

    #[test]
    fn an_identifier_kind_is_classified_as_an_identifier() {
        assert_eq!(PYTHON.role("identifier", true), Role::Identifier);
    }

    #[test]
    fn literal_kinds_carry_their_type_tag() {
        let kinds = ["integer", "float", "string", "true", "none", "ellipsis"];
        let roles: Vec<Role> = kinds.iter().map(|k| PYTHON.role(k, true)).collect();
        assert_eq!(
            roles,
            vec![
                Role::Literal(LiteralTag::Number),
                Role::Literal(LiteralTag::Number),
                Role::Literal(LiteralTag::String),
                Role::Literal(LiteralTag::Boolean),
                Role::Literal(LiteralTag::Nil),
                Role::Literal(LiteralTag::Other),
            ]
        );
    }

    #[test]
    fn an_unlisted_named_kind_is_structural() {
        assert_eq!(PYTHON.role("if_statement", true), Role::Structural);
    }

    #[test]
    fn a_comment_is_ignored_even_though_it_is_a_named_node() {
        assert_eq!(PYTHON.role("comment", true), Role::Ignored);
    }

    #[test]
    fn an_anonymous_operator_is_kept_as_structure() {
        // Losing these would make `a + b` and `a - b` identical.
        let roles: Vec<Role> = ["+", "-=", "**"].iter().map(|k| PYTHON.role(k, false)).collect();
        assert_eq!(roles, vec![Role::Structural; 3]);
    }

    #[test]
    fn an_anonymous_word_operator_is_kept_as_structure() {
        // Python spells conjunction as a word; dropping it would make
        // `a and b` indistinguishable from `a or b`.
        let roles: Vec<Role> =
            ["and", "or", "not", "in", "is"].iter().map(|k| PYTHON.role(k, false)).collect();
        assert_eq!(roles, vec![Role::Structural; 5]);
    }

    #[test]
    fn an_anonymous_keyword_is_ignored_because_its_parent_already_says_so() {
        let roles: Vec<Role> =
            ["def", "if", "return", "class", "await"].iter().map(|k| PYTHON.role(k, false)).collect();
        assert_eq!(roles, vec![Role::Ignored; 5]);
    }

    #[test]
    fn anonymous_grouping_punctuation_is_ignored() {
        let punct = ["(", ")", "[", "]", "{", "}", ",", ":", ".", "@", "->"];
        let roles: Vec<Role> = punct.iter().map(|k| PYTHON.role(k, false)).collect();
        assert_eq!(roles, vec![Role::Ignored; punct.len()]);
    }

    #[test]
    fn the_operator_table_has_no_duplicates() {
        let mut sorted = OPERATOR_TOKENS.to_vec();
        let len_with_dupes = sorted.len();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), len_with_dupes);
    }

    // ------------------------------------------------------------ name fields

    #[test]
    fn a_unit_kind_reports_the_field_holding_its_name() {
        assert_eq!(PYTHON.unit_name_field("function_definition"), Some("name"));
        assert_eq!(PYTHON.unit_name_field("class_definition"), None);
    }

    #[test]
    fn a_scope_kind_reports_the_field_holding_its_name() {
        assert_eq!(PYTHON.scope_name_field("class_definition"), Some("name"));
        assert_eq!(PYTHON.scope_name_field("function_definition"), Some("name"));
        assert_eq!(PYTHON.scope_name_field("if_statement"), None);
    }

    // --------------------------------------------------------------- registry

    #[test]
    fn the_registry_exposes_python_and_nothing_it_does_not_have() {
        assert_eq!(by_name("python").map(|l| l.name), Some("python"));
        assert!(by_name("cobol").is_none());
    }

    #[test]
    fn a_path_resolves_to_a_language_by_extension() {
        assert_eq!(for_path(Path::new("a/b/c.py")).map(|l| l.name), Some("python"));
        assert_eq!(for_path(Path::new("stubs.pyi")).map(|l| l.name), Some("python"));
    }

    #[test]
    fn extension_matching_ignores_case() {
        assert_eq!(for_path(Path::new("SHOUTING.PY")).map(|l| l.name), Some("python"));
    }

    #[test]
    fn a_path_with_no_or_unknown_extension_resolves_to_nothing() {
        assert!(for_path(Path::new("Makefile")).is_none());
        assert!(for_path(Path::new("notes.txt")).is_none());
        assert!(!is_supported(Path::new("notes.txt")));
        assert!(is_supported(&PathBuf::from("x.py")));
    }

    #[test]
    fn every_registered_language_builds_its_grammar_and_declares_kinds_it_really_has() {
        // A registry entry whose grammar fails to load, or that declares a node
        // kind the grammar has never heard of, is an entry that silently finds
        // nothing. That is the failure mode this whole project exists to
        // prevent, so it is checked for every language, not just new ones.
        let mut checks: Vec<(&str, bool, bool, bool, bool)> = Vec::new();
        for lang in all() {
            let grammar = lang.grammar();
            let named_kind_known = |k: &&str| grammar.id_for_node_kind(k, true) != 0;
            checks.push((
                lang.name,
                !lang.extensions.is_empty(),
                !lang.unit_kinds.is_empty(),
                lang.unit_kinds.iter().map(|(k, _)| k).all(named_kind_known)
                    && lang.scope_kinds.iter().map(|(k, _)| k).all(named_kind_known),
                lang.identifier_kinds.iter().all(named_kind_known)
                    && lang.literal_kinds.iter().map(|(k, _)| k).all(named_kind_known)
                    && lang.ignored_kinds.iter().all(named_kind_known),
            ));
        }
        let expected: Vec<_> = all().iter().map(|l| (l.name, true, true, true, true)).collect();
        assert_eq!(checks, expected);
    }

    #[test]
    fn registry_names_and_extensions_are_unique_across_languages() {
        // Two languages claiming `.h` would make `for_path` order-dependent.
        let mut names: Vec<&str> = all().iter().map(|l| l.name).collect();
        let name_count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), name_count);

        let mut exts: Vec<&str> = all().iter().flat_map(|l| l.extensions.iter().copied()).collect();
        let ext_count = exts.len();
        exts.sort_unstable();
        exts.dedup();
        assert_eq!(exts.len(), ext_count);
    }

    #[test]
    fn a_language_debugs_with_its_name() {
        assert!(format!("{PYTHON:?}").contains("python"));
    }
}
