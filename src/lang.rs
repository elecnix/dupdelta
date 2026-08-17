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
//!
//! # A kind string can name two different nodes
//!
//! Several grammars reuse a literal's kind string as the anonymous token that
//! spells a type keyword: TypeScript's `predefined_type` wraps an anonymous
//! child literally named `number` -- the exact string this registry uses for
//! the *numeric literal* kind. Without checking `named`, [`Language::role`]
//! would tag that anonymous type keyword `NUM`, when it is a keyword no
//! different from `if` or `def` and should be dropped like one. So
//! [`Language::role`] only matches [`Language::identifier_kinds`] and
//! [`Language::literal_kinds`] against *named* nodes; an anonymous node
//! sharing the string falls through to the keyword/operator rule instead.

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
        // Identifier and literal kinds are matched by string, and a grammar
        // can reuse a literal's kind string as an anonymous keyword's
        // spelling (see the module docs). Gate both on `named` so that
        // collision can only ever mis-tag the identifier/literal that legally
        // owns the string, never the anonymous keyword sharing it.
        if named && self.identifier_kinds.contains(&kind) {
            return Role::Identifier;
        }
        if named {
            if let Some(&(_, tag)) = self.literal_kinds.iter().find(|(k, _)| *k == kind) {
                return Role::Literal(tag);
            }
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

/// C.
static C: Language = Language {
    name: "c",
    extensions: &["c", "h"],
    grammar: || tree_sitter_c::LANGUAGE.into(),
    // `function_definition`'s own fields are `type`, `declarator`, `body` --
    // never `name`. C's declarator grammar names a function through a nested
    // `function_declarator` (itself wrapped again for a pointer return type),
    // so the plain identifier is a grandchild, not a field of the unit node.
    // `unit_name_field` only supports a single field lookup on the unit
    // itself, so the qualname segment is `function_declarator`'s own text --
    // the name *and* its parameter list, e.g. `"add(int a, int b)"` rather
    // than the bare `"add"`. This is still deterministic and never wrong, and
    // it is the only name the extractor can reach without extract.rs
    // resolving nested declarators, which is out of this file's reach. Pinned
    // by `a_free_function_is_extracted_with_its_signature_as_its_name`.
    unit_kinds: &[("function_definition", "declarator")],
    // Standard C has no class/struct-nested method to qualify a unit by --
    // functions are always top-level. `struct`/`union`/`enum` are registered
    // anyway for any type declared nested inside another (C permits an inline
    // nested struct/union field). `function_definition` is registered too:
    // tree-sitter-c's grammar (matching GCC's nested-function extension)
    // parses a `function_definition` inside another one's body as a real
    // nested node, so this qualifies it the same way Rust qualifies a nested
    // `fn`. Pinned by `a_nested_function_is_qualified_by_its_enclosing_function`.
    scope_kinds: &[
        ("struct_specifier", "name"),
        ("union_specifier", "name"),
        ("enum_specifier", "name"),
        ("function_definition", "declarator"),
    ],
    identifier_kinds: &["identifier", "field_identifier", "type_identifier", "statement_identifier"],
    literal_kinds: &[
        ("number_literal", LiteralTag::Number),
        ("string_literal", LiteralTag::String),
        ("char_literal", LiteralTag::String),
        ("true", LiteralTag::Boolean),
        ("false", LiteralTag::Boolean),
        ("null", LiteralTag::Nil), // the NULL macro; tree-sitter-c recognizes it as a literal node
    ],
    ignored_kinds: &["comment"],
};

/// C++.
///
/// Shares C's declarator-naming quirk (see the note on [`C`]) for free
/// functions *and* methods: a class's methods are `function_definition`
/// nodes nested inside `class_specifier`'s body, exactly like C's free
/// functions, so the same single-field limitation applies to both.
static CPP: Language = Language {
    name: "cpp",
    extensions: &["cpp", "cc", "cxx", "hpp", "hh", "hxx"],
    grammar: || tree_sitter_cpp::LANGUAGE.into(),
    unit_kinds: &[
        ("function_definition", "declarator"),
        // A lambda's `declarator` field is its parameter list, not a name --
        // registering "name" instead (a field it does not have) reaches the
        // same `<anonymous@line>` fallback JavaScript's arrow_function uses,
        // rather than surfacing the parameter list as a fake name.
        ("lambda_expression", "name"),
    ],
    scope_kinds: &[
        ("struct_specifier", "name"),
        ("class_specifier", "name"),
        ("union_specifier", "name"),
        ("enum_specifier", "name"),
        ("namespace_definition", "name"),
        ("function_definition", "declarator"),
    ],
    identifier_kinds: &[
        "identifier",
        "field_identifier",
        "type_identifier",
        "statement_identifier",
        "namespace_identifier",
    ],
    literal_kinds: &[
        ("number_literal", LiteralTag::Number),
        ("string_literal", LiteralTag::String),
        ("raw_string_literal", LiteralTag::String),
        ("char_literal", LiteralTag::String),
        ("true", LiteralTag::Boolean),
        ("false", LiteralTag::Boolean),
        ("null", LiteralTag::Nil), // covers both NULL and nullptr
    ],
    ignored_kinds: &["comment"],
};

/// C#.
static CSHARP: Language = Language {
    name: "csharp",
    extensions: &["cs"],
    grammar: || tree_sitter_c_sharp::LANGUAGE.into(),
    unit_kinds: &[
        ("method_declaration", "name"),
        ("constructor_declaration", "name"),
        ("local_function_statement", "name"),
        ("lambda_expression", "name"),
        ("anonymous_method_expression", "name"),
    ],
    scope_kinds: &[
        ("class_declaration", "name"),
        ("interface_declaration", "name"),
        ("struct_declaration", "name"),
        ("namespace_declaration", "name"),
        ("method_declaration", "name"),
        ("constructor_declaration", "name"),
        ("local_function_statement", "name"),
    ],
    identifier_kinds: &["identifier"],
    literal_kinds: &[
        ("integer_literal", LiteralTag::Number),
        ("real_literal", LiteralTag::Number),
        ("string_literal", LiteralTag::String),
        ("verbatim_string_literal", LiteralTag::String),
        ("raw_string_literal", LiteralTag::String),
        ("character_literal", LiteralTag::String),
        ("boolean_literal", LiteralTag::Boolean),
        ("null_literal", LiteralTag::Nil),
    ],
    ignored_kinds: &["comment"],
};

/// Go.
///
/// Note what is deliberately absent: a method's receiver type (`func (p
/// Point) Area()`) is a sibling field of the method's own `name`, not an
/// ancestor node, because Go does not nest a method inside its receiver's
/// type declaration the way a class nests a method. `scope_kinds` can only
/// contribute a qualname prefix from an *ancestor*, so a Go method's qualname
/// is its bare name -- `"Area"`, never `"Point.Area"`. Reaching the latter
/// would mean reading the `receiver` field when building a method's own
/// qualname, which needs extract.rs to compose two fields into one segment;
/// this file cannot do that alone.
static GO: Language = Language {
    name: "go",
    extensions: &["go"],
    grammar: || tree_sitter_go::LANGUAGE.into(),
    unit_kinds: &[("function_declaration", "name"), ("method_declaration", "name"), ("func_literal", "name")],
    scope_kinds: &[("type_spec", "name"), ("function_declaration", "name"), ("method_declaration", "name")],
    identifier_kinds: &[
        "identifier",
        "field_identifier",
        "type_identifier",
        "package_identifier",
        "blank_identifier",
    ],
    literal_kinds: &[
        ("int_literal", LiteralTag::Number),
        ("float_literal", LiteralTag::Number),
        ("imaginary_literal", LiteralTag::Number),
        ("interpreted_string_literal", LiteralTag::String),
        ("raw_string_literal", LiteralTag::String),
        ("rune_literal", LiteralTag::String),
        ("true", LiteralTag::Boolean),
        ("false", LiteralTag::Boolean),
        ("nil", LiteralTag::Nil),
    ],
    ignored_kinds: &["comment"],
};

/// Java.
static JAVA: Language = Language {
    name: "java",
    extensions: &["java"],
    grammar: || tree_sitter_java::LANGUAGE.into(),
    unit_kinds: &[
        // `method_declaration` also matches an interface method signature,
        // which has no `body` field -- it is still a real, comparable (if
        // empty) unit, and the grammar gives no other kind to tell the two
        // apart. `min_nodes` naturally filters out the near-empty ones.
        ("method_declaration", "name"),
        ("constructor_declaration", "name"),
        ("lambda_expression", "name"),
    ],
    scope_kinds: &[
        ("class_declaration", "name"),
        ("interface_declaration", "name"),
        ("enum_declaration", "name"),
        ("method_declaration", "name"),
        ("constructor_declaration", "name"),
    ],
    identifier_kinds: &["identifier", "type_identifier"],
    literal_kinds: &[
        ("decimal_integer_literal", LiteralTag::Number),
        ("hex_integer_literal", LiteralTag::Number),
        ("octal_integer_literal", LiteralTag::Number),
        ("binary_integer_literal", LiteralTag::Number),
        ("decimal_floating_point_literal", LiteralTag::Number),
        ("hex_floating_point_literal", LiteralTag::Number),
        ("string_literal", LiteralTag::String),
        ("character_literal", LiteralTag::String),
        ("true", LiteralTag::Boolean),
        ("false", LiteralTag::Boolean),
        ("null_literal", LiteralTag::Nil),
    ],
    ignored_kinds: &["line_comment", "block_comment"],
};

/// PHP.
static PHP: Language = Language {
    name: "php",
    extensions: &["php"],
    grammar: || tree_sitter_php::LANGUAGE_PHP.into(),
    unit_kinds: &[
        ("method_declaration", "name"),
        ("function_definition", "name"),
        ("anonymous_function", "name"),
        ("arrow_function", "name"),
    ],
    scope_kinds: &[
        ("class_declaration", "name"),
        ("interface_declaration", "name"),
        ("trait_declaration", "name"),
        ("namespace_definition", "name"),
        ("enum_declaration", "name"),
        ("method_declaration", "name"),
        ("function_definition", "name"),
    ],
    // PHP's grammar names every declaration and every variable reference
    // with the same `name` kind (a class's name, a method's name, and a
    // `$variable`'s inner text are all `name` nodes) -- exactly the pattern
    // Python's `identifier` already follows, so no special case is needed.
    identifier_kinds: &["name"],
    literal_kinds: &[
        ("integer", LiteralTag::Number),
        ("float", LiteralTag::Number),
        ("string", LiteralTag::String),
        ("encapsed_string", LiteralTag::String),
        ("heredoc", LiteralTag::String),
        ("nowdoc", LiteralTag::String),
        ("boolean", LiteralTag::Boolean),
        ("null", LiteralTag::Nil),
    ],
    ignored_kinds: &["comment"],
};

/// Ruby.
static RUBY: Language = Language {
    name: "ruby",
    extensions: &["rb"],
    grammar: || tree_sitter_ruby::LANGUAGE.into(),
    unit_kinds: &[
        ("method", "name"),
        ("singleton_method", "name"), // `def self.foo` -- its "object" field (self) is ignored, same tradeoff as Go's receiver
        ("lambda", "name"),           // `->(x) { }`; no name field, falls back to anonymous
        ("block", "name"),            // `{ |x| }` block literal; no name field, falls back to anonymous
        ("do_block", "name"),         // `do |x| ... end` block literal; same as `block`
    ],
    scope_kinds: &[("module", "name"), ("class", "name"), ("method", "name"), ("singleton_method", "name")],
    identifier_kinds: &["identifier", "constant", "instance_variable", "class_variable", "global_variable"],
    literal_kinds: &[
        ("integer", LiteralTag::Number),
        ("float", LiteralTag::Number),
        ("string", LiteralTag::String),
        ("true", LiteralTag::Boolean),
        ("false", LiteralTag::Boolean),
        ("nil", LiteralTag::Nil),
        // A symbol (`:foo`) is not a string -- it is an interned name with no
        // string methods, so it gets `Other` rather than being folded into
        // `String` and made to look like something it is not.
        ("simple_symbol", LiteralTag::Other),
        ("bare_symbol", LiteralTag::Other),
        ("delimited_symbol", LiteralTag::Other),
        ("hash_key_symbol", LiteralTag::Other),
    ],
    ignored_kinds: &["comment"],
};

/// Rust.
static RUST: Language = Language {
    name: "rust",
    extensions: &["rs"],
    grammar: || tree_sitter_rust::LANGUAGE.into(),
    unit_kinds: &[
        ("function_item", "name"),
        ("closure_expression", "name"), // `|x| x * 2`; no name field, falls back to anonymous
    ],
    scope_kinds: &[
        ("struct_item", "name"),
        ("trait_item", "name"),
        // `impl_item` has no `name` field: `impl Trait for Type` and `impl
        // Type` both name their subject through `type`, never `trait`, so a
        // method qualifies by the type it is implemented for regardless of
        // which trait (if any) it implements.
        ("impl_item", "type"),
        ("mod_item", "name"),
        ("function_item", "name"),
    ],
    // `primitive_type` (`i32`, `f64`, ...) is deliberately absent: it is a
    // type keyword, not a name a caller chose, and it is already a childless
    // leaf so leaving it Structural cannot leak its text -- only its kind.
    identifier_kinds: &["identifier", "type_identifier", "field_identifier", "shorthand_field_identifier"],
    literal_kinds: &[
        ("integer_literal", LiteralTag::Number),
        ("float_literal", LiteralTag::Number),
        ("string_literal", LiteralTag::String),
        ("raw_string_literal", LiteralTag::String),
        ("char_literal", LiteralTag::String),
        ("boolean_literal", LiteralTag::Boolean),
    ],
    // Rust's `//!`/`///` doc comments are themselves `line_comment` (or
    // `block_comment`) nodes carrying a nested `doc_comment` child; since an
    // ignored node is never descended into, listing the outer kind is enough.
    ignored_kinds: &["line_comment", "block_comment"],
};

/// Node kinds shared by TypeScript and its JSX dialect, TSX -- both grammars
/// declare the same non-JSX kinds, so the tables live here once rather than
/// twice. Only the grammar function and file extension differ per dialect.
const TS_UNIT_KINDS: &[NamedKind] = &[
    ("function_declaration", "name"),
    ("generator_function_declaration", "name"),
    ("function_expression", "name"),
    ("method_definition", "name"),
    ("arrow_function", "name"),
];
const TS_SCOPE_KINDS: &[NamedKind] = &[
    ("class_declaration", "name"),
    ("variable_declarator", "name"),
    ("function_declaration", "name"),
    ("generator_function_declaration", "name"),
    ("method_definition", "name"),
    ("interface_declaration", "name"),
    ("internal_module", "name"), // `namespace Foo { }`
];
const TS_IDENTIFIER_KINDS: &[&str] = &[
    "identifier",
    "property_identifier",
    "private_property_identifier",
    "shorthand_property_identifier",
    "shorthand_property_identifier_pattern",
    "type_identifier",
];
const TS_LITERAL_KINDS: &[(&str, LiteralTag)] = &[
    ("number", LiteralTag::Number),
    ("string", LiteralTag::String),
    ("template_string", LiteralTag::String),
    ("regex", LiteralTag::Other),
    ("true", LiteralTag::Boolean),
    ("false", LiteralTag::Boolean),
    ("null", LiteralTag::Nil),
    ("undefined", LiteralTag::Nil),
];
const TS_IGNORED_KINDS: &[&str] = &["comment"];

/// TypeScript.
///
/// `number` and `string` are two of TypeScript's oddest kind strings: a type
/// annotation's `predefined_type` wraps an *anonymous* child spelled `number`
/// or `string` -- see the module docs' "a kind string can name two different
/// nodes" -- so `TS_LITERAL_KINDS` relies on [`Language::role`] gating that
/// match on `named` to avoid tagging the type keyword as a literal.
static TYPESCRIPT: Language = Language {
    name: "typescript",
    extensions: &["ts"], // "tsx" is claimed by the TSX dialect below, not this one
    grammar: || tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
    unit_kinds: TS_UNIT_KINDS,
    scope_kinds: TS_SCOPE_KINDS,
    identifier_kinds: TS_IDENTIFIER_KINDS,
    literal_kinds: TS_LITERAL_KINDS,
    ignored_kinds: TS_IGNORED_KINDS,
};

/// TSX -- TypeScript with JSX. A distinct grammar and registry entry from
/// [`TYPESCRIPT`] (per [`tree_sitter_typescript::LANGUAGE_TSX`]), because JSX
/// syntax is ambiguous with TypeScript's generic-arguments syntax and the two
/// grammars resolve it differently; sharing one grammar would misparse one of
/// the two. Node kinds this file cares about are identical either way, so the
/// tables above are reused as-is.
static TSX: Language = Language {
    name: "tsx",
    extensions: &["tsx"],
    grammar: || tree_sitter_typescript::LANGUAGE_TSX.into(),
    unit_kinds: TS_UNIT_KINDS,
    scope_kinds: TS_SCOPE_KINDS,
    identifier_kinds: TS_IDENTIFIER_KINDS,
    literal_kinds: TS_LITERAL_KINDS,
    ignored_kinds: TS_IGNORED_KINDS,
};

/// Every language the registry knows, ordered by name.
///
/// Adding a language means adding a `static` above and one entry here. No code
/// outside this module changes.
static ALL: &[&Language] =
    &[&C, &CPP, &CSHARP, &GO, &JAVA, &JAVASCRIPT, &PHP, &PYTHON, &RUBY, &RUST, &TSX, &TYPESCRIPT];

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

    // ---------------------------------------------------- the ten additions
    //
    // These drive the real `crate::extract::Extractor`, not a hand-copied
    // formula (CONTRIBUTING.md's "test the engine, not a copy of it"): each
    // test asserts on `Unit`s the extractor actually produced.

    use crate::extract::Extractor;
    use crate::token::{ContentHash, Interner};

    /// Qualified names `Extractor` finds in `source`, in source order.
    fn qualnames_of(lang: &'static Language, source: &str) -> Vec<String> {
        let mut interner = Interner::new();
        let path = PathBuf::from(format!("sample.{}", lang.extensions[0]));
        Extractor::new(lang)
            .extract(source, &path, 1, &mut interner)
            .units
            .into_iter()
            .map(|u| u.qualname)
            .collect()
    }

    /// Content hashes `Extractor` finds in `source`, in source order.
    fn hashes_of(lang: &'static Language, source: &str) -> Vec<ContentHash> {
        let mut interner = Interner::new();
        let path = PathBuf::from(format!("sample.{}", lang.extensions[0]));
        Extractor::new(lang)
            .extract(source, &path, 1, &mut interner)
            .units
            .into_iter()
            .map(|u| u.stream.hash().clone())
            .collect()
    }

    // ---- C, C++, Go: kept as dedicated tests, not folded into the table
    // below.
    //
    // C and C++ name a unit through a nested declarator rather than a `name`
    // field (see the notes on `C` and `CPP`), so their qualnames carry a
    // parameter list -- and, for C++, a trailing `const` -- instead of the
    // plain `Type.method` every other language produces. Go's method
    // receiver is a sibling field rather than an ancestor (see the note on
    // `GO`), so a Go method's qualname is deliberately bare, never
    // `Type.method`. Each of these is a documented, load-bearing exception;
    // squeezing it into a generic table would either drop the assertion
    // that makes the exception visible or bury its rationale under rows that
    // do not need one.

    #[test]
    fn a_free_function_is_extracted_with_its_signature_as_its_name() {
        // C names a function through a nested `function_declarator`, not a
        // `name` field on `function_definition` itself (see the note on
        // `C`), so the qualname is the declarator's own text -- name and
        // parameter list together, not the bare name. `outer`/`inner` also
        // proves `scope_kinds` nests a GNU nested function under its
        // enclosing one.
        let source = "struct Point { int x; int y; };\n\nint square(int n) {\n    return n * n;\n}\n\nint outer() {\n    int inner() {\n        return 1;\n    }\n    return inner();\n}\n";
        assert_eq!(qualnames_of(&C, source), vec!["square(int n)", "outer()", "outer().inner()"]);
    }

    #[test]
    fn a_cpp_method_is_qualified_by_its_class() {
        // Same declarator quirk as C (see the note on `CPP`): `area`'s own
        // name segment includes its `const` qualifier.
        let source = "class Circle {\npublic:\n    double area() const {\n        return 3.14 * 2.0;\n    }\n};\n\nint add(int a, int b) {\n    return a + b;\n}\n";
        assert_eq!(qualnames_of(&CPP, source), vec!["Circle.area() const", "add(int a, int b)"]);
    }

    #[test]
    fn a_go_method_is_extracted_by_its_own_name_not_its_receiver_type() {
        // See the note on `GO`: `Area`'s receiver `Point` is a sibling
        // field, not an ancestor, so it cannot contribute a qualname
        // prefix -- this is the honest, tested consequence of that limit.
        let source = "package main\n\ntype Point struct {\n\tX int\n\tY int\n}\n\nfunc Add(a int, b int) int {\n\treturn a + b\n}\n\nfunc (p Point) Area() int {\n\treturn p.X * p.Y\n}\n";
        assert_eq!(qualnames_of(&GO, source), vec!["Add", "Area"]);
    }

    // ---- qualnames: table-driven across every language whose units follow
    // the plain `Type.method` / bare-function pattern (C, C++, and Go are
    // excluded -- see above).

    #[test]
    fn a_method_is_qualified_by_its_enclosing_type() {
        let cases: &[(&Language, &str, &[&str])] = &[
            (
                &CSHARP,
                "class Circle {\n    public double Area() {\n        return 3.14 * 2.0;\n    }\n}\n\nclass Program {\n    static int Add(int a, int b) {\n        return a + b;\n    }\n}\n",
                &["Circle.Area", "Program.Add"],
            ),
            (
                &JAVA,
                "class Circle {\n    Circle() {}\n    double area() {\n        return 3.14 * 2.0;\n    }\n}\n\nclass Main {\n    static int add(int a, int b) {\n        return a + b;\n    }\n}\n",
                &["Circle.Circle", "Circle.area", "Main.add"],
            ),
            (
                &PHP,
                "<?php\nclass Circle {\n    public function area() {\n        return 3.14 * 2.0;\n    }\n}\n\nfunction add($a, $b) {\n    return $a + $b;\n}\n",
                &["Circle.area", "add"],
            ),
            (
                &RUBY,
                "class Circle\n  def area\n    3.14 * 2.0\n  end\nend\n\ndef add(a, b)\n  a + b\nend\n",
                &["Circle.area", "add"],
            ),
            (
                &RUST,
                "struct Circle;\n\nimpl Circle {\n    fn area(&self) -> f64 {\n        3.14 * 2.0\n    }\n}\n\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
                &["Circle.area", "add"],
            ),
            (
                &TYPESCRIPT,
                "class Circle {\n  area(): number {\n    return 3.14 * 2;\n  }\n}\n\nfunction add(a: number, b: number): number {\n  return a + b;\n}\n",
                &["Circle.area", "add"],
            ),
            (
                &TSX,
                // Also proves JSX parses cleanly: `App`'s body returns a JSX element.
                "class Widget {\n  render(): number {\n    return 1;\n  }\n}\n\nfunction App() {\n  return <div>hi</div>;\n}\n",
                &["Widget.render", "App"],
            ),
        ];

        let got: Vec<(&str, Vec<String>)> =
            cases.iter().map(|(lang, source, _)| (lang.name, qualnames_of(lang, source))).collect();
        let want: Vec<(&str, Vec<String>)> = cases
            .iter()
            .map(|(lang, _, expected)| (lang.name, expected.iter().map(|s| s.to_string()).collect()))
            .collect();
        assert_eq!(got, want);
    }

    // ---- hashes: table-driven across every language with a hash-invariance
    // test. Unlike the qualname table above, C, C++, and Go fit here without
    // exception -- the quirks that keep them out of the qualname table only
    // affect how a unit is *named*, not how its body hashes.

    #[test]
    fn functions_differing_only_in_names_share_a_hash_but_not_in_operator() {
        let cases: &[(&Language, &str)] = &[
            (
                &C,
                "int add(int a, int b) { return a + b; }\nint sum(int x, int y) { return x + y; }\nint sub(int a, int b) { return a - b; }\n",
            ),
            (
                &CPP,
                "int add(int a, int b) { return a + b; }\nint sum(int x, int y) { return x + y; }\nint sub(int a, int b) { return a - b; }\n",
            ),
            (
                &CSHARP,
                "class C {\n    static int Add(int a, int b) { return a + b; }\n    static int Sum(int x, int y) { return x + y; }\n    static int Sub(int a, int b) { return a - b; }\n}\n",
            ),
            (
                &GO,
                "package main\nfunc add(a int, b int) int { return a + b }\nfunc sum(x int, y int) int { return x + y }\nfunc sub(a int, b int) int { return a - b }\n",
            ),
            (
                &JAVA,
                "class C {\n    static int add(int a, int b) { return a + b; }\n    static int sum(int x, int y) { return x + y; }\n    static int sub(int a, int b) { return a - b; }\n}\n",
            ),
            (
                &PHP,
                "<?php\nfunction add($a, $b) { return $a + $b; }\nfunction sum($x, $y) { return $x + $y; }\nfunction sub($a, $b) { return $a - $b; }\n",
            ),
            (&RUBY, "def add(a, b)\n  a + b\nend\n\ndef sum(x, y)\n  x + y\nend\n\ndef sub(a, b)\n  a - b\nend\n"),
            (
                &RUST,
                "fn add(a: i32, b: i32) -> i32 { a + b }\nfn sum(x: i32, y: i32) -> i32 { x + y }\nfn sub(a: i32, b: i32) -> i32 { a - b }\n",
            ),
            (
                &TYPESCRIPT,
                "function add(a: number, b: number): number { return a + b; }\nfunction sum(x: number, y: number): number { return x + y; }\nfunction sub(a: number, b: number): number { return a - b; }\n",
            ),
            (
                &TSX,
                "function add(a: number, b: number): number { return a + b; }\nfunction sum(x: number, y: number): number { return x + y; }\nfunction sub(a: number, b: number): number { return a - b; }\n",
            ),
        ];

        // (name, unit count, renamed copy hashes the same, different operator hashes differently)
        let got: Vec<(&str, usize, bool, bool)> = cases
            .iter()
            .map(|(lang, source)| {
                let hashes = hashes_of(lang, source);
                let renamed_matches = hashes.first().zip(hashes.get(1)).is_some_and(|(a, b)| a == b);
                let operator_differs = hashes.first().zip(hashes.get(2)).is_some_and(|(a, c)| a != c);
                (lang.name, hashes.len(), renamed_matches, operator_differs)
            })
            .collect();
        let want: Vec<(&str, usize, bool, bool)> =
            cases.iter().map(|(lang, _)| (lang.name, 3, true, true)).collect();
        assert_eq!(got, want);
    }

    // ---- TypeScript: kept separate
    //
    // Exercises the `named`-gated collision fix on `role`, a different
    // scenario from the shared rename/operator shape in the table above, so
    // it does not belong in either table.

    #[test]
    fn typescript_scalar_type_annotations_do_not_leak_into_the_hash() {
        // Pins the `named`-gated fix documented on `role`: TypeScript's
        // `predefined_type` wraps an anonymous child spelled `number` or
        // `string` -- exactly the strings `TS_LITERAL_KINDS` uses for the
        // numeric/string literal kinds. Without gating that match on
        // `named`, these two functions -- identical but for a return type
        // that is a keyword, not a value -- would hash differently.
        let source =
            "function f(a: number): number { return a; }\nfunction g(a: string): string { return a; }\n";
        let hashes = hashes_of(&TYPESCRIPT, source);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], hashes[1]);
    }

    // ---------------------------------------------------------- cross-cutting

    /// Every language name exercised by a dedicated extraction test above.
    /// Kept as an explicit list (mirroring `registry_names_and_extensions_are_unique_across_languages`'s
    /// style) rather than derived, because there is no way to introspect
    /// "which tests ran" from within a test -- so this is a manual ledger a
    /// reviewer must update, and the assertion below is what makes forgetting
    /// to update it visible: a language added to `ALL` without a matching
    /// entry here fails the count or the set comparison.
    const LANGUAGES_WITH_EXTRACTION_TESTS: &[&str] = &[
        "c",
        "cpp",
        "csharp",
        "go",
        "java",
        "javascript",
        "php",
        "python",
        "ruby",
        "rust",
        "tsx",
        "typescript",
    ];

    #[test]
    fn every_registered_language_has_a_dedicated_extraction_test() {
        assert_eq!(all().len(), 12);
        let mut registered: Vec<&str> = all().iter().map(|l| l.name).collect();
        registered.sort_unstable();
        let mut tested: Vec<&str> = LANGUAGES_WITH_EXTRACTION_TESTS.to_vec();
        tested.sort_unstable();
        assert_eq!(registered, tested);
    }
}
