//! Normalized token streams and their content identity.
//!
//! A [`TokenStream`] is what every detector reduces a piece of code to: a flat
//! sequence of *structural* tokens with identifiers blind-renamed and literals
//! abstracted to type tags. Two streams that compare equal describe two pieces
//! of code with the same shape, whatever they called their variables.
//!
//! # Why tokens are interned
//!
//! Similarity is an O(n²) comparison over every pair of units in a tree. Doing
//! that on `String`s means hashing and comparing bytes in the innermost loop.
//! Interning maps each distinct token name to a `u32` once, so the hot loop
//! compares integers.
//!
//! # Why the content hash is NOT computed from interned ids
//!
//! Interned ids are assignment-order dependent: the same function scanned in a
//! different file order gets different ids. The whole point of a content hash
//! is that it is *stable across processes* — the delta engine compares a hash
//! produced by a scan of the HEAD tree against one produced by a separate scan
//! of the merge-base tree. So the hash is always computed over the token
//! **names**, never over their ids.

use serde::{Deserialize, Serialize};

/// Separator used when hashing a token stream.
///
/// An in-band-impossible byte, so that `["ab", "c"]` and `["a", "bc"]` cannot
/// collide by concatenation.
const HASH_SEPARATOR: u8 = 0x1f;

/// Number of hex characters kept from the underlying digest.
///
/// 32 hex chars = 128 bits. Collisions are the only way the delta engine can
/// mistake genuinely-new duplication for pre-existing duplication and stay
/// silent about it, so this is deliberately far above birthday-bound risk for
/// the millions-of-units scale this tool targets.
const HASH_HEX_LEN: usize = 32;

/// Stable identity of a normalized token stream.
///
/// Independent of file path, symbol name, and line numbers: a unit that moved
/// or was renamed between two scans keeps the same [`ContentHash`]. That is
/// precisely what lets the delta engine say "this duplicate already existed,
/// it just moved" instead of re-reporting it on every pull request.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(String);

impl ContentHash {
    /// Hash a sequence of token names.
    ///
    /// This is the canonical constructor: identical `names` always produce an
    /// identical hash, in any process, on any machine.
    pub fn of<S: AsRef<str>>(names: &[S]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for name in names {
            hasher.update(name.as_ref().as_bytes());
            hasher.update(&[HASH_SEPARATOR]);
        }
        let hex = hasher.finalize().to_hex();
        ContentHash(hex[..HASH_HEX_LEN].to_string())
    }

    /// The hash as a lowercase hex string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Maps token names to dense `u32` ids for cheap comparison.
///
/// One interner is shared by every unit in a single scan, so that two units
/// from different files that use the same token get the same id.
#[derive(Debug, Default)]
pub struct Interner {
    ids: std::collections::HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    /// Create an empty interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the id for `name`, assigning a fresh one if unseen.
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    /// Recover the name behind an id, or `None` if the id was never issued.
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }

    /// Number of distinct tokens interned so far.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether nothing has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// A normalized, interned token sequence plus its stable content hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenStream {
    tokens: Vec<u32>,
    hash: ContentHash,
}

impl TokenStream {
    /// Intern `names` into `interner` and capture the stream's content hash.
    ///
    /// The hash is taken over `names` before interning, so it does not depend
    /// on what the interner had already seen.
    pub fn intern<S: AsRef<str>>(names: &[S], interner: &mut Interner) -> Self {
        let hash = ContentHash::of(names);
        let tokens = names.iter().map(|n| interner.intern(n.as_ref())).collect();
        TokenStream { tokens, hash }
    }

    /// The interned token ids, for similarity comparison.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The stream's stable content hash.
    pub fn hash(&self) -> &ContentHash {
        &self.hash
    }

    /// Number of tokens in the stream.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the stream carries no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------ ContentHash

    #[test]
    fn content_hash_is_stable_for_identical_names() {
        assert_eq!(ContentHash::of(&["a", "b"]), ContentHash::of(&["a", "b"]));
    }

    #[test]
    fn content_hash_differs_when_token_order_differs() {
        assert_ne!(ContentHash::of(&["a", "b"]), ContentHash::of(&["b", "a"]));
    }

    #[test]
    fn content_hash_separator_prevents_concatenation_collisions() {
        // Without a separator, ["ab","c"] and ["a","bc"] both hash "abc".
        assert_ne!(ContentHash::of(&["ab", "c"]), ContentHash::of(&["a", "bc"]));
    }

    #[test]
    fn content_hash_of_empty_stream_is_defined_and_distinct_from_one_empty_token() {
        let empty: [&str; 0] = [];
        assert_ne!(ContentHash::of(&empty), ContentHash::of(&[""]));
    }

    #[test]
    fn content_hash_is_fixed_width_lowercase_hex() {
        let h = ContentHash::of(&["x"]);
        assert_eq!(h.as_str().len(), HASH_HEX_LEN);
        assert!(h.as_str().chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn content_hash_displays_and_debugs_as_its_hex() {
        let h = ContentHash::of(&["x"]);
        assert_eq!(h.to_string(), h.as_str());
        assert!(format!("{h:?}").contains(h.as_str()));
    }

    #[test]
    fn content_hash_serializes_transparently_as_a_string() {
        let h = ContentHash::of(&["x"]);
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(json, format!("\"{}\"", h.as_str()));
        let back: ContentHash = serde_json::from_str(&json).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn content_hash_orders_and_clones_for_use_as_a_map_key() {
        let mut hashes = [ContentHash::of(&["b"]), ContentHash::of(&["a"])];
        hashes.sort();
        let cloned = hashes[0].clone();
        assert!(hashes[0] <= hashes[1]);
        assert_eq!(cloned, hashes[0]);
    }

    // --------------------------------------------------------------- Interner

    #[test]
    fn interner_returns_the_same_id_for_a_repeated_name() {
        let mut i = Interner::new();
        assert_eq!(i.intern("f"), i.intern("f"));
    }

    #[test]
    fn interner_returns_distinct_dense_ids_for_distinct_names() {
        let mut i = Interner::new();
        assert_eq!(i.intern("a"), 0);
        assert_eq!(i.intern("b"), 1);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn interner_round_trips_an_id_back_to_its_name() {
        let mut i = Interner::new();
        let id = i.intern("call");
        assert_eq!(i.name(id), Some("call"));
    }

    #[test]
    fn interner_reports_none_for_an_id_it_never_issued() {
        let i = Interner::new();
        assert_eq!(i.name(7), None);
    }

    #[test]
    fn interner_starts_empty_and_reports_it() {
        let mut i = Interner::default();
        assert!(i.is_empty());
        i.intern("x");
        assert!(!i.is_empty());
        assert!(format!("{i:?}").contains("Interner"));
    }

    // ------------------------------------------------------------ TokenStream

    #[test]
    fn token_stream_interns_names_into_ids() {
        let mut i = Interner::new();
        let s = TokenStream::intern(&["a", "b", "a"], &mut i);
        assert_eq!(s.tokens(), &[0, 1, 0]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
    }

    #[test]
    fn token_stream_hash_ignores_interner_state() {
        // The same names interned into two interners primed differently must
        // still hash identically -- this is the cross-process stability that
        // the whole delta engine rests on.
        let mut fresh = Interner::new();
        let mut primed = Interner::new();
        primed.intern("unrelated");

        let a = TokenStream::intern(&["x", "y"], &mut fresh);
        let b = TokenStream::intern(&["x", "y"], &mut primed);

        assert_ne!(a.tokens(), b.tokens(), "ids must differ, proving the point");
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn token_stream_of_no_tokens_is_empty() {
        let mut i = Interner::new();
        let empty: [&str; 0] = [];
        let s = TokenStream::intern(&empty, &mut i);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn token_stream_clones_and_compares_and_debugs() {
        let mut i = Interner::new();
        let s = TokenStream::intern(&["a"], &mut i);
        let c = s.clone();
        assert_eq!(s, c);
        assert!(format!("{s:?}").contains("TokenStream"));
    }
}
