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
//! # The scope is a type, not advice
//!
//! Ids are only comparable within the space they were assigned in, so *who
//! shares a space* is a correctness question, not a style one. [`TokenScope`]
//! is that question's answer: a detector asks a scope for a stream and can
//! reach an id no other way. Two scopes are kept apart by construction --
//! each one numbers its ids from its own boundary of [`SCOPE_STRIDE`] -- so a
//! caller who builds one per file cannot end up comparing `return` in one file
//! against `call` in another and calling it a match.
//!
//! # Why the content hash is NOT computed from interned ids
//!
//! Interned ids are assignment-order dependent: the same function scanned in a
//! different file order gets different ids. The whole point of a content hash
//! is that it is *stable across processes* — the delta engine compares a hash
//! produced by a scan of the HEAD tree against one produced by a separate scan
//! of the merge-base tree. So the hash is always computed over the token
//! **names**, never over their ids.

use std::sync::atomic::{AtomicU32, Ordering};

use serde::{Deserialize, Serialize};

/// Ids reserved for one [`TokenScope`].
///
/// A scope numbers its names from its own multiple of this stride, so the
/// ranges of any two live scopes are disjoint however few names either holds.
/// 2¹⁶ names per scope is far above what normalization can produce (it
/// abstracts identifiers and literals away, leaving node kinds and type tags),
/// and 2¹⁶ scopes per process is far above what a scan creates.
const SCOPE_STRIDE: u32 = 1 << 16;

/// Hands out each new scope its own id range.
static NEXT_SCOPE: AtomicU32 = AtomicU32::new(0);

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

/// The id space one scan shares across every unit it extracts.
///
/// # One per scan
///
/// A scan creates exactly one of these and passes it by `&mut` to everything
/// that mints a stream, so that two units from *different files* that use the
/// same token get the same id. That is the whole reason this type exists:
/// without a shared scope, similarity between units in different files is
/// comparing ids that were assigned independently.
///
/// # Two scopes can never be mistaken for one
///
/// Each scope's ids are `namespace + local`, and the namespace is a distinct
/// multiple of [`SCOPE_STRIDE`]. Two scopes therefore never issue the same id
/// for different token names -- which a plain "count from zero" interner does,
/// all the time: `return` interned first in one file and `call` interned first
/// in another both come back as `0`, and every window of ids that lines up
/// positionally then matches. Nothing downstream can catch that. `blocks`'
/// `windows_match` exists to reject 64-bit rolling-hash collisions and cannot:
/// it is handed ids, not names, and from where it sits they really are equal.
///
/// So the failure is made visible rather than merely unlikely. A caller who
/// builds a scope per file still gets correct hashes -- those are taken over
/// names, before interning, and stay stable across processes -- but gets *no*
/// cross-file matches, instead of fabricated ones.
///
/// # What a detector may and may not do
///
/// [`stream`](Self::stream) is the way to get a [`TokenStream`] and
/// [`intern`](Self::intern) is the primitive under it, for the detectors that
/// need raw ids rather than a whole stream. Neither hands out the table itself:
/// the id/name pairing cannot be walked by a caller that might come to rely on
/// it, so there is no way to compare names across a boundary by accident.
#[derive(Debug)]
pub struct TokenScope {
    base: u32,
    ids: std::collections::HashMap<String, u32>,
    names: Vec<String>,
}

impl TokenScope {
    /// Claim a fresh id namespace for one scan.
    ///
    /// Named for what it establishes rather than left as a bare `new`: the
    /// invariant this type carries is about *how many* of these a scan has, and
    /// the call site is the only place that can be honest about it. A caller
    /// that writes this inside a per-file loop has written down the mistake.
    pub fn for_a_scan() -> Self {
        // `fetch_add`, not `fetch_update`: the latter was renamed to `try_update`
        // for consistency, and this crate's MSRV is 1.88, which predates that
        // rename. The bound this needs is not the counter's anyway (below), so
        // the closure `fetch_update` exists to provide was never the check.
        let namespace = NEXT_SCOPE.fetch_add(1, Ordering::Relaxed);
        // No message: same rule as `intern` below -- an `assert!` carrying one
        // leaves that string permanently uncovered (CONTRIBUTING).
        //
        // The bound is on the id space, not the counter. `base` is
        // `namespace * SCOPE_STRIDE`, so this stops one namespace *before*
        // that product wraps u32 -- 2^16 scopes, not 2^32. Checking only the
        // counter's own overflow would let scope 65536 compute a base of 0 and
        // hand itself the same ids as the first scope, which is exactly the
        // collision the namespace exists to prevent.
        assert!(namespace <= u32::MAX / SCOPE_STRIDE);
        TokenScope {
            base: namespace * SCOPE_STRIDE,
            ids: std::collections::HashMap::new(),
            names: Vec::new(),
        }
    }

    /// Return the id for `name` within this scope, assigning a fresh one if
    /// unseen.
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let local = self.names.len();
        // No message: `assert!` only evaluates one when it fails, and that
        // would leave the string permanently uncovered (CONTRIBUTING). The
        // comment is where the explanation belongs.
        assert!(local < SCOPE_STRIDE as usize);
        let id = self.base + local as u32;
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    /// Intern `names` and capture the stream's content hash.
    ///
    /// The hash is taken over `names` before interning, so it does not depend
    /// on what this scope had already seen.
    pub fn stream<S: AsRef<str>>(&mut self, names: &[S]) -> TokenStream {
        let hash = ContentHash::of(names);
        let tokens = names.iter().map(|n| self.intern(n.as_ref())).collect();
        TokenStream { tokens, hash }
    }
}

/// A normalized, interned token sequence plus its stable content hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenStream {
    tokens: Vec<u32>,
    hash: ContentHash,
}

impl TokenStream {
    /// The interned token ids, for similarity comparison.
    ///
    /// Comparable only against other streams from the same
    /// [`TokenScope`]; see that type for why a scope keeps them apart.
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
    use std::collections::BTreeSet;

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

    // ------------------------------------------------------------ TokenScope

    #[test]
    fn two_scopes_never_issue_the_same_id_for_different_names() {
        // The half that cannot be got wrong, and the reason this type is worth
        // its name. Against a plain "count from zero" interner this test
        // *fails*: each interner hands out 0 for its own first name, so
        // `return` in one file and `call` in another compare equal and every
        // positionally aligned window matches -- fabricated findings that no
        // hash check catches, because the hashes are taken over names and stay
        // perfectly correct throughout.
        let mut one = TokenScope::for_a_scan();
        let mut two = TokenScope::for_a_scan();
        let first: BTreeSet<u32> = ["return", "ID", "#END"].iter().copied().map(|n| one.intern(n)).collect();
        let second: BTreeSet<u32> = ["call", "NUM"].iter().copied().map(|n| two.intern(n)).collect();
        assert_eq!(first.intersection(&second).count(), 0);
    }

    // ------------------------------------------------------------ TokenStream

    #[test]
    fn token_stream_interns_names_into_ids() {
        let mut scope = TokenScope::for_a_scan();
        let s = scope.stream(&["a", "b", "a"]);
        // Positions are what survive interning: the repeat is the same id,
        // the distinct name is a different one. The absolute values belong to
        // the scope's namespace, not to the caller.
        assert_eq!(s.tokens()[0], s.tokens()[2]);
        assert_ne!(s.tokens()[0], s.tokens()[1]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
    }

    #[test]
    fn token_stream_hash_ignores_scope_state() {
        // The same names interned into two scopes primed differently must
        // still hash identically -- this is the cross-process stability that
        // the whole delta engine rests on.
        let mut fresh = TokenScope::for_a_scan();
        let mut primed = TokenScope::for_a_scan();
        primed.intern("unrelated");

        let a = fresh.stream(&["x", "y"]);
        let b = primed.stream(&["x", "y"]);

        // Ids differ -- the scopes' namespaces are disjoint -- while the hash
        // does not, because it was taken over the names. That contrast is the
        // point of the test.
        assert_ne!(a.tokens(), b.tokens());
        assert_eq!(a.hash(), b.hash());
        assert!(format!("{fresh:?}").contains("TokenScope"));
    }

    #[test]
    fn token_stream_of_no_tokens_is_empty() {
        let mut scope = TokenScope::for_a_scan();
        let empty: [&str; 0] = [];
        let s = scope.stream(&empty);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn token_stream_clones_and_compares_and_debugs() {
        let mut scope = TokenScope::for_a_scan();
        let s = scope.stream(&["a"]);
        let c = s.clone();
        assert_eq!(s, c);
        assert!(format!("{s:?}").contains("TokenStream"));
    }
}
