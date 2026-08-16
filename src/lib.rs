//! **dupdelta** — delta-aware, multi-language near-duplicate code detection.
//!
//! Most duplication detectors answer the question "how much duplication does
//! this codebase contain?" That number is large, mostly pre-existing, mostly
//! already accepted, and it does not change much between one commit and the
//! next — so a CI job built on it warns about the same things on every pull
//! request and is muted within a week.
//!
//! `dupdelta` answers a different question: **"did this change make it worse?"**
//! It scans two trees — your branch and its merge-base — and reports only what
//! the diff between them introduced. Nothing to triage, no allowlist to
//! maintain, no baseline file to keep current. Duplication that was already
//! there is silent; duplication you just added is not.
//!
//! # Layers
//!
//! - [`token`] — normalized token streams and their cross-process content
//!   identity. Everything else is built on these.
//! - [`similarity`] — Ratcliff–Obershelp similarity with a two-tier prune.
//! - [`unionfind`] — grouping duplicate pairs into classes.
//!
//! Further layers (language frontends, extraction, scanning, the delta engine
//! and reporting) build strictly on top of these and are added in sequence.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod annotate;
pub mod config;
pub mod extract;
pub mod lang;
pub mod normalize;
pub mod similarity;
pub mod token;
pub mod unionfind;
pub mod walk;
