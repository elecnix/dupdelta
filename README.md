# dupdelta

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/elecnix/dupdelta)

**Duplication detection that only tells you about duplication *you* introduced.**

Most clone detectors answer "how much duplication does this codebase contain?" That number is
large, mostly pre-existing, mostly already accepted, and it barely moves between one commit and the
next. A CI job built on it warns about the same things on every pull request, and everyone learns to
ignore it inside a week.

`dupdelta` answers a different question:

> **Did this change make it worse?**

It scans two trees — your branch and its merge-base — and reports only what the diff between them
introduced. Duplication that was already there stays silent. There is no baseline file to regenerate,
no allowlist to triage, no threshold to keep tuning. A contributor who introduces no new duplication
sees a clean job, indefinitely, on a codebase that has plenty of it.

---

## Quick start

### As a GitHub Action

```yaml
name: Duplication
on: pull_request

jobs:
  dupdelta:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with:
          fetch-depth: 0        # the merge-base has to be reachable
      - uses: elecnix/dupdelta@v1
```

That is the whole setup. The action resolves the merge-base, scans both trees, and annotates the
pull request inline with anything new. It **warns; it never fails the build** — see
[Why it never blocks a merge](#why-it-never-blocks-a-merge).

### On the command line

```sh
# What would this branch add, compared to main?
dupdelta ci --base main

# Just scan a tree and write a report
dupdelta scan . --out report.json

# Compare two reports you already have
dupdelta diff --head head.json --base base.json
```

---

## What it looks for

Three detectors, because duplication has three shapes and no single technique sees all of them.

### 1. Function clones — renamed copies

Every function is parsed, then **normalized**: identifiers are blind-renamed to a single
placeholder, literals are abstracted to type tags, and structure is kept. Two functions that differ
only in their variable names and magic numbers reduce to the *same* token stream.

```python
def total(rate, years):      def compute(x, n):
    return rate * years          return x * n
```

These are reported as identical. That is deliberate and it is the point: the same rule copied with a
tweaked constant is exactly the duplication worth finding, and a detector that treated `0.05` and
`0.055` as a difference would miss it.

Similarity is Ratcliff–Obershelp — the same measure `difflib.SequenceMatcher(...).ratio()` computes
— which counts only *contiguous* runs, because that is what "copy-pasted, then edited" looks like.

### 2. Module vocabulary — a second engine

Function-level comparison structurally cannot see a module that reimplements another module's rules
with a completely different set of functions. Such a module scores near zero on any clone metric,
while sharing the domain nouns that gave the original its meaning.

So a second detector compares files by their **un-renamed identifier vocabulary** — the exact
inversion of what the first one does, and worth knowing before you "fix" it. Overlap is measured
against the smaller vocabulary, and a pair is ranked first when either side has **no inbound
imports**: heavy vocabulary overlap plus nothing importing it is the signature of a dead second
engine.

That inbound-import count is an acknowledged heuristic. A CLI entry point and a package root both
read as zero-inbound, and neither is dead code. It is a strong prior, never a verdict.

### 3. Blocks — copy-paste inside functions

The function detector compares whole bodies, so a chunk pasted into the middle of two differently
shaped functions is invisible to it. The block detector finds maximal repeated runs of normalized
tokens wherever they sit, including inside a single file.

---

## How the delta works

Every finding carries a **content hash** of the normalized code it describes — not a file path, not
a line number. A finding is "new" only if its hash is absent from the merge-base scan.

This is what makes the silence trustworthy:

| what happened to a pre-existing duplicate | reported? |
|---|---|
| nothing | no |
| the code above it grew, shifting its line numbers | no |
| it was moved to another file | no |
| the function was renamed | no |
| its variables were renamed | no |
| **its logic was changed into a new near-duplicate** | **yes** |

A pair can only be new if at least one side's normalized body did not exist anywhere in the
merge-base tree — which means this change is the reason it exists.

### Why not a committed baseline or an allowlist?

Both were considered. An allowlist works well for a finite, enumerable set of findings a human
triages once. Clone findings are neither: they are combinatorial, and they shift with every
unrelated edit to either side of a pair. A docstring tweak on one of two 86%-similar functions
changes nothing about the duplication but changes the syntax tree. A snippet-keyed allowlist would
go stale constantly, and would either rot into a rubber stamp or demand re-triage of things
unrelated to the change under review.

The merge-base delta needs no human-maintained state at all. It asks "did *this diff* make it
worse", which is the actual question, every time, for free. Once a change lands, its new duplication
becomes part of the merge-base for the next one and stops being reported — exactly like any other
pre-existing pattern.

### Why there is no "was this file touched?" filter

Detectors without a stable per-fragment identity have to fall back on "only report a duplicate if
the pull request touched one of its files". That filter is a poor substitute: it stays silent on
real new duplication in files the diff did not name, and it re-nags about the same untouched
fragment forever once you do touch the file.

Every detector here — including the block detector, where this is traditionally the hard case —
produces a content hash, so the delta is exact and no such filter is needed.

---

## Languages

Parsing is tree-sitter, so a language is *data* in a registry rather than code:

Python · JavaScript · TypeScript · TSX · Rust · Go · Java · Ruby · C · C++ · C# · PHP

A test asserts that every node kind a language declares actually exists in the grammar linked
against it, so a typo cannot quietly turn a language into one that finds nothing.

Adding a language means adding a `static Language` to `src/lang.rs` and one entry to the registry.
No other file changes.

---

## Configuration

Everything tunable lives in `.dupdelta.toml`, discovered by walking up from the working directory:

```toml
excludes = ["/vendor/", "/node_modules/", "/target/"]

[function]
min_similarity = 0.85   # where true positives concentrate
min_nodes = 40          # ignore units smaller than this many syntax nodes

[vocab]
min_overlap = 0.55
min_vocabulary = 30
worsened_delta = 0.05   # how much an existing pair must grow to count as worse

[blocks]
min_tokens = 100        # normalized tokens, which are denser than source lines

[vocab.noise]
python = ["self", "cls", "args", "kwargs"]

[report]
# max_findings = 50     # absent means no cap; 0 means report nothing
```

An unknown key is an **error**, not a shrug. Someone who writes `min_similarty = 0.9` gets told,
rather than silently receiving the default while believing they configured something.

### Thresholds should rise with repository size

Both the function and vocabulary detectors compare every pair, so the number of *candidate* pairs
grows with the square of the codebase. A threshold that gives a handful of findings on a 20-file
project can give thousands on a 1,500-file one — not because that project is worse, but because it
has 5,000 times as many pairs to draw from.

Measured on a 445-file Python repository, raising `min_nodes` from 30 to 80 and `min_similarity`
from 0.85 to 0.90 took the scan from 59,264 function pairs to 724. On a 1,472-file polyglot
repository, raising `min_overlap` from 0.30 to 0.55 and `min_vocabulary` from 15 to 30 took the
vocabulary findings from 61,852 to 3,873.

The defaults above suit a small-to-medium repository. On a large one, raise them — and note that
this only affects the size of the *scan*. What a pull request actually sees is the delta, which is
bounded by what that change introduced.

---

## Why it never blocks a merge

A hard gate on a similarity heuristic gets worked around or switched off — and any real codebase has
duplication it has legitimately accepted, like two account types with parallel structure that are
genuinely different rules. So this reports, and leaves the judgement to you:

- **Extract the shared logic** into one place both call — the intended outcome for a real accidental
  duplicate; or
- **Leave it**, if the resemblance is coincidental or the two sides are genuinely different rules
  that happen to share a shape.

There is nothing to "accept" permanently and no allowlist to update. Once your change lands, its
duplication is part of the merge-base and will not be mentioned again.

---

## Design notes

**Zero is a value, not a fallback.** A missing input fails loudly rather than defaulting to
something plausible. A root that does not exist is an error, not an empty scan — because a scan that
silently examines nothing reports "no duplication" and looks exactly like a clean tree.

**Syntax errors are surfaced, never swallowed.** tree-sitter is error-tolerant and returns a partial
tree rather than refusing, which is useful but is also how a scan quietly degrades to finding
nothing. Files that failed to parse are counted and named in the report.

**Determinism.** Two scans of identical trees produce byte-identical reports, thread count
notwithstanding.

**Known limitation, stated plainly:** vocabulary pairs are keyed by file path, because a module has
no single normalized body to hash. **Renaming a file makes its vocabulary pairs look new.** Function
and block findings are unaffected.

---

## Reports as a contract

A scan writes JSON described in `src/report.rs`. Anything that can emit that shape participates in
the delta machinery — whatever language it is written in, whatever technique it uses. The format
carries a version, and a report announcing a version this build does not know is **refused** rather
than read optimistically: a delta computed from a misread report is silently wrong, which is worse
than an error.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: 100% line coverage per file, enforced in
CI by `scripts/coverage.sh`, and never lowered to go green. If a line cannot be reached by any test,
that is evidence the line should not exist.

`dupdelta` runs against its own pull requests.

## License

MIT — see [LICENSE](LICENSE).
