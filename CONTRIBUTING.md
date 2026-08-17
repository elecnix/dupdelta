# Contributing to dupdelta

## The bar

This is a tool whose entire value is that people trust its silence. If it says
"no new duplication," someone merges on that. A detector that quietly stops
detecting is worse than no detector at all, because it converts an unknown into
a false assurance.

So the standard here is not "does it work." It is:

> **When this is wrong, will anyone find out?**

Prefer the loud failure. A missing input should crash, not default to something
plausible. A rule that cannot be evaluated should say so, not silently pass.

## Non-negotiables

1. **100% line coverage, per file.** Not per project — per file. `scripts/coverage.sh`
   enforces it in CI and prints the exact uncovered line numbers when it fails.
2. **Never lower the bar to go green.** No `#[allow(dead_code)]` to hide an
   untested path, no threshold tweak, no exclusion added to the gate. If a line
   cannot be reached by any test, that is evidence the line should not exist —
   delete it. (`merge_adjacent` was deleted for exactly this reason; see the
   note in `Matcher::matching_blocks`.)
3. **Red, green, refactor.** Write the failing test first. A test written after
   the code tends to assert what the code does rather than what it should do.
4. **`cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, then
   `scripts/coverage.sh`.** All four, before every commit.

## The timing gate

`scripts/test-timing.py` guards against tests getting dramatically slower. It stores **no absolute
durations** — a duration recorded on a laptop means nothing on a shared CI runner. Instead a fixed
reference set of CPU-bound tests defines one *unit*, and every test's cost is recorded as a multiple
of it. A machine that is twice as slow runs the reference set twice as slowly too, so every ratio is
unchanged.

```sh
scripts/test-timing.py            # gate (what CI runs)
scripts/test-timing.py --report   # where the time actually goes
scripts/test-timing.py --update   # regenerate tests/timing_baseline.json
```

It needs nightly, because libtest's per-test timing output is still unstable.

Three things worth knowing before you meet it:

- **It catches order-of-magnitude regressions, not 10% ones.** The suite is a fifth of a second and
  most of it goes on spawning `git`. A tolerance tight enough to see 10% would fire on scheduler
  noise. The tolerances in the baseline come from measured run-to-run variance, not from optimism.
- **It does not ratchet, deliberately — unlike the coverage gate.** Coverage has no noise floor, so
  tightening it automatically is safe. Timing does have one, and a ratchet would lock in whichever
  run happened to be luckiest and then fail on every honest run afterwards. The baseline moves only
  when a human regenerates it.
- **Tests below the cost floor are not gated individually**, because sub-millisecond tests cannot be
  timed reliably — but they are still covered by their module's total, which is gated. The gate
  prints how many fell below the floor rather than quietly dropping them.

If you legitimately made something slower, regenerate the baseline **and say why in the commit
message**. A regenerated baseline with no explanation is indistinguishable from one regenerated to
make the build go green.

## Test style

Lean, single-responsibility tests. One behaviour per test, named as the sentence
it proves — `ratio_of_disjoint_sequences_is_zero`, not `test_ratio_2`.

### Assertions must not contain failure-only branches

This is the one non-obvious rule, and it falls straight out of the 100% bar.

```rust
// NO — the message is a branch that only runs when the test fails, so it is
// permanently uncovered.
assert!(got == want, "mismatch on {a:?}: {got} != {want}");

// NO — same problem, the push only happens on failure.
if got != want {
    failures.push(format!("{a:?}"));
}
assert_eq!(failures, Vec::<String>::new());

// YES — both sides always computed, compared whole. Full diagnostics from
// assert_eq!'s own output, no branch that only runs on failure.
let got: Vec<f64> = cases.iter().map(|c| round12(actual(c))).collect();
let want: Vec<f64> = cases.iter().map(|c| round12(expected(c))).collect();
assert_eq!(got, want);
```

For float comparisons, round to a fixed number of decimals (`round12`) so whole
vectors can be compared with `assert_eq!` instead of pairwise epsilon checks.

### Table-driven tests must not pass vacuously

If a property is asserted over a corpus, assert the corpus actually exercised it:

```rust
assert!(!abuts_in_both.is_empty());          // the corpus produced multi-block results
assert_eq!(abuts_in_both, vec![false; abuts_in_both.len()]);
```

### Derived impls need exercising too

`#[derive(Debug, Clone, PartialEq)]` generates code the gate counts. Either use
the derive in a test or do not derive it. A one-line `assert!(format!("{x:?}").contains("Name"))`
is enough and is honest: it proves the impl exists and does not panic.

### Test the engine, not a copy of it

A test that reimplements the production formula passes while production is
broken. Drive the real entry point and assert on its output. The one legitimate
exception is a deliberately naive *reference implementation* used to cross-check
an optimized one (`reference_ratio` in `similarity.rs`) — that is a second
opinion, not a copy, and it is written to be obviously correct rather than fast.

## Documentation style

Doc comments explain **why**, not what. `/// Returns the ratio` is noise;
"`quick_ratio` is order-blind, which is exactly why it is only ever a prune" is
the kind of thing that stops the next person from breaking an invariant they
did not know was there.

When you make a non-obvious choice — an algorithm that deviates from the
textbook, a bound that must hold in one direction — write down the argument for
it next to the code, and pin it with a test that names the argument.

## Commits

Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
One logical change per commit.
