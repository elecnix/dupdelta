#!/usr/bin/env python3
"""Unit tests for the timing gate's drift check.

Run:  python3 scripts/test_timing_gate.py       (no third-party dependencies)

The gate is the only thing standing between a slow test and a merge, and the
defect it had was one it could not report: `check()` computed which tests the
baseline did not know about, printed the count, and exited 0. These tests pin
the exit code, not the wording, so rewording the message does not break them.

Nothing here runs the Rust suite -- `check()` takes a baseline and a cost
mapping as arguments, which is exactly the seam that makes it testable without a
compiler.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parent / "test-timing.py"

# Loading the gate by path would otherwise drop a scripts/__pycache__ into a
# tree that does not ignore one.
sys.dont_write_bytecode = True

_TIMING = None


def gate():
    """`scripts/test-timing.py`, imported by path on first use.

    Loaded lazily rather than at module scope, on purpose. A module-scope load
    runs before `unittest.main()` collects anything, so a gate that is missing,
    renamed, or raises while importing takes the interpreter down with a
    traceback and exit code 1 and reports *no tests at all*. In CI that reads
    as a failing test while naming nothing. Here the same breakage surfaces as
    a named failure, and every test that needs the gate still fails loudly
    rather than quietly passing against something that was never loaded --
    which is the failure mode the gate exists to prevent.
    """
    global _TIMING
    if _TIMING is None:
        spec = importlib.util.spec_from_file_location("test_timing", SCRIPT)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the gate from {SCRIPT}")
        _TIMING = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TIMING)
    return _TIMING


def baseline(tests: dict[str, float], **overrides) -> dict:
    """A baseline shaped exactly as `build_baseline` writes one.

    Costs are multiples of one unit, so they are written as unit multiples
    throughout: 0.5 means half the reference set's total time.
    """
    timing = gate()
    modules: dict[str, float] = {}
    for name, cost in tests.items():
        module = timing.module_of(name)
        modules[module] = modules.get(module, 0.0) + cost
    doc = {
        "version": timing.BASELINE_VERSION,
        "reference_modules": sorted(timing.DEFAULT_REFERENCE),
        "runs": timing.DEFAULT_RUNS,
        "tolerance": dict(timing.DEFAULT_TOLERANCE),
        "ungated_modules": sorted(timing.DEFAULT_UNGATED_MODULES),
        "min_gated_cost": timing.DEFAULT_MIN_GATED_COST,
        "total_cost": round(sum(tests.values()), 4),
        "gated_cost": round(timing.gated_total(tests, timing.DEFAULT_UNGATED_MODULES), 4),
        "modules": {m: round(c, 4) for m, c in sorted(modules.items())},
        "tests": dict(tests),
    }
    doc.update(overrides)
    return doc


def run_gate(base: dict, costs: dict[str, float]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gate().check(base, costs)
    return code, out.getvalue(), err.getvalue()


class TestTheGateLoads(unittest.TestCase):
    """The harness can reach the gate under test at all.

    Not a formality: it is the failure that a module-scope import turned into
    an unnamed traceback, given a name.
    """

    def test_the_gate_imports_and_exposes_the_seam_these_tests_drive(self):
        timing = gate()
        self.assertTrue(callable(timing.check), "check() is the seam every other test drives")
        self.assertEqual(timing.BASELINE_PATH.parent.name, "tests")
        self.assertEqual(timing.BASELINE_PATH.suffix, ".json")

    def test_loading_the_gate_does_not_write_a_pycache(self):
        gate()
        self.assertTrue(sys.dont_write_bytecode)
        self.assertFalse((SCRIPT.parent / "__pycache__").exists())


class GateCase(unittest.TestCase):
    """A scenario: a suite, a change to it, and what the gate must say.

    Every test in this file builds a cost mapping, runs the gate over it, and
    then reads the same three things -- an exit code, and what stderr says.
    That reading lives here once, so a test is left stating only the scenario
    it is about. Which is also what stops the scenarios reading as copies of
    one another: what used to be four repeated lines of ceremony around each
    mutation is now one call, and the mutation itself is all that varies.
    """

    #: The suite the scenarios start from. Costs are unit multiples and are
    #: chosen above the 0.02 floor, so nothing passes merely by being too small
    #: to time -- which is how a test escapes the gate a second way.
    SUITE: dict[str, float] = {}

    def suite_with(self, name: str, cost: float) -> dict[str, float]:
        """`SUITE` plus one more test in it, costing `cost` units."""
        return dict(self.SUITE, **{name: cost})

    def suite_without(self, name: str) -> dict[str, float]:
        """`SUITE` with one test gone."""
        return {k: v for k, v in self.SUITE.items() if k != name}

    def suite_renamed(self, old: str, new: str) -> dict[str, float]:
        """What a rename looks like to the gate: one name gone, another in."""
        costs = self.suite_without(old)
        costs[new] = self.SUITE[old]
        return costs

    def verdict(self, costs: dict[str, float], code: int, *in_err: str) -> None:
        """Run the gate over `costs`: it must exit `code`, with each of `in_err` in stderr.

        Expecting nothing in stderr also asserts stderr is empty, so a gate
        that passes while complaining still fails here.
        """
        got, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(got, code)
        if in_err:
            for text in in_err:
                self.assertIn(text, err)
        else:
            self.assertEqual(err, "", "a verdict of 0 must not be accompanied by a complaint")


class DriftIsFatal(GateCase):
    """The regression this file exists for."""

    SUITE = {
        "blocks::tests::a": 0.30,
        "blocks::tests::b": 0.20,
        "vocab::tests::a": 0.10,
        "cli::tests::a": 0.90,
    }

    def test_identical_suite_passes(self):
        self.verdict(dict(self.SUITE), 0)

    def test_a_test_the_baseline_does_not_record_is_fatal_wherever_it_hides(self):
        # One sentence, three ways in -- and the ways are the whole point, each
        # being a route by which a new test once passed uncompared. An ordinary
        # addition; one in `cli`, which is never gated per-test and so was
        # invisible in every dimension of the gate; and one below the cost
        # floor, which is never compared on its own though still counted by its
        # module. Same drift, same verdict, three nouns -- which is why these
        # are one table and not three tests that read alike.
        cases = (
            ("an ordinary addition", "blocks::tests::c", 0.05),
            ("in an ungated module", "cli::tests::brand_new", 0.40),
            ("below the cost floor", "blocks::tests::tiny", 0.001),
        )
        for label, name, cost in cases:
            with self.subTest(label):
                self.verdict(self.suite_with(name, cost), 1, name, "--update")

    def test_a_test_the_baseline_records_but_the_suite_lost_is_fatal(self):
        self.verdict(self.suite_without("blocks::tests::b"), 1, "blocks::tests::b", "no longer exist")

    def test_a_rename_names_both_sides(self):
        # A rename is the common case, and a contributor must not have to guess
        # which half of their diff is the problem.
        self.verdict(
            self.suite_renamed("blocks::tests::b", "blocks::tests::b_renamed"),
            1,
            "blocks::tests::b",
            "blocks::tests::b_renamed",
        )

    def test_drift_and_slowdown_are_both_reported(self):
        # A contributor who does both must not have to regenerate twice.
        costs = self.suite_with("blocks::tests::new", 0.05)
        costs["blocks::tests::a"] = 0.30 * 4
        self.verdict(costs, 1, "blocks::tests::new", "got slower relative to the reference set")

    def test_a_long_drift_is_truncated_but_counted(self):
        extra = {f"blocks::tests::bulk_{i}": 0.03 for i in range(60)}
        self.verdict(dict(self.SUITE, **extra), 1, "60 test(s)", "and 40 more")


class TheComparisonItselfIsUnchanged(GateCase):
    """The fix must not have loosened anything that already had teeth."""

    SUITE = {
        "blocks::tests::a": 0.30,
        "blocks::tests::b": 0.20,
        "vocab::tests::a": 0.10,
    }

    def test_the_per_test_tolerance_holds_from_both_sides(self):
        # One test, slowed. Over the limit it must fail; under it, noise must
        # not. These were two tests that looked identical because each spelled
        # out the same ceremony, but they are one boundary, and a gate that
        # passes a real regression is as broken as one that fails a noisy run.
        # Each case carries the exit code it expects, so both verdicts stay
        # visible and neither is the default.
        cases = (
            ("over the limit", 2.6, 1, ("got slower relative to the reference set",)),
            ("under the limit", 1.4, 0, ()),
        )
        for label, multiple, code, in_err in cases:
            with self.subTest(label):
                self.verdict(dict(self.SUITE, **{"blocks::tests::a": 0.30 * multiple}), code, *in_err)

    def test_a_slow_module_still_fails(self):
        # Every test in the module, rather than any one of them: the module
        # total is a separate gate with its own, looser limit.
        costs = dict(self.SUITE)
        costs["blocks::tests::a"] *= 2
        costs["blocks::tests::b"] *= 2
        self.verdict(costs, 1, "module blocks")

    def test_an_ungated_module_is_still_never_gated(self):
        # Pre-existing, deliberate: gating fork/exec-bound modules produces
        # failures that have nothing to do with the code. Recorded so the drift
        # fix cannot quietly become a licence to gate them.
        suite = dict(self.SUITE, **{"cli::tests::a": 0.90})
        costs = dict(suite, **{"cli::tests::a": 0.90 * 100})
        code, _, err = run_gate(baseline(suite), costs)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")


class TheCommittedBaseline(unittest.TestCase):
    """Checks against the artefact this repository actually ships."""

    def setUp(self):
        self.baseline = json.loads(gate().BASELINE_PATH.read_text())

    def test_it_loads_and_has_the_documented_shape(self):
        loaded = gate().load_baseline()
        self.assertEqual(loaded["version"], gate().BASELINE_VERSION)
        self.assertIn("tolerance", loaded)
        self.assertEqual(
            set(loaded["tolerance"]),
            set(gate().DEFAULT_TOLERANCE),
            "the committed tolerances are the script's defaults",
        )

    def test_every_recorded_cost_is_relative_not_a_duration(self):
        # A cost is a multiple of one unit. A value that looks like seconds
        # means an absolute duration was committed, which is the thing this
        # gate exists to avoid.
        unit = self.baseline["unit_seconds_when_recorded"]
        self.assertGreater(unit, 0.0)
        for name, cost in self.baseline["tests"].items():
            self.assertGreaterEqual(cost, 0.0, name)
            self.assertLess(cost, 100 * unit, f"{name} is too large to be a unit multiple")

    def test_the_baseline_has_not_lost_or_gained_a_test_since_main(self):
        # Cheap regression guard for the drift this branch fixes: the count must
        # match the number of `#[test]` attributes in the crate.
        #
        # Counted at attribute position, not as a substring. src/parse.rs
        # explains itself in a comment that mentions `#[test]` -- "runs each
        # `#[test]` on its own thread" -- and a substring count scores that
        # prose as a test, so this guard reports a phantom extra test and
        # sends you to regenerate a baseline that is already correct. A guard
        # that cries wolf on a correct baseline gets ignored, which is worse
        # than not having written it.
        declared = sum(
            len(re.findall(r"^\s*#\[test\]", path.read_text(), re.M))
            for path in (gate().REPO / "src").glob("*.rs")
        )
        self.assertEqual(
            len(self.baseline["tests"]),
            declared,
            "tests/timing_baseline.json does not match the number of #[test] "
            "attributes; regenerate it with scripts/test-timing.py --update",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)