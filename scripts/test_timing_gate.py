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
import sys
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parent / "test-timing.py"

# Loading the gate by path would otherwise drop a scripts/__pycache__ into a
# tree that does not ignore one.
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("test_timing", SCRIPT)
assert _spec and _spec.loader
timing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(timing)


def baseline(tests: dict[str, float], **overrides) -> dict:
    """A baseline shaped exactly as `build_baseline` writes one.

    Costs are multiples of one unit, so they are written as unit multiples
    throughout: 0.5 means half the reference set's total time.
    """
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
        code = timing.check(base, costs)
    return code, out.getvalue(), err.getvalue()


class DriftIsFatal(unittest.TestCase):
    """The regression this file exists for."""

    # Costs chosen above the 0.02 floor, so nothing here passes merely by being
    # too small to time -- which is how a test escapes the gate a second way.
    SUITE = {
        "blocks::tests::a": 0.30,
        "blocks::tests::b": 0.20,
        "vocab::tests::a": 0.10,
        "cli::tests::a": 0.90,
    }

    def test_identical_suite_passes(self):
        code, out, err = run_gate(baseline(self.SUITE), dict(self.SUITE))
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)
        self.assertEqual(err, "")

    def test_an_added_test_fails(self):
        costs = dict(self.SUITE, **{"blocks::tests::c": 0.05})
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("blocks::tests::c", err)
        self.assertIn("--update", err)

    def test_an_added_test_in_an_ungated_module_still_fails(self):
        # The exact shape of the drift this shipped with: the unrecorded test
        # was in `cli`, which is never gated per-test, so nothing else about it
        # was visible to the gate at all.
        costs = dict(self.SUITE, **{"cli::tests::brand_new": 0.40})
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("cli::tests::brand_new", err)

    def test_an_added_test_below_the_cost_floor_still_fails(self):
        # Below the floor a test is not compared on its own, but it is still a
        # test the baseline does not record, and the drift is the same.
        costs = dict(self.SUITE, **{"blocks::tests::tiny": 0.001})
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("blocks::tests::tiny", err)

    def test_a_removed_test_fails(self):
        costs = {k: v for k, v in self.SUITE.items() if k != "blocks::tests::b"}
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("blocks::tests::b", err)
        self.assertIn("no longer exist", err)

    def test_a_rename_names_both_sides(self):
        # A rename is the common case, and a contributor must not have to guess
        # which half of their diff is the problem.
        costs = {k: v for k, v in self.SUITE.items() if k != "blocks::tests::b"}
        costs["blocks::tests::b_renamed"] = 0.20
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("blocks::tests::b", err)
        self.assertIn("blocks::tests::b_renamed", err)

    def test_drift_and_slowdown_are_both_reported(self):
        # A contributor who does both must not have to regenerate twice.
        costs = dict(self.SUITE, **{"blocks::tests::new": 0.05})
        costs["blocks::tests::a"] = 0.30 * 4
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("blocks::tests::new", err)
        self.assertIn("got slower relative to the reference set", err)

    def test_a_long_drift_is_truncated_but_counted(self):
        extra = {f"blocks::tests::bulk_{i}": 0.03 for i in range(60)}
        costs = dict(self.SUITE, **extra)
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("60 test(s)", err)
        self.assertIn("and 40 more", err)


class TheComparisonItselfIsUnchanged(unittest.TestCase):
    """The fix must not have loosened anything that already had teeth."""

    SUITE = {
        "blocks::tests::a": 0.30,
        "blocks::tests::b": 0.20,
        "vocab::tests::a": 0.10,
    }

    def test_a_slow_test_still_fails(self):
        costs = dict(self.SUITE, **{"blocks::tests::a": 0.30 * 2.6})
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("got slower relative to the reference set", err)

    def test_a_slow_module_still_fails(self):
        costs = dict(self.SUITE)
        costs["blocks::tests::a"] *= 2
        costs["blocks::tests::b"] *= 2
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 1)
        self.assertIn("module blocks", err)

    def test_noise_within_tolerance_still_passes(self):
        costs = dict(self.SUITE, **{"blocks::tests::a": 0.30 * 1.4})
        code, _, err = run_gate(baseline(self.SUITE), costs)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")

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
        self.baseline = json.loads(timing.BASELINE_PATH.read_text())

    def test_it_loads_and_has_the_documented_shape(self):
        loaded = timing.load_baseline()
        self.assertEqual(loaded["version"], timing.BASELINE_VERSION)
        self.assertIn("tolerance", loaded)
        self.assertEqual(
            set(loaded["tolerance"]), set(timing.DEFAULT_TOLERANCE),
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
        declared = sum(
            path.read_text().count("#[test]")
            for path in (timing.REPO / "src").glob("*.rs")
        )
        self.assertEqual(
            len(self.baseline["tests"]), declared,
            "tests/timing_baseline.json does not match the number of #[test] "
            "attributes; regenerate it with scripts/test-timing.py --update",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)