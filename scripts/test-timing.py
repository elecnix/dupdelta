#!/usr/bin/env python3
"""
Relative test-timing gate.

Absolute test durations are worthless as a committed baseline: the number
depends on the machine, and CI runners are shared, throttled, and replaced. A
baseline recorded on a laptop fails on a runner for reasons that have nothing
to do with the code.

So nothing absolute is stored. A fixed **reference set** of tests defines one
*unit* of time, and every other test's cost is recorded as a multiple of that
unit. If the machine is twice as slow, the reference set is twice as slow too,
and every multiplier is unchanged.

    unit          = total time of the reference-set tests
    cost(test)    = time(test) / unit

## Choosing the reference set

It has to be CPU-bound, deterministic, free of subprocesses and filesystem
work, and *big enough to measure*. Both halves matter, and they pull in
opposite directions:

  - Measured over three runs, this suite's smallest modules had the **worst**
    relative variance -- `token` totals 0.3 ms and swung 2.6x run to run,
    purely because it is too small to time. Those cannot anchor anything.
  - The largest modules (`cli`, `git`) spend their time spawning `git`
    processes, so they measure process-creation speed rather than CPU speed.

The default reference set -- blocks, extract, lang, normalize, vocab -- is the
middle: real parsing and token work, no I/O, no subprocesses, ~93 ms total, and
a measured run-to-run spread of 1.04x as a composite.

## What this gate can and cannot catch

It catches **order-of-magnitude** regressions: an accidental O(n^3), a sleep, a
detector that stopped pruning. It does not catch 10% ones, and it does not try
to: the suite is a fifth of a second, dominated by process spawning, and a
tolerance tight enough to see 10% would fire constantly on scheduler noise.
Tolerances below are set from measured variance, not from optimism.

## A regression *inside* the reference set

The reference set defines the unit, so if it slows down uniformly the unit
grows and everything else looks cheaper -- the regression hides, and the totals
move the wrong way. This is real: slowing one `normalize` test by 150 ms drops
the gated total from 1.80 to 1.30 units.

What catches it is the per-module layer. The five reference modules' costs sum
to exactly 1.0 by construction, so one of them slowing down raises *its share*
of the unit, and its module gate fires (`module normalize`, in that experiment)
even as the total falls. That is why this gate compares three things -- total,
module, test -- rather than just a total.

The residue: a slowdown hitting all five reference modules by the same factor
is invisible. That would mean tree-sitter parsing itself got uniformly slower,
which no change in this repository can cause on its own.

## Why this does not ratchet, unlike the coverage gate

`scripts/coverage.sh` tightens automatically, because coverage only moves when
someone writes or deletes a test -- it has no noise floor. Timing does. A
ratchet here would lock in whichever run happened to be luckiest and then fail
on every subsequent honest run. The baseline moves only when a human
regenerates it.

Usage:
    scripts/test-timing.py --update    # record tests/timing_baseline.json
    scripts/test-timing.py             # gate (what CI runs)
    scripts/test-timing.py --report    # show the costliest tests
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO / "tests" / "timing_baseline.json"

BASELINE_VERSION = 1

# Modules whose combined time is one unit. See the module docstring for why
# these and not the fastest or the slowest.
DEFAULT_REFERENCE = ["blocks", "extract", "lang", "normalize", "vocab"]

DEFAULT_TOLERANCE = {
    # Per test. Loosest, because a single test is the noisiest thing measured.
    "test": 2.5,
    # Per module. Aggregating dozens of tests cancels much of the noise.
    "module": 1.6,
    # Whole suite.
    "total": 1.5,
}

# Modules measured and reported, but never gated: their cost is dominated by
# spawning `git`, not by anything in this codebase.
#
# Two reasons, and the second is the decisive one.
#
# 1. Normalizing against a reference set removes differences in machine
#    *speed*. It cannot remove differences in machine *shape* -- the ratio
#    between forking a process and doing arithmetic. Measured, not assumed: the
#    committed baseline was recorded on a developer laptop, and the same commit
#    on a GitHub ubuntu-latest runner reported the whole suite at 0.75x it,
#    entirely because these two modules land differently against a CPU-defined
#    unit. A runner where forking is comparatively expensive pushes it the
#    other way, and the gate would fail with nothing about the code changed.
#
# 2. Gating them would not buy anything anyway. These tests spend their time in
#    fork/exec, so a real regression in our code would be a rounding error
#    against that, invisible under the noise. A gate that cannot see the thing
#    it nominally watches is worse than no gate, because people trust it.
#
# Note this is emphatically NOT an argument for mocking `git`. `git.rs` exists
# to shell out; a test against a fake would pass while the real thing was
# broken, which is the failure mode CONTRIBUTING.md calls out by name. The
# tests keep spawning real processes. We just stop pretending the clock is
# measuring us when it does.
DEFAULT_UNGATED_MODULES = ["cli", "git"]

# Tests cheaper than this many units are recorded but not gated: below roughly
# a millisecond, run-to-run noise exceeds any regression worth naming. The
# count of ungated tests is always printed -- a cap that hides what it dropped
# is exactly the failure this project exists to catch elsewhere.
DEFAULT_MIN_GATED_COST = 0.02

DEFAULT_RUNS = 3


def run_suite(toolchain: str, target_dir: str | None) -> dict[str, float]:
    """Run the test suite once and return per-test wall time in seconds.

    Requires a nightly toolchain: libtest's machine-readable timing output is
    still unstable. That is why the CI job pins nightly rather than stable.
    """
    env = dict(os.environ)
    if target_dir:
        env["CARGO_TARGET_DIR"] = target_dir
    proc = subprocess.run(
        [
            "cargo", f"+{toolchain}", "test", "--lib", "--",
            "-Z", "unstable-options", "--format", "json", "--report-time",
        ],
        cwd=REPO, env=env, capture_output=True, text=True,
    )

    times: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "test" and event.get("event") == "ok":
            times[event["name"]] = float(event["exec_time"])

    if not times:
        sys.exit(
            "test-timing: the suite produced no timing events.\n"
            f"cargo exited {proc.returncode}. A gate that silently measures nothing "
            "would pass forever, so this is fatal.\n\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    if proc.returncode != 0:
        sys.exit(f"test-timing: the test suite failed (exit {proc.returncode}); timings are meaningless.")
    return times


def measure(runs: int, toolchain: str, target_dir: str | None) -> dict[str, float]:
    """Time the suite `runs` times and keep each test's fastest observation.

    The minimum, not the mean: a test cannot run faster than its true cost, so
    every deviation upward is contention. The minimum is the standard robust
    estimator for timing and it converges far faster than averaging.
    """
    best: dict[str, float] = {}
    for index in range(runs):
        print(f"  run {index + 1}/{runs}", file=sys.stderr)
        for name, seconds in run_suite(toolchain, target_dir).items():
            if name not in best or seconds < best[name]:
                best[name] = seconds
    return best


def module_of(test_name: str) -> str:
    return test_name.split("::", 1)[0]


def unit_seconds(times: dict[str, float], reference: list[str]) -> float:
    """Total seconds spent in the reference set: one unit."""
    members = [t for name, t in times.items() if module_of(name) in reference]
    if not members:
        sys.exit(
            "test-timing: the reference set matched no tests.\n"
            f"Reference modules: {reference}\n"
            "Without a unit every cost is undefined, so this is fatal rather "
            "than a silent fallback to absolute times."
        )
    total = sum(members)
    if total <= 0.0:
        sys.exit("test-timing: the reference set measured as zero seconds; the clock is unusable.")
    return total


def costs_of(times: dict[str, float], unit: float) -> dict[str, float]:
    return {name: seconds / unit for name, seconds in times.items()}


def gated_total(costs: dict[str, float], ungated_modules: list[str]) -> float:
    """Total cost of everything this gate actually watches.

    Excludes the process-bound modules, so the number reflects work this
    codebase controls rather than the host's fork/exec speed.
    """
    return sum(c for name, c in costs.items() if module_of(name) not in ungated_modules)


def module_costs(costs: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = collections.defaultdict(float)
    for name, cost in costs.items():
        totals[module_of(name)] += cost
    return dict(totals)


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        sys.exit(f"test-timing: no baseline at {BASELINE_PATH}. Create one with --update.")
    baseline = json.loads(BASELINE_PATH.read_text())
    found = baseline.get("version")
    if found != BASELINE_VERSION:
        sys.exit(
            f"test-timing: baseline format version {found} is not readable by this script, "
            f"which understands {BASELINE_VERSION}. Regenerate it with --update."
        )
    return baseline


def build_baseline(costs: dict[str, float], unit: float, args: argparse.Namespace) -> dict:
    return {
        "version": BASELINE_VERSION,
        "comment": (
            "Relative test costs. One unit is the combined time of the reference "
            "modules; every value here is a multiple of that unit, never a duration. "
            "Regenerate with scripts/test-timing.py --update."
        ),
        "reference_modules": sorted(args.reference),
        "runs": args.runs,
        "tolerance": DEFAULT_TOLERANCE,
        "ungated_modules": sorted(DEFAULT_UNGATED_MODULES),
        "min_gated_cost": args.min_gated_cost,
        # Recorded for humans only -- never used in a comparison. It says how
        # fast the machine that produced this file was, which is useful context
        # and meaningless as a threshold.
        "unit_seconds_when_recorded": round(unit, 6),
        "total_cost": round(sum(costs.values()), 4),
        "gated_cost": round(gated_total(costs, DEFAULT_UNGATED_MODULES), 4),
        "modules": {m: round(c, 4) for m, c in sorted(module_costs(costs).items())},
        "tests": {name: round(cost, 4) for name, cost in sorted(costs.items())},
    }


def check(baseline: dict, costs: dict[str, float]) -> int:
    tolerance = baseline["tolerance"]
    floor = baseline["min_gated_cost"]
    ungated_modules = baseline.get("ungated_modules", [])
    measured_modules = module_costs(costs)
    measured_total = sum(costs.values())
    measured_gated = gated_total(costs, ungated_modules)

    failures: list[str] = []

    def compare(label: str, kind: str, was: float, now: float) -> None:
        if now > was * tolerance[kind]:
            failures.append(
                f"  {label}\n"
                f"      baseline {was:.4f} units, now {now:.4f} units "
                f"({now / was:.2f}x, limit {tolerance[kind]:.2f}x)"
            )

    compare("gated suite", "total", baseline["gated_cost"], measured_gated)

    for module, was in sorted(baseline["modules"].items()):
        if module in ungated_modules or module not in measured_modules:
            continue
        compare(f"module {module}", "module", was, measured_modules[module])

    gated = below_floor = skipped = 0
    for name, was in sorted(baseline["tests"].items()):
        if module_of(name) in ungated_modules:
            skipped += 1
            continue
        if was < floor:
            below_floor += 1
            continue
        if name in costs:
            gated += 1
            compare(f"test {name}", "test", was, costs[name])

    added = sorted(set(costs) - set(baseline["tests"]))
    removed = sorted(set(baseline["tests"]) - set(costs))

    print(
        f"gated suite: {measured_gated:.4f} units (baseline {baseline['gated_cost']:.4f})\n"
        f"whole suite: {measured_total:.4f} units (baseline {baseline['total_cost']:.4f}) "
        "-- reported, not gated"
    )
    print(
        f"gated individually: {gated} test(s). "
        f"Below the {floor}-unit floor: {below_floor} test(s) -- too small to time reliably, "
        "but still covered collectively by their module's total, which is gated. "
        f"In process-bound modules {ungated_modules}, measured but never gated: {skipped} test(s)."
    )
    if added:
        print(f"new since the baseline, not gated: {len(added)} test(s) -- regenerate with --update")
    if removed:
        print(f"in the baseline but no longer present: {len(removed)} test(s) -- regenerate with --update")

    if failures:
        print("\ntest-timing: FAIL -- these got slower relative to the reference set\n", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        print(
            "\nCosts are relative, so this is not a slow machine: something here now takes "
            "more work per unit of reference work than it did.\nIf the change is intended, "
            "regenerate with scripts/test-timing.py --update and say why in the commit.",
            file=sys.stderr,
        )
        return 1

    print("\ntest-timing: PASS")
    return 0


def report(costs: dict[str, float], unit: float, top: int) -> None:
    print(f"one unit = {unit:.4f}s on this machine\n")
    print(f"{'module':<14}{'units':>10}{'share':>9}")
    total = sum(costs.values())
    for module, cost in sorted(module_costs(costs).items(), key=lambda kv: -kv[1]):
        print(f"{module:<14}{cost:>10.3f}{cost / total * 100:>8.1f}%")
    print(f"\ntop {top} tests by cost:")
    for name, cost in sorted(costs.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {cost:>8.3f}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="record a new baseline")
    parser.add_argument("--report", action="store_true", help="print the costliest modules and tests")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=f"timed runs; the fastest wins (default: the baseline's own count, else {DEFAULT_RUNS})",
    )
    parser.add_argument("--toolchain", default="nightly", help="toolchain providing --report-time")
    parser.add_argument("--target-dir", default=None, help="CARGO_TARGET_DIR for the timed runs")
    parser.add_argument("--reference", nargs="+", default=DEFAULT_REFERENCE, help="reference-set modules")
    parser.add_argument("--min-gated-cost", type=float, default=DEFAULT_MIN_GATED_COST)
    args = parser.parse_args()

    reference = args.reference
    runs = args.runs if args.runs is not None else DEFAULT_RUNS
    if not args.update and BASELINE_PATH.exists():
        stored = load_baseline()
        # Gate against the reference set the baseline was built with, not
        # whatever the default happens to be today: changing the reference set
        # silently rescales every cost.
        reference = stored.get("reference_modules") or reference
        # And use the same number of runs. `measure` keeps each test's fastest
        # observation, which converges downward as runs increase -- so a
        # baseline recorded over five runs is systematically faster than a
        # check over three, and every cost would drift upward for a reason
        # that has nothing to do with the code. Matching the counts removes
        # the bias; an explicit --runs still overrides, for experimenting.
        if args.runs is None:
            runs = stored.get("runs", DEFAULT_RUNS)
    args.runs = runs

    print(f"timing {runs} run(s) on {args.toolchain}...", file=sys.stderr)
    times = measure(runs, args.toolchain, args.target_dir)
    unit = unit_seconds(times, reference)
    costs = costs_of(times, unit)

    if args.report:
        report(costs, unit, top=15)
        return 0

    if args.update:
        args.reference = reference
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(build_baseline(costs, unit, args), indent=2) + "\n")
        print(f"wrote {BASELINE_PATH.relative_to(REPO)} ({len(costs)} tests, unit = {unit:.4f}s)")
        return 0

    return check(load_baseline(), costs)


if __name__ == "__main__":
    sys.exit(main())
