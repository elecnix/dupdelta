#!/usr/bin/env bash
#
# Per-file 100% line-coverage gate.
#
# A whole-project percentage hides exactly the thing that matters: one file at
# 0% inside a project at 99% is invisible, and an untested file is where a
# detector silently stops detecting. So the gate is per file, and the bar is
# 100% — every line of every module, or the build fails.
#
# `src/main.rs` is excluded. It is a shim with no behaviour in it; everything
# testable lives in the library. If that file ever grows logic, the exclusion
# stops being honest and this line must go.
#
# Usage:
#   scripts/coverage.sh          # gate (what CI runs)
#   scripts/coverage.sh --html   # gate, and write an HTML report to target/
set -euo pipefail

# Percentages are formatted with `printf %f`, which honours the locale's
# decimal separator. Under e.g. fr_FR that is a comma and printf rejects the
# dot-formatted numbers jq emits. Pin the numeric locale.
export LC_ALL=C

THRESHOLD="${DUPDELTA_COVERAGE_THRESHOLD:-100}"
IGNORE_RE='src/main\.rs'
JSON_OUT="${JSON_OUT:-target/coverage.json}"

mkdir -p "$(dirname "$JSON_OUT")"

cargo llvm-cov --all-features --ignore-filename-regex "$IGNORE_RE" \
  --json --output-path "$JSON_OUT" >/dev/null

if [[ "${1:-}" == "--html" ]]; then
  cargo llvm-cov report --html --ignore-filename-regex "$IGNORE_RE" >/dev/null
  echo "HTML report: target/llvm-cov/html/index.html"
fi

# llvm-cov export format: .data[0].files[] with .filename and .summary.lines.
mapfile -t rows < <(
  jq -r --argjson t "$THRESHOLD" '
    .data[0].files[]
    | select(.summary.lines.percent < $t)
    | "\(.filename)\t\(.summary.lines.percent)\t\(.summary.lines.count - .summary.lines.covered)"
  ' "$JSON_OUT"
)

total=$(jq -r '.data[0].totals.lines.percent' "$JSON_OUT")
files=$(jq -r '.data[0].files | length' "$JSON_OUT")

if [[ ${#rows[@]} -eq 0 ]]; then
  printf 'coverage gate: PASS — %s file(s), all at %s%% line coverage (total %.2f%%)\n' \
    "$files" "$THRESHOLD" "$total"
  exit 0
fi

printf 'coverage gate: FAIL — %d file(s) below %s%% line coverage\n\n' "${#rows[@]}" "$THRESHOLD" >&2
printf '%-56s %8s %10s\n' "FILE" "LINES%" "UNCOVERED" >&2
for row in "${rows[@]}"; do
  IFS=$'\t' read -r file pct missing <<<"$row"
  printf '%-56s %7.2f%% %10s\n' "${file#"$PWD"/}" "$pct" "$missing" >&2
done

printf '\nUncovered lines, by file:\n' >&2
for row in "${rows[@]}"; do
  IFS=$'\t' read -r file _ _ <<<"$row"
  # Coverage segments are [line, col, count, has_count, is_region_entry, is_gap].
  # A segment that enters a region with a zero execution count starts a run of
  # code nothing ran.
  lines=$(jq -r --arg f "$file" '
    .data[0].files[]
    | select(.filename == $f)
    | .segments[]
    | select(.[3] == true and .[4] == true and .[2] == 0)
    | .[0]
  ' "$JSON_OUT" | sort -n -u | tr '\n' ' ')
  printf '  %s: %s\n' "${file#"$PWD"/}" "${lines:-(see --html report)}" >&2
done

printf '\nWrite a test that exercises those lines. Do not lower the bar.\n' >&2
exit 1
