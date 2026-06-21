#!/usr/bin/env bash
# Sample run — a small, fast, FAIR mirrored batch to validate the harness before
# committing to the full N=100. Uses bots (no API key needed) so it's free and
# instant; swap --a/--b for LLM specs (e.g. claude:claude-haiku-4-5) once wired.
#
# What it demonstrates:
#   1. A mirrored batch (every seed played twice, seats swapped).
#   2. The fairness proof on one pair: same board AND same dice (shared deck),
#      so the only thing that changes between the two games is who sits where.
#
# Usage:
#   scripts/sample_run.sh                 # defaults: value vs weighted, 5 seeds
#   scripts/sample_run.sh value random 8  # A, B, number of seeds
set -euo pipefail
cd "$(dirname "$0")/.."

A="${1:-value}"
B="${2:-weighted}"
N="${3:-5}"
PY="${PYTHON:-.venv/bin/python}"
RUN_DIR="transcripts/sample"

echo "================================================================"
echo " SAMPLE RUN  —  A=$A  B=$B  seeds=$N  (mirrored, seeded i.i.d. dice)"
echo "================================================================"
echo
echo "### 1/2  Fairness pair on seed 1 (board + dice identity proof)"
"$PY" -m harness.cli --a "$A" --b "$B" --pair 1 --run-dir "$RUN_DIR/pair"
echo
echo "### 2/2  Mirrored batch over $N seeds"
"$PY" -m harness.cli --a "$A" --b "$B" --n "$N" --run-dir "$RUN_DIR/batch"
echo
echo "Done. Transcripts under $RUN_DIR/ . Scale up with: --n 100 --concurrency 8"

# Share the run by default (opt out: SHARE_TRANSCRIPTS=0).
"$(dirname "$0")/share_transcripts.sh" "$RUN_DIR" || true
