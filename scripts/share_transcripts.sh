#!/usr/bin/env bash
# Share run transcripts with the team by default: stage + commit + push the run
# dir to the current branch so every run is visible without manual git.
#
#   scripts/share_transcripts.sh transcripts/selfplay
#   scripts/share_transcripts.sh                 # defaults to transcripts/
#
# Called automatically at the end of a run (e.g. selfplay_sample.py). Opt out of
# a single run with SHARE_TRANSCRIPTS=0. Scratch dirs (transcripts/_*) are
# gitignored and never shared — name a throwaway run with a leading underscore.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${SHARE_TRANSCRIPTS:-1}" = "0" ]; then
  echo "share: skipped (SHARE_TRANSCRIPTS=0)"; exit 0
fi

target="${1:-transcripts}"
git add -- "$target" 2>/dev/null || true
if git diff --cached --quiet -- "$target"; then
  echo "share: no new transcripts under $target"; exit 0
fi

n=$(git diff --cached --numstat -- "$target" | wc -l | tr -d ' ')
git commit -q -m "transcripts: share run output ($target)"
branch="$(git rev-parse --abbrev-ref HEAD)"
if git push -q origin "$branch" 2>/dev/null; then
  echo "share: committed + pushed $n transcript file(s) under $target -> $branch"
else
  echo "share: committed $n transcript file(s) (push failed — push $branch manually)"
fi
