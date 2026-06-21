#!/usr/bin/env bash
# Open the replay viewer on a transcript (or a directory of them).
#
#   scripts/view.sh transcripts/sample/batch/seed1_norm.json
#   scripts/view.sh transcripts/sample/batch        # builds all, opens the first
#
# Builds the .view.json if needed, serves the repo over http (the viewer fetches
# the data file), and opens the browser at the right URL.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
PORT="${PORT:-8000}"

target="${1:?usage: view.sh <transcript.json | dir>}"

# Build view data.
"$PY" scripts/build_viewer_data.py "$target"

# Resolve the .view.json to open.
if [ -d "$target" ]; then
  view="$(ls "$target"/*.view.json | head -1)"
else
  view="${target%.json}.view.json"
fi
url="http://localhost:${PORT}/viewer/?data=/${view}"

# Serve repo root if nothing is already on the port.
if ! curl -s "http://localhost:${PORT}/viewer/" >/dev/null 2>&1; then
  echo "starting http server on :${PORT} ..."
  "$PY" -m http.server "$PORT" >/dev/null 2>&1 &
  sleep 1
fi

echo "Opening $url"
command -v open >/dev/null && open "$url" || echo "Open this URL in your browser: $url"
