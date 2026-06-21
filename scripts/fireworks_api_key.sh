#!/usr/bin/env bash
# Prints the Fireworks API key from the gitignored local store.
# Handy for `export FIREWORKS_API_KEY="$(scripts/fireworks_api_key.sh)"`.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
store="$here/.secrets/fireworks_keys.json"
[ -f "$store" ] || { echo "no key store at $store" >&2; exit 1; }
python3 -c "import json,sys; print(json.load(open('$store'))['keys'][0])"
