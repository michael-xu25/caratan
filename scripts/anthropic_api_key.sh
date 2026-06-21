#!/usr/bin/env bash
# Prints the primary Anthropic API key from the gitignored local store.
# Used as Claude Code's `apiKeyHelper` (so the key never sits in settings.json)
# and handy for `export ANTHROPIC_API_KEY="$(scripts/anthropic_api_key.sh)"`.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
store="$here/.secrets/anthropic_keys.json"
[ -f "$store" ] || { echo "no key store at $store" >&2; exit 1; }
python3 -c "import json,sys; print(json.load(open('$store'))['keys'][0])"
