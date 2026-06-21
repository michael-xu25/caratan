#!/usr/bin/env bash
# Start the "play vs the model" server with the live Fireworks key + deployment.
# Deployment ids rotate; this reads FIREWORKS_MODEL from .env so it never goes stale.
#   ./scripts/play.sh            # opponent = fireworks:$FIREWORKS_MODEL from .env
#   ./scripts/play.sh value      # opponent = value bot (no API key)
set -euo pipefail
cd "$(dirname "$0")/.."

# Key: prefer the keychain helper, fall back to whatever .env / env already has.
if [ -x scripts/fireworks_api_key.sh ]; then
  export FIREWORKS_API_KEY="$(scripts/fireworks_api_key.sh)"
fi
# Model (FIREWORKS_MODEL) + any other run config.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

PORT="${PORT:-8000}"
pkill -f play_server.py 2>/dev/null || true
sleep 1
if [ "${1:-}" != "" ]; then
  exec .venv/bin/python scripts/play_server.py --model "$1" --port "$PORT"
else
  exec .venv/bin/python scripts/play_server.py --port "$PORT"   # default: fireworks:$FIREWORKS_MODEL
fi
