#!/usr/bin/env bash
# Upload an exported PEFT LoRA adapter to Fireworks + deploy on-demand, so Cara's
# self-play harness can run fast (--b fireworks:...). Run AFTER export_lora.py
# produces peft_build/ or peft_placement/. Needs firectl (brew install fw-ai/firectl/firectl).
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ../.env; set +a            # FIREWORKS_API_KEY
WHICH="${1:-build}"                        # build | placement
ADAPTER="peft_${WHICH}"
MODEL_ID="catan-grpo-q8b-${WHICH}-fw"
ACCT="accounts/brickedup25/models/${MODEL_ID}"

firectl create model "$MODEL_ID" "$ADAPTER" --base-model accounts/fireworks/models/qwen3-8b
echo "waiting for model to be READY..."; firectl get model "$ACCT"
firectl create deployment "$ACCT"          # on-demand (LoRA can't be serverless)
echo "done -> Cara: harness --b fireworks:${ACCT}"
