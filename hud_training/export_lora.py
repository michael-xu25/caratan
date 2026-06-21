"""Export a trained LoRA adapter from a Tinker checkpoint -> PEFT (for Fireworks).

Needs TINKER_API_KEY with access to the checkpoint's Tinker org. Our checkpoints
live on HUD's Tinker org (tinker://b548f490-...), so this only works with a key
HUD provides (or if brickedup25's own Tinker key is the one HUD trained under).

    .venv-tinker/bin/python export_lora.py --which build   # or: placement
Output: ./peft_<which>/adapter_config.json + adapter_model.safetensors
"""
import argparse
import os

from tinker_cookbook import weights

CKPTS = {
    "build": "tinker://b548f490-dcba-5b66-8f84-6d77a21ac372:train:11/sampler_weights/step-000104",
    "placement": "tinker://b548f490-dcba-5b66-8f84-6d77a21ac372:train:11/sampler_weights/step-000054",
}
BASE = "Qwen/Qwen3-8B"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=list(CKPTS), default="build")
    a = p.parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("set TINKER_API_KEY (from HUD / your Tinker account) first")
    raw = weights.download(tinker_path=CKPTS[a.which], output_dir=f"/tmp/lora_raw_{a.which}")
    out = f"peft_{a.which}"
    weights.build_lora_adapter(base_model=BASE, adapter_path=raw, output_path=out)
    print(f"wrote {out}/adapter_config.json + adapter_model.safetensors")
