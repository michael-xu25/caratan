# Baseline placement eval — BEFORE training

- Model: qwen3-4b-instruct-2507, queried via its **deployment** `qhzroqz3`
  (the bare model id `accounts/fireworks/models/...` 404s — not serverless).
- Boards: held-out (`grader_games`), reward mode: normalized, answer-only format.

**REAL baseline (10 boards, 100% valid answers): overall 0.837**
| opening | snake | mean reward |
|---|---|---|
| 1 | A | 0.768 |
| 2 | B | 0.811 |
| 3 | B | 0.847 |
| 4 | A | 0.921 |
| **overall** | | **0.837** |

Command (note: target the DEPLOYMENT, not the bare model id):
`python -m goldilocks_eval.placement_env eval --split grader_games --n 30 --model fireworks:accounts/brickedup25/deployments/qhzroqz3`
Re-run on the trained model's deployment for the AFTER row. (TODO: rerun at n=30
for the official figure; 10-board number above is the trustworthy interim.)

### ⚠️ The earlier "0.770" was FAKE
It was produced by querying the bare model id, which 404s; the eval then **silently
fell back** to "pick first legal node" (and unparseable answers fell back to *best*,
inflating). Fixed in `placement_env.py`: invalid/failed answers now score 0.0 and the
eval prints a **valid-answer rate** (0% ⇒ "numbers are MEANINGLESS"). Always eval via
a deployment id.

### Why Qwen3, not Qwen2.5
The project model is Qwen2.5-7B-Instruct, but **no Qwen2.5 model is RL-trainable on
Fireworks** (all are `rlLoraTunable=False`/`rlFullParameterTunable=False`; the RFT
API rejects them with a 400). Only the Qwen3 family is RL-trainable. We picked
`qwen3-4b-instruct-2507`: instruction-tuned, **non-thinking by design** (clean fit
for our answer-only format), RL LoRA+full tunable, free RFT (<16B).
For reference, the original Qwen2.5-7B deployment (`blpxetwj`) scored overall 0.742
(0.717/0.791/0.698/0.761) on the same boards — not the training base, kept only as
a data point.
