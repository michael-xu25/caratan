# Build-doc alignment & work summary

This doc (a) summarizes the work done on the measurement half and (b) checks it
line-by-line against the original brief (`eval-infra-build-doc.md`). For *how the
harness works*, see `HARNESS.md`; this is the "what got built and does it match
the spec" view.

---

## Alignment with `eval-infra-build-doc.md`

Legend: ✅ done · 🟡 supported, needs input/convention · ⬜ not yet (out of scope / next)

### Start here
| Requirement | Status | Where |
|---|---|---|
| Pull Catanatron, run a bot game | ✅ | vendored `catanatron/`, runs via `Game(...).play()` |
| Baselines: Random / WeightedRandom / ValueFunction / AlphaBeta as opponents | ✅ | `harness/agents.py` (`random`, `weighted`, `value`, `alphabeta[:depth]`) |

### Match runner (core deliverable)
| Requirement | Status | Where |
|---|---|---|
| Two agents head-to-head, **1v1 throughout** | ✅ | `runner.py` `SEATS=(RED,BLUE)` |
| **Model-agnostic agent interface**, backend swappable by one config flag | ✅ | `agents.py` `make_agent(spec)`; one `decide()` contract |
| Backend swap Gemini ↔ Claude ↔ small/trained model | ✅ / 🟡 | Claude + **Fireworks** (trained models) wired; Gemini is a one-method stub |
| **Readable transcripts** — machine JSON via `GameEncoder` + human log (board, decisions + reasoning, outcome) | ✅ | `transcripts.py` → `<label>.json` (GameEncoder + `decisions[]`) and `<label>.log` |
| **Async runner**, parallelism ceiling = concurrent-LLM-call limit | ✅ | `runner.py` process pool; `--concurrency` knob |

### Fairness (makes or breaks the result)
| Requirement | Status | Where |
|---|---|---|
| **Seeded board + dice**, reproducible/replayable | ✅ | seed → board; dice now seed-controlled (see below). Byte-stable replay verified. |
| **Mirrored games** — every seed twice, seats swapped, compare across the pair | ✅ | `run_mirror_pair`, mirrored `run_batch`; board+dice identity proof in pair report |
| **Balanced dice deck** (shuffled deck of outcomes, colonist-style) | ✅ | `harness/dice.py` — **this was the gap; now fixed** (see "Determinism" below) |
| Hold out eval seeds never used in training | 🟡 | supported by convention: eval picks its own seed set; scenario schema carries `split: heldout`. Enforced at data-split time, not in harness code. |
| **Volume** — enough mirrored pairs to clear the noise band | ✅ | `--n` / `--seeds`; N chosen once per-game wall-time is known |

### Metric to support
| Requirement | Status | Where |
|---|---|---|
| **Primary: per-weakness before/after accuracy on held-out instances** vs ground truth | ✅ | `goldilocks_eval.scenario` (`evaluate` / `before_after`), re-exported via `harness.scenario`; smoke-tested in `scripts/scorer_smoke.py` |
| **Secondary: mirrored full-game head-to-head** win-rate | ✅ | `run_batch` → `BatchResult.win_rate_a` + summary table |
| Build for both; beware **catastrophic forgetting** (don't rely on head-to-head alone) | ✅ | both metrics exist independently; primary is the headline |

### Coming next (not blocking)
| Requirement | Status | Notes |
|---|---|---|
| Model ablation Gemini vs Claude; prep = model-agnostic interface | ✅ prep | interface is live (Claude + Fireworks prove the swap); Gemini = stub to fill |

### Not my job (Michael owns)
Synthetic env / data generation = `goldilocks_eval/`. Respected — I only fixed the
re-export seam and contributed shared backend changes (flagged for a PR to `main`).

---

## The one fairness gap I closed: dice determinism

The board was always a pure function of the seed (identical across a mirrored
pair). **Dice were not** — Catanatron rolls them from Python's *global* RNG as
the game advances, so once the two seatings make different moves the dice streams
desync (observed: same board, 66 vs 382 turns). That reintroduces the dice luck
mirroring is supposed to cancel.

**Fix** (`harness/dice.py`): deal dice from a per-seed deck drawn from a
*dedicated* RNG, decoupled from the global stream. Both games of a pair use the
same seed → roll #N is identical in both, regardless of dev-card/robber RNG. It's
"colonist-style" balanced too — each 36-roll cycle yields the exact 2–12
distribution (7 appears 6×), cutting variance. Proven, not asserted: the pair
report shows `board: IDENTICAL` **and** `dice: first N rolls IDENTICAL`.

Verified: same-seed decks are byte-identical; cross-run replay is stable
(same dice fingerprint, same winners).

---

## Changelog (this work)

**New**
- `harness/dice.py` — balanced, decision-independent dice deck (the determinism fix)
- `goldilocks_eval/agents/fireworks_backend.py` — backend for trained models (stdlib, OpenAI-compatible)
- `scripts/sample_run.sh` — pre-100 sample run with the fairness proof
- `scripts/scorer_smoke.py` — primary-metric integration test (no API key)
- `scripts/fireworks_api_key.sh` + `.secrets/fireworks_keys.json` (gitignored)

**Changed**
- `harness/scenario.py` — fixed the re-export seam Michael's refactor broke
- `harness/runner.py` — thread balanced dice; add `dice_fingerprint` / `dice_rolls`
- `harness/transcripts.py` — pair report proves dice identity (shared prefix)
- `harness/cli.py` — `--no-balanced-dice` toggle (demos before/after of the fix)
- `harness/agents.py`, `harness/backends.py` — register `fireworks` as an LLM backend
- `goldilocks_eval/agents/factory.py` — register the `fireworks` spec *(shared → PR to main)*
- `goldilocks_eval/agents/claude_backend.py` — default `max_tokens` 512→2048 *(shared → PR to main)*

**Verified live**
- Bot mirrored pair + batch; board & dice fairness proof
- Scenario scorer end-to-end (tiered reward, `before_after` delta) on frozen fixtures
- Claude in-game decision + scorer (same backend); legality fallback (dead backend finishes a game)
- Fireworks backend: key valid, transport reaches the API (Cloudflare UA gotcha fixed)

---

## Open items

1. **Fireworks model id** — the key is account/deployment-scoped (serves only models
   we deploy/fine-tune, not serverless). Pass the trained model via
   `fireworks:accounts/<acct>/models/<model-id>` once training produces one.
2. **Sync shared edits** — `claude_backend.py` (max_tokens) and the Fireworks
   backend + factory registration touch Michael's package and live only on `cara`;
   PR them to `main` so they don't silently fork/conflict.
3. **`max_tokens=512` truncation** (fixed): at 512, reasoning-mode answers were cut
   off before `<answer>` → false fall-backs that silently depressed accuracy. Now 2048.
4. **Rotate the Fireworks key** — it was pasted in plaintext.
