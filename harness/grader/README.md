# harness/grader — dual-grader pipeline

Turns game transcripts into a **ranked list of failure modes** for env generation,
implementing the recommended design from `grading-rubric-proposal.md`.

```
transcripts ─▶ regret oracle (gate) ─▶ Claude + OpenAI categorize ─▶ reconcile ─▶ ranked failure modes
```

## The idea

The hard part of dual grading is merging disagreements. We sidestep most of it:
the **regret oracle** (Catanatron's value function) objectively answers *"was this
a mistake, and how bad?"*, so the two LLMs never vote on that — they only
**categorize why** into a fixed taxonomy. That shrinks the merge to categorical
labels, which are measurable.

## Modules

- `oracle.py` — replays a transcript (deterministic, recorded ActionRecords) and
  computes per-decision `regret = value(best legal) − value(chosen)`, in raw
  value-fn units and `regret_vp` (VP-equivalents = `regret / public_vps`). Gates
  decisions at a regret percentile (default 75th).
- `taxonomy.py` — the closed-set failure-mode labels (+ `other`/`none`). Both
  graders pick from this so labels are comparable.
- `prompts.py` — the shared grading prompt (board + decision + stated reasoning +
  **oracle context**) and a robust JSON verdict parser.
- `graders.py` — runs one LLM grader (`claude:…` / `openai:…`) into a normalized
  `Verdict`; optional self-consistency (majority of N samples).
- `reconcile.py` — merges two verdicts: **consensus → oracle-anchored tie-break →
  optional judge → unresolved**; averages numeric fields, keeps the spread and both
  raw verdicts. Reports **Cohen's κ** (chance-corrected) per batch.
- `pipeline.py` — orchestrates gate → dual-grade → reconcile → aggregate; ranks
  failure modes by total VP-regret (= ROI to fix).

## Usage

```bash
# preview gating only — no API calls, free:
python scripts/grade_transcripts.py transcripts/selfplay --dry-run

# full dual grade over a run:
export ANTHROPIC_API_KEY="$(scripts/anthropic_api_key.sh)"
export OPENAI_API_KEY="$(scripts/openai_api_key.sh)"
python scripts/grade_transcripts.py transcripts/selfplay \
    --grader-a claude:claude-opus-4-8 --grader-b openai:gpt-4o \
    --judge claude:claude-opus-4-8      # optional tie-breaker; off by default
```

Outputs `<run>/grading/findings.jsonl` (one per graded decision, both raw verdicts
kept) and `<run>/grading/report.json` (ranked failure modes + agreement/κ).

## Recommended defaults (and how to tune)

- **Gate** at the 75th regret percentile (`--gate-pct`) + 10% low-regret sample
  (`--low-regret-rate`) to catch "lucky-right" reasoning failures and audit the
  oracle. Raise `--gate-pct` to cut cost (these 400-cap games have ~300 decisions
  each, so the gate matters).
- **Merge**: consensus first, then oracle-anchored tie-break (free), then judge.
- **Report two numbers** when measuring before/after: consensus rate (precision)
  and union coverage (recall). κ audits the taxonomy — low-κ labels are fuzzy and
  should be merged/redefined.

## Notes / caveats

- The value fn weights VP at `3e14` ("win at all costs"), so VP-changing mistakes
  dominate raw regret. `regret_vp` makes it legible; finer per-decision-type
  normalization is a tunable ([DECIDE] in the proposal).
- No human gold labels exist yet (`data/examples` `gold_action` is empty), so the
  oracle is the objective anchor; add a calibration pass vs gold when labels land.
- Players are gemma/qwen and graders are claude/openai — no family overlap, so no
  self-preference bias. Keep it that way.
```
