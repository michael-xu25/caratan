# Env Generation — Handoff Contract (for Michael)

The loop: **Cara's grader finds weaknesses → you generate envs targeting them → Cara
measures the fix.** This doc is the seam where your half meets hers. Two rules make the
loop close; everything else is detail.

> **Rule 1 — same vocab, both sides.** Tag every generated env using ONLY the criterion
> IDs and state tags in `taxonomy.py`. A weakness you generate for and a weakness Cara
> measures must be the *same string*. Off-vocab = silently broken loop.
>
> **Rule 2 — every env needs a known correct answer.** A scenario with no ground-truth
> label can't be trained on and can't be scored. No label, no env.

---

## What flows to you: generation targets

Cara's aggregator emits a ranked table of discovered weaknesses. Each row is a target:

```
decision_type   criterion        tag             fail%   n
build_spend      timing           behind           75%   143
placement        pip_coverage     contested        61%    88
trade            opponent_benefit near_discard      57%    64
```

Each row = `(decision_type, criterion, tag)`. Your job: generate scenarios that **stress
that criterion in that state**, with a known correct move. Top rows first (highest fail%,
enough `n` to be real — the table is already sorted that way, Wilson-ranked so small-n
flukes don't reach the top).

---

## What you emit: the env schema

One JSON object per generated scenario. Cara's scorer reads exactly this shape.

```json
{
  "instance_id": "string, unique",

  "target_weakness": {
    "decision_type": "build_spend",
    "criterion": "timing",
    "tag": "behind"
  },

  "decision_type": "build_spend",
  "state_tags": ["mid", "behind", "robber_threat"],

  "game_state": { "...": "Catanatron-serializable position (board, buildings, hands, etc.)" },
  "legal_options": ["...the actions actually available at this decision..."],

  "ground_truth": {
    "correct": "<the right option, from legal_options>",
    "acceptable": ["<other defensible options, may be empty>"],
    "label_source": "michael | solver | judge"
  },

  "split": "train | heldout"
}
```

### Field rules

- **`target_weakness.tag` must appear in `state_tags`.** The scenario has to actually be
  in the state it claims to target. (Cara's loader can assert this.)
- **`game_state` + `legal_options`** must be loadable into Catanatron so the model can be
  asked to `decide` on this exact position. If you generate from Michael's real games,
  the position is the seed; synthetic variations perturb it.
- **`ground_truth.correct` is required.** This is the move the env trains toward and
  scores against. Three label sources, in descending trust:
  - `michael` — your actual move in a real game. Highest trust, limited supply.
  - `solver` — a strong bot's move on the position (e.g. AlphaBeta at depth). Scales
    infinitely, lower ceiling. Good for bulk.
  - `judge` — LLM-as-judge pick. Scales, but soft — use for graded dimensions, not as
    sole ground truth on a sharp decision.
  Record which one per env so we can weight trust later.
- **`acceptable` is load-bearing, not optional.** It lets the scorer give *partial credit*
  (correct = full, acceptable = partial, else = fail). Partial credit creates reward
  *variance* within a group, which is exactly what GRPO needs to learn — an all-or-nothing
  binary reward where most samples score 0 gives near-zero gradient. Populate it whenever a
  position has more than one defensible move.
- **`split`** — `train` envs feed training; `heldout` envs are measured before/after and
  **must be different instances than train** (same weakness, fresh positions). If train and
  held-out overlap, the before/after number measures memorization, not skill, and the demo
  claim collapses.

---

## How your envs get scored (so you know what "good" looks like downstream)

For each held-out env: Cara runs `model.decide(game_state, legal_options)` → gets a pick →
compares to `ground_truth` → pass/fail (or partial) → that becomes a verdict on
`target_weakness.criterion` in `target_weakness.tag` → into the same aggregator →
before/after fail-rate. So a clean env is one where:

1. the position genuinely exercises the target criterion,
2. the correct move is unambiguous enough to score,
3. it's tagged from the frozen vocab,
4. train and held-out instances are disjoint.

---

## Goldilocks difficulty

Aim for scenarios where the *current* model gets it right sometimes but not reliably —
roughly 1–3 of 8 sampled attempts correct. Too easy (always right) = no gradient, nothing
to learn. Too hard (never right) = no gradient either, and no signal that training helped.
The fail-rate from discovery is a guide: a 75%-fail weakness is already near the right
band; generate around that difficulty, don't make them impossible.

---

## Suggested volume per target

Per weakness row you target:
- enough `train` instances to actually move the policy (more is better; diverse positions
  within the weakness, not near-duplicates — duplicates invite memorization / reward hacking),
- **≥ ~15–20 `heldout` instances** so the before/after fail-rate clears the noise floor
  (the aggregator won't rank a bucket below that count).

---

## The one-line contract

> Tag from `taxonomy.py`, give every env a real correct answer with its source, keep
> train and held-out disjoint, and aim for Goldilocks difficulty. Do that and your envs
> drop straight into Cara's scorer and the loop closes.

Files you need from Cara's side: **`taxonomy.py`** (the vocab — import it, validate against
it) and the **generation-targets table** (the ranked rows, produced by `aggregator.py`).
