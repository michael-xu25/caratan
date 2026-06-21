# Weakness Discovery — Guidance for Cara

*Your half of the next phase: once Michael's generated boards exist, you run the evals, compile transcripts, and take them to Claude to find the model's weaknesses. Those weaknesses become the envs we generate and train on — so this step decides what we build. Getting it rigorous matters more than getting it fast.*

---

## 0. Transcript readiness (the interpretability gap is now closed — one check remains)
The transcripts now record what analysis needs: the **full legal option set** at each decision (not just a count), **VP context** (`my_vp`/`opp_vp`, so "while behind/ahead" is visible per decision), and the model's **reasoning** — in both JSON (full fidelity) and a scannable human log. So they're correct, readable, and interpretable by construction.

The one thing still on you: **spot-check one game by hand against the runner** before trusting any analysis — confirm the logged sequence == what actually happened. The action log is sourced directly from Catanatron's `action_records` so it's faithful by construction, but a single manual pass is cheap insurance against a silent logging bug poisoning every downstream weakness.

## 1. Whose transcripts, and which games
- **Analyze the model we intend to train** (E4B / Qwen-4B), not the analyst models. We're hunting *its* weaknesses so we can target them. Claude and GPT-5.5 are the analysts, not the subject.
- **Weight toward losses, against a strong opponent** (AlphaBeta or Value). Mistakes show up where they're punished. Include some wins so you're not only seeing failure modes — but losses are where the signal is.
- **Enough games to see recurrence.** One game tells you nothing; a pattern across ~10–20 is signal. Batch them so the analyst can compare across games, not just within one.

## 1b. Two analysts: Claude + GPT-5.5 (you have both keys)
We run **two independent analysts** over the same transcripts and use their *agreement* as a cheap corroboration layer. Decisions:
- **Models (locked):** Claude **Opus 4.8** and **`gpt-5.5` (non-Pro) at `reasoning_effort: xhigh`** as the standing config for every real analysis run. Different labs/training → their agreement is genuine independent signal, not an echo. (xhigh is slower per call, but this is an infrequent offline step, so pay for the quality.)
- **GPT-5.5 Pro is reserved as a tiebreaker only** — invoke it (or Michael/oracle) on the *specific* weaknesses where Claude and GPT-5.5 disagree. It's 6× the cost and takes minutes per call (use background mode), so it's not the workhorse.
- **Run them independently — do NOT show one model's output to the other.** Independence is the whole point; cross-contaminating them collapses two opinions into one.
- **Both are on Batch API**, which fits this offline batch analysis well.
- **How to use the agreement:**
  - Flagged by **both** → high-priority candidate (cross-model corroboration).
  - Flagged by **one** → lower-priority; scrutinize or send to the tiebreaker.
  - **Disagreement is signal, not noise** — it marks where the analysis is genuinely uncertain.
- **Caveat that still matters:** cross-model agreement is a *pre-filter*, not truth. Two confident LLMs can share the same wrong Catan intuition. Agreement raises priority; only the oracle/champion verification (§3) confers truth.

## 2. The rubric (paste this with the transcripts — run on EACH analyst independently)

> You are one of two independent analysts examining Catan game transcripts from a small model we are about to RL-train. Your job is to find **systematic, recurring, targetable weaknesses** — specific decision types where the model reliably errs — that we can turn into verifiable training environments. This is not a "rate the play" task. Analyze independently; you will not see the other analyst's output.
>
> **Discipline (read first):**
> - **Separate observation from judgment.** "The model did X" is reliable. "X was a mistake" is your judgment and may be wrong — you are not a strong Catan player either. Flag your confidence on every claimed error, and mark which need oracle/champion verification before we trust them.
> - **Corroborate across games.** Only surface a weakness you can show in **multiple distinct games**. A single bad move is noise. Cite specific instances (game id + turn + the decision) as evidence — no vibes.
> - **Verifiability is a hard filter.** A weakness is only useful to us if we can construct positions of that decision type and score the model's choice against a ground truth (an oracle like AlphaBeta, the champion's label, or exact EV). If you can't see how a weakness would become a verifiable, gradeable decision, say so — and deprioritize it.
> - **Don't invent Catan strategy as fact.** If you're unsure what optimal play is in a situation, flag it as needing expert/oracle confirmation rather than asserting it.
>
> **For each candidate weakness, give:**
> 1. **Key** — `(decision_type, game_phase, condition)` using the fixed `decision_type` enum (settlement, road, robber, discard, trade, dev_card, …). This is how the two analysts' lists get matched mechanically — be disciplined about it.
> 2. **Name** — the decision type in plain words, precisely and narrowly (e.g. "robber placement when trailing in VP," not "bad robber play").
> 3. **Evidence** — 2+ instances (game id, turn, what it chose vs the claimed-better option, the context that made it a mistake). The "chose X / should've been Y" must be explicit — the oracle check depends on it.
> 4. **Frequency** — roughly how often this decision type came up and how often it was botched.
> 5. **Confidence + verification need** — how sure you are it's a real error, and what ground truth would confirm it (oracle / champion / exact).
> 6. **Targetability** — can we generate verifiable scenarios of this decision type? What's the ground-truth source? Is it in the model's reach (it sometimes gets it right) or near-hopeless (always fails)?
> 7. **Impact** — does the error actually cost games/VP, or is it cosmetic?
>
> **Then rank** the candidates by `impact × frequency × targetability × improvability`. We want the few weaknesses that are real, recurring, gradeable, and fixable — not the longest list.

## 2b. Comparing the two analysts + flagging low-confidence grading
The hard part isn't running two models — it's that "are we confident in this grade?" is *several different questions*, and the obvious one (did they disagree?) is the least dangerous. Work through it in this order.

**First, make the two lists comparable at all.** Free-form names won't match — Claude's "over-extends roads early" and GPT-5.5's "prioritizes longest road over production" might be the same weakness or not. Don't resolve that with a third LLM call (that just adds more fuzzy judgment). Instead **force both analysts to emit each weakness against a shared key**: `(decision_type, game_phase, condition)` — e.g. `(robber_placement, midgame, trailing_vp)`, with `decision_type` constrained to a fixed enum (settlement, road, robber, discard, trade, dev_card, …). Now "do they agree" is a mechanical key-match, not a vibe. Add this as a required output field in the rubric.

**Then separate the two questions that "confidence" smears together:**
- **(A) Do the analysts agree?** — measurable from their outputs. Cheap.
- **(B) Are the analysts right?** — NOT measurable from their outputs. Needs an oracle/champion.

The dangerous trap is treating (A) as a proxy for (B). Two models with the *same* Catan blind spot produce high agreement on a wrong grade — agreement *feels* like correctness and isn't. So: use (A) to **triage**, use the oracle/champion for (B) to **confer truth**. Agreement never promotes a weakness to training on its own.

**What to flag as low-confidence — it's a vector, not "they disagreed":**
1. **Agreement** — both flag the same key / only one flags / one flags *and the other explicitly considered and dismissed it* (strongest disagreement). Same key but very different severity also counts as disagreement.
2. **Self-confidence** — each analyst's stated confidence. Both agreeing but both unsure is shaky *despite* agreement.
3. **Evidence weight** — 2 cited instances vs 15. Thin evidence is low-confidence even under agreement.
4. **Oracle corroboration** *(the one that catches shared blind spots)* — for each cited "chose X, should've been Y," have **AlphaBeta evaluate X vs Y on that exact state**. If the oracle says X wasn't actually worse, the *grade itself is wrong* no matter how confidently both models agreed. This is the only automatable check on question (B). Buildable cheaply because games are **seeded**: replay the seeded game (game_id + turn) to the cited decision → live Catanatron `State` → run AlphaBeta on the actual vs claimed-better action. No deserializer needed; you only do it for the handful of cited instances, not every decision.
5. **Verifiability** — if you can't construct a gradeable env for it, you can't grade it at all. Hard reject, not a soft flag.

**Triage rule:**
- **Fast-track to env:** same key, both confident, ≥ ~3 instances, oracle corroborates where checkable, clearly verifiable.
- **Tiebreak (Pro / oracle / Michael):** any disagreement, OR low self-confidence either side, OR thin evidence, OR oracle disagrees with the mistake-claim.
- **Reject:** unverifiable, OR oracle says it wasn't a mistake.

**Never skip:** even a fast-track weakness gets at least a spot oracle/champion check before training — because the dangerous failure mode (both models confidently wrong on a shared Catan blind spot) is *invisible* to anything computed from the two models' outputs. Only ground truth sees it. The oracle cross-check (#4) is the highest-value automation to build here; it's what makes "both agreed" safe to act on.

Per-weakness **confidence card** (build the merge around this):
```
key:          (decision_type, phase, condition)
flagged_by:   claude | gpt | both
claude_conf:  high/med/low      gpt_conf: high/med/low
n_instances:  N
oracle_check: agrees | disagrees | uncheckable   (AlphaBeta on cited states)
verifiable:   yes/no + ground-truth source
→ triage:     fast-track | tiebreak | reject
```

## 3. What comes out, and where it goes
- From each analyst: a **ranked list of candidate weaknesses** keyed as in §2b. Merge into one confidence card per key.
- **Candidates are not yet envs.** Before any weakness becomes a training env it's verified — AlphaBeta oracle (automatable, §2b #4) or Michael (champion, for fuzzy decisions) confirms the flagged decisions are actually mistakes. The analysts propose; the oracle/champion disposes.
- **Order into verification per the triage rule:** fast-track first, tiebreaks next, rejects dropped. The top *verified* weaknesses → hand to Michael's env-generation side, where each becomes a scenario set with a ground-truth reward (same schema/contract as placement).

## The thing to watch
The failure mode is an analyst confidently naming weaknesses that are either (a) one-game flukes dressed up as patterns, (b) real but unverifiable, or (c) plain wrong Catan takes stated with confidence. The rubric pushes against all three (corroboration, verifiability filter, confidence flags), and the second analyst catches some of (c) — but **two models can be confidently wrong together**, so cross-model agreement is a priority booster, not a truth stamp. You're the backstop: if a "weakness" can't be turned into a gradeable decision with a ground truth, it doesn't enter the loop no matter how plausible it sounds or how many models flagged it.
