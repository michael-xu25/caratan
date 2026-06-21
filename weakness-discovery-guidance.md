# Weakness Discovery — Guidance for Cara

*Your half of the next phase: once Michael's generated boards exist, you run the evals, compile transcripts, and take them to Claude to find the model's weaknesses. Those weaknesses become the envs we generate and train on — so this step decides what we build. Getting it rigorous matters more than getting it fast.*

---

## 0. Before anything: verify the transcripts are actually usable
Garbage transcripts → garbage analysis. Claude can only find real weaknesses if it can *follow the game*. Before you feed a single transcript to Claude, sanity-check that a transcript is:

- **Correct** — the logged sequence matches what actually happened in the game (moves, whose turn, resulting state). Spot-check one game by hand against the runner.
- **Readable** — a human can follow it: board state, whose turn, the legal options, what the model chose, and **its stated reasoning**, in order. If you can't follow a game start-to-finish and understand *why* the model did each thing, Claude can't either.
- **Interpretable** — each decision shows the *choices it had*, not just the choice it made. "Played robber on hex 7" is weak; "chose hex 7 from {7, 12, 3} while behind on VP" is analyzable. The legal action set + game context at each decision is what makes a mistake visible.

If any of these fail, fix the transcript format first — it's cheaper than re-running analysis on unreadable logs.

## 1. Whose transcripts, and which games
- **Analyze the model we intend to train** (E4B / Qwen-4B), not Claude. We're hunting *its* weaknesses so we can target them. Claude is the analyst, not the subject.
- **Weight toward losses, against a strong opponent** (AlphaBeta or Value). Mistakes show up where they're punished. Include some wins so you're not only seeing failure modes — but losses are where the signal is.
- **Enough games to see recurrence.** One game tells you nothing; a pattern across ~10–20 is signal. Batch them so Claude can compare across games, not just within one.

## 2. The rubric for Claude (paste this with the transcripts)

> You are analyzing Catan game transcripts from a small model we are about to RL-train. Your job is to find **systematic, recurring, targetable weaknesses** — specific decision types where the model reliably errs — that we can turn into verifiable training environments. This is not a "rate the play" task.
>
> **Discipline (read first):**
> - **Separate observation from judgment.** "The model did X" is reliable. "X was a mistake" is your judgment and may be wrong — you are not a strong Catan player either. Flag your confidence on every claimed error, and mark which need oracle/champion verification before we trust them.
> - **Corroborate across games.** Only surface a weakness you can show in **multiple distinct games**. A single bad move is noise. Cite specific instances (game id + turn + the decision) as evidence — no vibes.
> - **Verifiability is a hard filter.** A weakness is only useful to us if we can construct positions of that decision type and score the model's choice against a ground truth (an oracle like AlphaBeta, the champion's label, or exact EV). If you can't see how a weakness would become a verifiable, gradeable decision, say so — and deprioritize it.
> - **Don't invent Catan strategy as fact.** If you're unsure what optimal play is in a situation, flag it as needing expert/oracle confirmation rather than asserting it.
>
> **For each candidate weakness, give:**
> 1. **Name** — the decision type, precisely and narrowly (e.g. "robber placement when trailing in VP," not "bad robber play").
> 2. **Evidence** — 2+ instances (game id, turn, what it chose vs the better option, the context that made it a mistake).
> 3. **Frequency** — roughly how often this decision type came up and how often it was botched.
> 4. **Confidence + verification need** — how sure you are it's a real error, and what ground truth would confirm it (oracle / champion / exact).
> 5. **Targetability** — can we generate verifiable scenarios of this decision type? What's the ground-truth source? Is it in the model's reach (it sometimes gets it right) or near-hopeless (always fails)?
> 6. **Impact** — does the error actually cost games/VP, or is it cosmetic?
>
> **Then rank** the candidates by `impact × frequency × targetability × improvability`. We want the few weaknesses that are real, recurring, gradeable, and fixable — not the longest list.

## 3. What comes out, and where it goes
- Output = a **ranked list of candidate weaknesses** with the rubric fields filled.
- **Candidates are not yet envs.** Before any weakness becomes a training env, it gets verified — Michael (champion) or an AlphaBeta oracle confirms the flagged decisions are actually mistakes. Claude proposes; the oracle/champion disposes. This guards against training on a "weakness" Claude hallucinated.
- The top verified weaknesses → hand to Michael's env-generation side, where each becomes a scenario set with a ground-truth reward (same schema/contract as placement).

## The thing to watch
The failure mode here is Claude confidently naming weaknesses that are either (a) one-game flukes dressed up as patterns, or (b) real but unverifiable, or (c) plain wrong Catan takes stated with confidence. The rubric pushes against all three (corroboration, verifiability filter, confidence flags), but you're the backstop — if a "weakness" can't be turned into a gradeable decision with a ground truth, it doesn't enter the loop no matter how plausible it sounds.
