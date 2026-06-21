# Words vs. RL: which capability goes where

The dividing test for any capability is: **does it compress to a statable rule,
or is it un-enumerable judgment?**

- **Statable → teach with words (in the prompt).** Some knowledge compresses to
  a short, near-context-free rule a model can simply follow: the rules of the
  game, the legal action set, how to read the board state, the output format,
  and the small set of strategy heuristics that are genuinely simple and
  universal. Putting these in the prompt is faster, cheaper, and smarter than
  building an environment to teach them — there's no reason to spend RL on a
  principle you can state in a sentence and that holds in every situation.

- **Un-enumerable → teach with RL (in the weights).** Most real strategy does
  not compress. Whether to take the risky high-pip spot vs. the safe diverse one
  given what opponents took; when to trade away a scarce resource; when to switch
  from expansion to dev cards to longest road; when to block vs. develop — every
  one depends on the full board, the opponents, the phase, and the interaction of
  all of them. You can't write these as prompt rules: you'd need thousands of
  conditionals over combinations you can't enumerate, and the model still
  couldn't apply them. This is where RL earns its keep: it learns a general
  policy from reward across many situations, generalizes to cases you never
  described, and bakes the knowledge into the weights — durably better with no
  coaching in the prompt.

## Why the line matters for the demo's validity
Anything we teach with **words**, the model is just *following instructions* —
improvement from that proves nothing about the loop. Anything learned via **RL**
is *genuinely learned strategy* — which is exactly what we're demonstrating. So
mechanics and the few statable heuristics go in the prompt (**identically for the
baseline and the trained model**); all un-enumerable strategy is taught only
through the RL reward. The improvement we show is then real, generalizing,
weight-level learning — not hints we fed the model.

## Practical rule
For each thing we want the model to know, ask: **"statable rule or un-enumerable
judgment?"**
- Statable → put it in the prompt; don't waste RL on it.
- Un-enumerable → that's the env's job, and that's where the value is, because the
  hard-and-valuable part of any capability is precisely the part that doesn't fit
  in a prompt.

(If strategy compressed to prompts, RL environments wouldn't be a field, and our
company wouldn't need to exist.)

## How this maps onto our code
- **Prompt (statable; identical for base & trained)** — `goldilocks_eval/prompt.py`
  `CATAN_RULES` + the live action glossary + `render_state`/`render_actions`
  (board, pips, ports, pieces, legal moves) + output format. Reviewable in
  `viewer/rules.html`.
- **Reward (un-enumerable judgment)** — the championship rubric / scenario
  `gold_action` labels (`goldilocks_eval/scenario.py`, calibration, training).
  Never shown to the model in the prompt.
- **The litmus we keep applying:** "this corner borders tiles 6, 9, 4 (pips
  5, 4, 3)" is statable → prompt. "prefer high-pip, resource-diverse corners" is
  un-enumerable judgment (depends on opponents, ports, phase) → reward only.
