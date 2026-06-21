#!/usr/bin/env python3
"""Generate a review UI for the 1v1 Catan rules the model is given.

Pulls the ACTUAL prompt strings from the code (goldilocks_eval.prompt /
goldilocks_eval.prompting) so what you review is byte-for-byte what the model
sees — no drift. Renders:

  1. The rules primer (CATAN_RULES), broken into readable sections.
  2. The live action glossary + the output formats.
  3. A REAL example prompt (system + user) for an actual board, so you see the
     board summary and the per-action production annotations in context.

Writes viewer/rules.html. Open it in a browser to review and point out issues.

    python scripts/rules_ui.py [--seed 1000]
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import ActionType

from goldilocks_eval import prompt as P
from goldilocks_eval import prompting as PP
from harness.prompt import SYSTEM_PROMPT_ACTION_ONLY

CSS = """
:root{--bg:#f6f4ee;--card:#fff;--ink:#23201a;--muted:#6b6457;--accent:#b3551f;--line:#e3ddd0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:28px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 24px}
.sub code{background:#ece6d8;padding:1px 5px;border-radius:4px;font-size:13px}
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:34px 0 12px;border-bottom:2px solid var(--line);padding-bottom:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.card .label{font-weight:700;color:var(--accent)}
.card ul{margin:8px 0 0;padding-left:20px}
.card li{margin:3px 0}
.intro{font-style:italic;color:var(--muted)}
pre{background:#23201a;color:#ece6d8;border-radius:10px;padding:16px;overflow:auto;max-height:520px;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
.tag{display:inline-block;background:#ece6d8;color:var(--muted);border-radius:999px;padding:2px 10px;font-size:12px;margin-left:8px}
details{margin:10px 0}
summary{cursor:pointer;color:var(--accent);font-weight:600}
.note{color:var(--muted);font-size:14px;margin:6px 0 0}
.banner{background:#fff5ec;border:1px solid var(--accent);border-radius:10px;padding:14px 18px;margin:18px 0;font-size:14.5px;line-height:1.5}
.banner b{color:var(--accent)}
"""


def esc(s: str) -> str:
    return html.escape(s)


_LABEL_RE = re.compile(r"^([A-Z0-9][A-Za-z0-9 /-]{1,22}):\s*(.*)$")


def rules_sections_html(rules: str) -> str:
    """Render CATAN_RULES (paragraphs split on blank lines) as labeled cards."""
    out = []
    for para in rules.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if "Rules that matter" in para:
            out.append(f'<div class="card intro">{esc(para)}</div>')
            continue
        lines = para.split("\n")
        bullets = [l[2:] for l in lines if l.startswith("- ")]
        m = _LABEL_RE.match(lines[0])
        if m and bullets:
            items = "".join(f"<li>{esc(b)}</li>" for b in bullets)
            out.append(f'<div class="card"><span class="label">{esc(m.group(1))}</span>'
                       f'<ul>{items}</ul></div>')
        elif m:
            body = m.group(2)
            if len(lines) > 1:
                body += "\n" + "\n".join(lines[1:])
            out.append(f'<div class="card"><span class="label">{esc(m.group(1))}:</span> '
                       f'{esc(body)}</div>')
        else:
            out.append(f'<div class="card">{esc(para)}</div>')
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000, help="board to render the example prompt from")
    ap.add_argument("--out", default="viewer/rules.html")
    args = ap.parse_args()

    # A real example prompt: opening decision on a real board.
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=args.seed)
    example_user = P.build_user_prompt(game, Color.RED, Color.BLUE, game.playable_actions)

    # A real placement scenario prompt (the grading path).
    from goldilocks_eval.sample_scenarios import scenarios_for_seed
    scn = scenarios_for_seed(args.seed, "example_pool")[0]
    placement_user = PP.build_prompt(scn)

    sections = rules_sections_html(P.CATAN_RULES)

    h = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>1v1 Catan — Rules Review</title><style>{CSS}</style></head><body><div class="wrap">
<h1>1v1 Catan — Rules the model is given</h1>
<p class="sub">Generated from the live code (<code>goldilocks_eval/prompt.py</code> +
<code>goldilocks_eval/prompting.py</code>) — this is byte-for-byte what the model sees.
Review the sections below and flag anything wrong or missing.</p>

<div class="banner">
<b>Why there is no strategy on this page.</b> This is the <b>statable floor</b> —
the game rules, the legal moves, how to read the board (including pips as plain
dice-odds facts), and the output format — given <b>identically to the baseline and
the trained model</b>. All un-enumerable judgment (when to take a high-pip vs. a
resource-diverse spot, when to trade or block, when to switch from expanding to
cities to longest road) is taught only through the <b>RL reward</b> and never
appears here. That separation is what makes the trained-vs-baseline before/after a
valid measure of <i>learned strategy</i> rather than hint-following. See
<code>words-vs-rl.md</code> for the full principle.
</div>

<h2>Rules primer <span class="tag">CATAN_RULES — shared by play &amp; grading</span></h2>
{sections}

<h2>Live action glossary <span class="tag">live play only</span></h2>
<div class="card"><pre>{esc(P._LIVE_ACTION_GLOSSARY)}</pre></div>

<h2>Output formats</h2>
<div class="card"><span class="label">Live, with reasoning</span><pre>{esc(P.SYSTEM_PROMPT[len(P.RULES_1V1):].strip())}</pre></div>
<div class="card"><span class="label">Live, action-only</span><pre>{esc(SYSTEM_PROMPT_ACTION_ONLY[len(P.RULES_1V1):].strip())}</pre></div>
<div class="card"><span class="label">Placement grading</span><pre>{esc(PP.SYSTEM[len(P.CATAN_RULES):].strip())}</pre></div>

<h2>Example — full live prompt (opening move, seed {args.seed})</h2>
<p class="note">System prompt (rules + glossary + output) then the user turn. Note the
board summary and the per-option production annotations (<code>-&gt; WOOD:11(2p)…</code>)
that make node ids meaningful.</p>
<details open><summary>SYSTEM prompt</summary><pre>{esc(P.SYSTEM_PROMPT)}</pre></details>
<details open><summary>USER prompt (the live game state)</summary><pre>{esc(example_user)}</pre></details>

<h2>Example — placement grading prompt (seed {args.seed}, pick 1)</h2>
<details><summary>SYSTEM prompt</summary><pre>{esc(PP.SYSTEM)}</pre></details>
<details><summary>USER prompt</summary><pre>{esc(placement_user)}</pre></details>
</div></body></html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(h)
    print(f"Wrote {out}  ({len(h)} bytes)")
    print(f"Open it:  open {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
