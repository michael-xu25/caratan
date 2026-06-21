# Caratan Viewer & Results Dashboard

A **static, dependency-free** UI (plain HTML/CSS/JS — no build step). Serve the
repo root and open the pages:

```bash
python -m http.server 8000
# replay viewer:     http://localhost:8000/viewer/index.html
# results dashboard: http://localhost:8000/viewer/matchups.html
```

Shared look-and-feel lives in **`catan-ui.css`** + **`catan-ui.js`** (`CatanUI`:
board rendering, dice bar, hands/VP panels, color helpers). Both pages import it,
so the replay and play views render identically.

> Color convention: the engine seats are `RED`/`BLUE`; the RED seat is **displayed
> as ORANGE**. The UI leads with the **model name**, not the color, throughout.

---

## Pages

### `index.html` — replay viewer
Step through any saved game: board, robber, roads/settlements/cities, the dice bar,
hands (resources public, dev cards click-to-reveal), VP, and awards (longest road /
largest army). Features:

- **Run picker** (top) — grouped by run directory (`runs.json`); each game labeled
  with the **winning model** (e.g. *catan-grpo-q8b won · 240 plies*).
- **🔄 mirror** — jump between a seed's `norm` and `swap` games (only shown when the
  sibling transcript exists). Makes seat-swap pairs one click apart.
- **⏪ game / game ⏩** — step to the previous/next transcript in the list (disabled
  at the ends), flanking the ply controls (⏮ ◀ ▶ Play ▶ ⏭).
- **Winner banner + seat legend** — names the winning **model** (color secondary)
  and makes the model↔color mapping obvious.
- **⚖️ Grading overlay** — appears when a `<game>.grading.json` sidecar exists:
  per-decision regret + oracle best move, dual Claude/GPT criteria, and game-level
  failure modes.
- **🎮 Play vs Model / 🕹️ Take over from here** — launch an interactive game
  (needs the play server, `scripts/play_server.py`).
- **📊 Matchups & Evals** — link to the dashboard.
- Drag-and-drop a `.view.json`, or deep-link with `?data=<view.json>`,
  `?run=<run-dir>`, `?step=<n>`.

### `matchups.html` — results dashboard
Three panels, each drawing from its own stats file, **auto-refreshing every 15s**
(numbers tick up live while a matchup runs):

1. **Head-to-head win-rate** (`data/matchups.json` ← `winrate_stats.py`) — one card
   per matchup: trained-win/draw/base bar, win-rate (decided), avg VP, cap-stalls,
   and a deep-link to browse that matchup's games. Cards carry **full-chain vs
   settlement-only** descriptors to distinguish the trained checkpoints.
2. **Game-quality stats** (`data/gamestats.json` ← `gamestats.py`) — untrained vs
   settlement-only vs fully-trained, with comparison bars + a full table:
   skipped-turn rate (turns where nothing happened), resource gain/game,
   settlements/cities/roads, dev buys, trades, and **pair-sweep rate**
   (adaptability — winning both mirror orientations).
3. **Per-decision held-out eval** (`data/eval_holdout.json`) — before/after on
   frozen scenarios (placement / build / maritime), with bars, metric tables, and a
   note on how this differs from the head-to-head games.

### Other pages
- **`play.html`** — interactive game vs a model (via the play server).
- **`rules.html`** — Catan rules reference.
- **`placement_grading.html`** — placement-scenario grading view.

---

## Data files (`viewer/data/` + `runs.json`)

| file | produced by | drives |
|---|---|---|
| `runs.json` | `scripts/build_viewer_index.py` | replay run picker (groups by run dir, skips `_`-prefixed scratch dirs) |
| `data/matchups.json` | `scripts/winrate_stats.py` | dashboard win-rate cards |
| `data/gamestats.json` | `scripts/gamestats.py` | dashboard game-quality stats |
| `data/eval_holdout.json` | the held-out eval | dashboard per-decision eval |
| `<game>.view.json` | `scripts/build_viewer_data.py` | one replay (lives next to its transcript) |
| `<game>.grading.json` | the grader pipeline | the ⚖️ overlay (optional sidecar) |

---

## How updates flow (live pipeline)

```
matchup run writes transcripts/<run>/seedN_{norm,swap}.json
        │
        ├─ build_viewer_data.py   → seedN_*.view.json   (replay files)
        ├─ build_viewer_index.py  → runs.json           (picker manifest)
        ├─ winrate_stats.py       → data/matchups.json  (win-rate)
        └─ gamestats.py           → data/gamestats.json (game-quality)
                                          │
                              matchups.html polls every 15s → redraws
```

During a run a background refresher re-runs the build + stats steps every ~45s, so
the dashboard stays current with no manual rebuild. To refresh by hand:

```bash
python scripts/build_viewer_data.py transcripts/hud-grpo-vs-base
python scripts/build_viewer_index.py
python scripts/winrate_stats.py transcripts/hud-grpo-vs-base
python scripts/gamestats.py
```
