# Caratan — web

A presentation-grade static site showcasing the Caratan RL project: teaching a
small open LLM (Qwen3-8B) to play 1v1 Settlers of Catan via GRPO.

Built with **Next.js (App Router, plain JavaScript)**. No database, no
serverless, no live play — it's a pure static export. The interactive Catan
board viewers are the original HTML viewers, copied into `public/viewer/`.

## Run locally

```bash
npm install
npm run dev
```

Then open http://localhost:3000.

To produce the static build (what Vercel deploys):

```bash
npm run build
```

The exported site lands in `out/`.

## Deploy to Vercel

This site is a static export and needs no environment variables, no database,
and no serverless functions.

1. Import the repo into Vercel.
2. Set **Root Directory** to `web`.
3. Framework preset: **Next.js** (auto-detected).
4. Leave build/output settings at their defaults — no env vars needed.
5. Deploy.

## What's inside

- `app/page.js` — the dashboard (hero, the loop, held-out results, training
  curves, session stats, explore links, local-play callout). Reads
  `app/data/results.json`.
- `app/globals.css` — single global stylesheet, dark theme.
- `public/data/results.json` — results data (also mirrored to `app/data/` for
  build-time import).
- `public/viewer/` — the original interactive HTML viewers (replay, play,
  rules, placement grading) plus `runs.json`.
- `public/transcripts/selfplay/` — the `.view.json` game files the replay
  viewer loads.

## Play vs the model (local only)

Live play runs locally against the inference gateway, not on this deployed site:

```bash
python scripts/play_server.py --model "fireworks:accounts/brickedup25/deployments/qhzroqz3"
# then open http://localhost:8000/viewer/play.html
```
