import results from "./data/results.json";
import GameGallery from "./GameGallery";
import HexBoard from "./HexBoard";

/* ---------- helpers ---------- */
function fmtVal(v) {
  // Values <= 1 are treated as rates and shown as percentages.
  if (typeof v === "number" && Math.abs(v) <= 1) {
    return Math.round(v * 100) + "%";
  }
  return String(v);
}

function num(v, digits) {
  if (typeof v !== "number") return String(v);
  return v.toFixed(digits == null ? 3 : digits);
}

/* ---------- inline-SVG line chart ---------- */
function Sparkline({ data }) {
  const W = 300;
  const H = 110;
  const padL = 8;
  const padR = 8;
  const padT = 12;
  const padB = 18;
  const n = data.length;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const x = (i) => padL + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
  const y = (v) => padT + plotH - ((v - min) / span) * plotH;

  const linePts = data.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const areaPts = `${padL},${padT + plotH} ${linePts} ${padL + plotW},${padT + plotH}`;

  const first = data[0];
  const last = data[n - 1];
  const baseY = padT + plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="training reward curve">
      <defs>
        <linearGradient id="curveFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(255,138,61,0.28)" />
          <stop offset="100%" stopColor="rgba(255,138,61,0)" />
        </linearGradient>
      </defs>
      {/* baseline axis */}
      <line
        x1={padL}
        y1={baseY}
        x2={padL + plotW}
        y2={baseY}
        stroke="#2a3646"
        strokeWidth="1"
      />
      {/* area + line */}
      <polygon points={areaPts} fill="url(#curveFill)" />
      <polyline
        points={linePts}
        fill="none"
        stroke="#ff8a3d"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {/* endpoint dots */}
      <circle cx={x(0)} cy={y(first)} r="3" fill="#93a3b8" />
      <circle cx={x(n - 1)} cy={y(last)} r="3.5" fill="#ff8a3d" />
    </svg>
  );
}

/* ---------- the loop ---------- */
const LOOP = [
  {
    t: "Play games",
    d: "Run 1v1 self-play with the base open model to generate full game transcripts.",
  },
  {
    t: "Discover weaknesses",
    d: "Mine the transcripts for systematic mistakes: poor openings, hoarding, aimless trades.",
  },
  {
    t: "Build a verifiable env",
    d: "Turn each weakness into a small RL env with a deterministic, programmatic reward.",
  },
  {
    t: "GRPO-train",
    d: "Group-relative policy optimization on Qwen3.5 8B (thinking off) via HUD / Tinker, our graders as reward.",
  },
  {
    t: "Measure improvement",
    d: "Score before vs after on held-out, unseen boards to confirm real generalization.",
  },
];

const EXPLORE = [
  {
    emoji: "🎲",
    title: "Game replays",
    desc: "Step through trained-model-vs-base games on rendered Catan boards.",
    href: "/viewer/index",
  },
  {
    emoji: "📊",
    title: "Matchups & evals",
    desc: "Head-to-head win rates and per-decision held-out scores.",
    href: "/viewer/matchups",
  },
  {
    emoji: "📍",
    title: "Placement grading",
    desc: "How opening settlement spots are scored against the optimal pick.",
    href: "/viewer/placement_grading",
  },
  {
    emoji: "📖",
    title: "Rules",
    desc: "The exact 1v1 Catan ruleset the engine and model play under.",
    href: "/viewer/rules",
  },
];

export default function Page() {
  const { model, curves, heldout, stats, winrate, loop_run } = results;
  const curveOrder = ["placement", "maritime", "build"];

  return (
    <>
      {/* nav */}
      <nav className="nav">
        <div className="wrap">
          <div className="brand">
            <span className="dot" />
            Caratan
          </div>
          <div className="nav-links">
            <a href="#games">Games</a>
            <a href="#winrate">Win rate</a>
            <a href="#loop">The loop</a>
            <a href="#results">Results</a>
            <a href="#curves">Training</a>
            <a href="#explore">Explore</a>
            <a href="#play">Play</a>
          </div>
        </div>
      </nav>

      {/* hero */}
      <header className="hero">
        <div className="hero-glow" aria-hidden="true" />
        <div className="hero-board" aria-hidden="true">
          <HexBoard />
        </div>
        <div className="wrap">
          <span className="tag">Reinforcement learning · Settlers of Catan</span>
          <h1>Caratan</h1>
          <p className="lede">
            Teaching a small, open LLM to play 1v1 Settlers of Catan with
            reinforcement learning — discover its weaknesses, build verifiable
            environments, and GRPO-train it to play measurably better.
          </p>
          <div className="model-tag">{model}</div>
          <div className="hero-cta">
            <a className="btn replay" href="#games">
              <span className="play-ico">▶</span> Watch the games
            </a>
            <a className="btn" href="#winrate">
              See the win rate
            </a>
          </div>
        </div>
      </header>

      {/* win rate — headline stat, first thing after hero */}
      <section id="winrate" className="winrate-band">
        <div className="wrap">
          <p className="eyebrow">Headline · head-to-head win rate</p>
          <h2 className="section-title">
            The trained model beats the baseline {Math.round(winrate.headline.trained * 100)}% of the time
          </h2>
          <p className="section-sub">
            {winrate.headline.games} games, trained vs. the untrained baseline — a coin-flip
            baseline is {Math.round(winrate.headline.baseline * 100)}%.
          </p>
          <div className="wr-headline">
            <div className="wr-hero best">
              <div className="wr-hero-val">{Math.round(winrate.headline.trained * 100)}%</div>
              <div className="wr-hero-lbl">trained model wins</div>
            </div>
            <div className="wr-vs">vs</div>
            <div className="wr-hero">
              <div className="wr-hero-val muted">{Math.round(winrate.headline.baseline * 100)}%</div>
              <div className="wr-hero-lbl">baseline (parity)</div>
            </div>
          </div>
          <div className="wr-cards">
            {winrate.entries.map((e, i) => (
              <div className={"wr-card" + (i === 0 ? " best" : "")} key={e.label}>
                {i === 0 && <span className="wr-flag">Head-to-head</span>}
                <div className="wr-value">{Math.round(e.value * 100)}%</div>
                <div className="wr-label">{e.label}</div>
                <div className="wr-sub">{e.sub}</div>
                <div className="wr-note">{e.note}</div>
              </div>
            ))}
          </div>
          {winrate.trajectory && (
            <div className="wr-traj">
              <div className="wr-traj-title">Win-rate trajectory · recursive self-improvement</div>
              <div className="wr-traj-row">
                {winrate.trajectory.map((t, i) => (
                  <div className="wr-traj-pt" key={i}>
                    {i > 0 && <span className="wr-traj-arrow">→</span>}
                    <div className="wr-traj-val">{Math.round(t.winrate * 100)}%</div>
                    <div className="wr-traj-lbl">{t.label}</div>
                    {t.games ? <div className="wr-traj-sub">{t.games} games</div> : null}
                  </div>
                ))}
              </div>
            </div>
          )}
          <p className="heldout-note">{winrate.footnote}</p>
        </div>
      </section>

      {/* game picker */}
      <section id="games">
        <div className="wrap">
          <p className="eyebrow">Watch the games</p>
          <h2 className="section-title">Pick a game to replay</h2>
          <p className="section-sub">
            Step through any game on a rendered board — the trained model beating the
            base, or the base model playing itself. Click a game to open its replay.
          </p>
          <GameGallery />
        </div>
      </section>

      {/* the loop */}
      <section id="loop">
        <div className="wrap">
          <p className="eyebrow">Fully autonomous · recursive self-improvement</p>
          <h2 className="section-title">An infinite self-improvement loop</h2>
          <p className="section-sub">
            No human in the loop. The system plays, finds its own weaknesses, builds the
            reward that fixes them, trains, and verifies on unseen boards — then starts
            over on a stronger model. Each pass compounds: recursive self-improvement
            that just keeps running.
          </p>
          <div className="loop">
            {LOOP.map((s, i) => (
              <div className="loop-card" key={s.t}>
                <div className="num">{i + 1}</div>
                <h3>{s.t}</h3>
                <p>{s.d}</p>
              </div>
            ))}
          </div>
          <div className="loop-cycle">
            ↻ repeats autonomously — every cycle makes the next one smarter
          </div>

          {loop_run && (
            <div className="loop-run">
              <p className="eyebrow">We left it running</p>
              <h3 className="section-title">
                {loop_run.rounds} autonomous rounds in {loop_run.hours} hours
              </h3>
              <div className="wr-traj">
                <div className="wr-traj-row">
                  <div className="wr-traj-pt">
                    <div className="wr-traj-val">{Math.round(loop_run.winrate_start * 100)}%</div>
                    <div className="wr-traj-lbl">fresh model vs base</div>
                  </div>
                  <div className="wr-traj-pt">
                    <span className="wr-traj-arrow">→</span>
                    <div className="wr-traj-val">{Math.round(loop_run.winrate_end * 100)}%</div>
                    <div className="wr-traj-lbl">after {loop_run.hours}h, fully autonomous</div>
                  </div>
                </div>
              </div>
              <p className="section-sub">{loop_run.note}</p>
            </div>
          )}
        </div>
      </section>

      {/* headline results */}
      <section id="results">
        <div className="wrap">
          <p className="eyebrow">Headline results</p>
          <h2 className="section-title">Before → after on held-out boards</h2>
          <p className="section-sub">
            Each environment evaluated on a disjoint set of unseen boards. The
            base model vs. the GRPO-trained checkpoint, both forced to answer
            directly so we measure pick quality, not formatting.
          </p>
          <div className="results">
            {heldout.map((h) => {
              // "good" direction: lower is better only for the maritime trade-rate.
              const lowerBetter = h.env === "maritime";
              const afterClass = lowerBetter ? "after bad" : "after";
              return (
                <div className="result-card" key={h.env}>
                  <div className="head">
                    <span className="title">{h.title}</span>
                    <span className="metric">{h.metric}</span>
                  </div>
                  <div className="ba">
                    <span className="before">{fmtVal(h.before)}</span>
                    <span className="arrow">→</span>
                    <span className={afterClass}>{fmtVal(h.after)}</span>
                  </div>
                  <div className="extra">{h.extra}</div>
                </div>
              );
            })}
          </div>
          <p className="heldout-note">
            Held-out = scored on boards the model never trained on. Placement and
            build generalize cleanly; maritime learned to stop over-trading
            (lower trade-rate is the intended direction here).
          </p>
        </div>
      </section>

      {/* training curves */}
      <section id="curves">
        <div className="wrap">
          <p className="eyebrow">Training</p>
          <h2 className="section-title">Reward over optimization steps</h2>
          <p className="section-sub">
            Per-step reward on the training boards for each environment during
            the final GRPO run. All three show a clear climb.
          </p>
          <div className="curves">
            {curveOrder.map((key) => {
              const c = curves[key];
              if (!c) return null;
              const arr = c.reward;
              return (
                <div className="curve-card" key={key}>
                  <div className="head">
                    <span className="name">{key}</span>
                    <span className="metric">{c.metric}</span>
                  </div>
                  <Sparkline data={arr} />
                  <div className="curve-foot">
                    <span>start {num(arr[0])}</span>
                    <span>{arr.length} steps</span>
                    <span>end {num(arr[arr.length - 1])}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* session stats */}
      <section id="stats">
        <div className="wrap">
          <p className="eyebrow">Session</p>
          <h2 className="section-title">What it took</h2>
          <p className="section-sub">
            The whole overnight GRPO session across all three environments.
          </p>
          <div className="stats">
            <div className="stat">
              <div className="big">{stats.optim_steps}</div>
              <div className="lbl">optim steps</div>
            </div>
            <div className="stat">
              <div className="big">{stats.rollouts.toLocaleString()}</div>
              <div className="lbl">rollouts</div>
            </div>
            <div className="stat">
              <div className="big">{stats.train_minutes} min</div>
              <div className="lbl">training time</div>
            </div>
            <div className="stat">
              <div className="big">~${stats.infer_cost_usd}</div>
              <div className="lbl">inference cost</div>
            </div>
          </div>
        </div>
      </section>

      {/* explore */}
      <section id="explore">
        <div className="wrap">
          <p className="eyebrow">Explore</p>
          <h2 className="section-title">Look under the hood</h2>
          <p className="section-sub">
            Interactive static viewers — open in a new tab.
          </p>
          <div className="explore">
            {EXPLORE.map((e) => (
              <a
                className="explore-card"
                key={e.title}
                href={e.href}
                target="_blank"
                rel="noreferrer"
              >
                <span className="emoji">{e.emoji}</span>
                <h3>{e.title}</h3>
                <p>{e.desc}</p>
                <span className="go">Open ↗</span>
              </a>
            ))}
          </div>
        </div>
      </section>

      {/* play vs model (local) */}
      <section id="play">
        <div className="wrap">
          <div className="callout">
            <span className="badge">Runs locally</span>
            <h3>Play 1v1 against the trained model</h3>
            <p>
              Live play isn&apos;t deployed on this site — it needs the inference
              gateway and a local game server. To play against the trained model
              yourself, run it locally from the repo:
            </p>
            <div className="code">
              <span className="cmt"># 1. start the local play server</span>
              {"\n"}python scripts/play_server.py --model{" "}
              <span className="accent">
                &quot;fireworks:accounts/brickedup25/deployments/qhzroqz3&quot;
              </span>
              {"\n\n"}
              <span className="cmt"># 2. then open the play UI in your browser</span>
              {"\n"}http://localhost:8000/viewer/play.html
            </div>
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap">
          Caratan · {model} · static demo built with Next.js
        </div>
      </footer>
    </>
  );
}
