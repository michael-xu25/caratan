"use client";
import { useEffect, useState } from "react";

/* Game picker for the live demo: reads /viewer/runs.json and shows every game as
   a card, grouped by matchup. Clicking a card opens that exact replay
   (/viewer/index?data=<view>) in a new tab. */
export default function GameGallery() {
  const [runs, setRuns] = useState(null);

  useEffect(() => {
    fetch("/viewer/runs.json", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => setRuns(d.runs || []))
      .catch(() => setRuns([]));
  }, []);

  if (runs === null) return <p className="section-sub">Loading games…</p>;
  if (!runs.length) return <p className="section-sub">No games found.</p>;

  return (
    <div className="gallery">
      {runs.map((run) => (
        <div className="gal-run" key={run.name}>
          <div className="gal-run-head">
            <h3>{run.name}</h3>
            <span className="gal-count">{run.games.length} games</span>
          </div>
          <div className="gal-grid">
            {run.games.map((g) => {
              const won = g.winner_model;
              return (
                <a
                  className={"gal-card" + (won ? " win" : "")}
                  key={g.view}
                  href={`/viewer#data=${g.view}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <div className="gal-card-top">
                    <span className="gal-seed">seed {g.seed}</span>
                    {won ? (
                      <span className="gal-badge win">{won} won</span>
                    ) : (
                      <span className="gal-badge draw">draw</span>
                    )}
                  </div>
                  <div className="gal-vp">
                    {g.final_vp
                      ? Object.entries(g.final_vp)
                          .map(([c, v]) => `${(g.seats && g.seats[c]) || c}: ${v} VP`)
                          .join("   ·   ")
                      : ""}
                  </div>
                  <div className="gal-foot">
                    <span>{g.num_steps} plies</span>
                    <span className="gal-play">▶ watch replay</span>
                  </div>
                </a>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
