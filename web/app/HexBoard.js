// Decorative Catan board for the hero — pure SVG, no interactivity.
// The classic 19-hex layout (columns of 3-4-5-4-3), resource-colored flat-top
// hexes with number tokens, tuned to glow on a dark background.
const R = 56;
const SQ3 = Math.sqrt(3);
const COLS = [3, 4, 5, 4, 3];

const COLOR = {
  wood: "#3a7d44", brick: "#c0623a", sheep: "#83ad58",
  wheat: "#cda434", ore: "#6f7d90", desert: "#9c8b6a",
};
// 19 tiles + their number tokens (null = desert). Decorative, not a real game.
const TILES = ["ore","sheep","wood","wheat","brick","sheep","desert","wood","wheat",
  "ore","wood","brick","sheep","wheat","ore","brick","wheat","sheep","wood"];
const NUMS = [10,2,9,12,6,4,null,10,9,11,3,8,8,4,5,6,3,11,5];

function hexPoints(cx, cy) {
  return [0, 60, 120, 180, 240, 300]
    .map((a) => {
      const r = (a * Math.PI) / 180;
      return `${(cx + R * Math.cos(r)).toFixed(1)},${(cy + R * Math.sin(r)).toFixed(1)}`;
    })
    .join(" ");
}

export default function HexBoard() {
  const centers = [];
  COLS.forEach((n, i) => {
    for (let j = 0; j < n; j++) {
      centers.push({ x: i * 1.5 * R, y: (j - (n - 1) / 2) * SQ3 * R });
    }
  });
  const xs = centers.map((c) => c.x), ys = centers.map((c) => c.y);
  const pad = R + 10;
  const minX = Math.min(...xs) - pad, minY = Math.min(...ys) - pad;
  const W = Math.max(...xs) + pad - minX, H = Math.max(...ys) + pad - minY;

  return (
    <svg className="hexsvg" viewBox={`${minX} ${minY} ${W} ${H}`} aria-hidden="true">
      {centers.map((c, k) => {
        const res = TILES[k] || "sheep";
        const num = NUMS[k];
        const hot = num === 6 || num === 8;
        return (
          <g key={k}>
            <polygon
              points={hexPoints(c.x, c.y)}
              fill={COLOR[res]}
              stroke="rgba(8,12,18,0.55)"
              strokeWidth="3"
            />
            {num != null && (
              <g>
                <circle cx={c.x} cy={c.y} r={R * 0.34} fill="#efe6d2" />
                <text
                  x={c.x}
                  y={c.y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={R * 0.4}
                  fontWeight="700"
                  fill={hot ? "#b5341f" : "#2a2018"}
                  fontFamily="var(--font-display, sans-serif)"
                >
                  {num}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
