/* ============================================================================
   Shared Caratan UI engine — board rendering + sidebar panels, used by BOTH
   play.html (interactive) and index.html (replay). Both pages pass the SAME
   shape (board + a per-state snapshot: roads / robber / buildings / vp / hands /
   rolls), produced by scripts/build_viewer_data.py and scripts/play_server.py.
   Edit here once; both UIs update.
   ============================================================================ */
(function (global) {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const txt = (x, y, s, o = {}) => { const t = el("text",
    { x, y, "text-anchor": "middle", "dominant-baseline": "middle", ...o }); t.textContent = s; return t; };

  // muted, modern resource palette (dark colonist look)
  const RES_COLOR = { WOOD:"#3f7d46", BRICK:"#b0543a", SHEEP:"#86a84e", WHEAT:"#c79f3c", ORE:"#647084", DESERT:"#c9b079" };
  const RES_ICON  = { WOOD:"🌲", BRICK:"🧱", SHEEP:"🐑", WHEAT:"🌾", ORE:"🪨", DEV:"🃏" };
  const RES_LABEL = { WOOD:"WOOD", BRICK:"BRICK", SHEEP:"WOOL", WHEAT:"GRAIN", ORE:"ORE" };
  const RES5 = ["WOOD","BRICK","SHEEP","WHEAT","ORE"];
  // engine seats are RED / BLUE; we DISPLAY the RED seat as ORANGE (data keys stay RED)
  const PCOL  = { RED:"#e8821e", BLUE:"#3b82f6", ORANGE:"#e8821e", WHITE:"#e9eef2" };
  const DNAME = { RED:"ORANGE" };
  const dname = c => DNAME[c] || c;
  const pcol  = c => PCOL[c] || "#aab";
  const ND = { NORTH:[0,-1], NORTHEAST:[Math.sqrt(3)/2,-.5], SOUTHEAST:[Math.sqrt(3)/2,.5],
               SOUTH:[0,1], SOUTHWEST:[-Math.sqrt(3)/2,.5], NORTHWEST:[-Math.sqrt(3)/2,-.5] };
  const PORT_DIR = { WEST:["NORTHWEST","SOUTHWEST"], NORTHWEST:["NORTH","NORTHWEST"],
                     NORTHEAST:["NORTHEAST","NORTH"], EAST:["SOUTHEAST","NORTHEAST"],
                     SOUTHEAST:["SOUTH","SOUTHEAST"], SOUTHWEST:["SOUTHWEST","SOUTH"] };
  const PIPS = {2:1,3:2,4:3,5:4,6:5,8:5,9:4,10:3,11:2,12:1};
  const COSTS = {
    BUILD_ROAD:{WOOD:1,BRICK:1}, BUILD_SETTLEMENT:{WOOD:1,BRICK:1,SHEEP:1,WHEAT:1},
    BUILD_CITY:{WHEAT:2,ORE:3}, BUY_DEVELOPMENT_CARD:{SHEEP:1,WHEAT:1,ORE:1},
  };

  // ---- board geometry --------------------------------------------------------
  function fit(svg, board) {
    const xs = [], ys = [];
    for (const n of Object.values(board.nodes)) { xs.push(n.x); ys.push(n.y); }
    const a = Math.min(...xs), b = Math.max(...xs), c = Math.min(...ys), d = Math.max(...ys), p = 1.25;
    svg.setAttribute("viewBox", `0 0 ${(b-a)+2*p} ${(d-c)+2*p}`);
    return (x, y) => [ (x-a)+p, (y-c)+p ];
  }
  const hexPts = (T, cx, cy, sc=1) =>
    Object.values(ND).map(([dx,dy]) => { const [x,y] = T(cx+dx*sc, cy+dy*sc); return `${x},${y}`; }).join(" ");

  // ---- board renderer (identical for play + replay) --------------------------
  // view: { board, roads, robber, buildings }; opts: { highlight:{node,type} }
  function drawBoard(svg, view, T, opts = {}) {
    const g = el("g"); svg.innerHTML = ""; svg.appendChild(g);
    const B = view.board;
    for (const t of B.tiles) {
      const water = (t.type === "WATER" || t.type === "PORT"), land = !water;
      const fill = t.type === "DESERT" ? RES_COLOR.DESERT : water ? "var(--water)" : (RES_COLOR[t.resource] || "#556");
      g.appendChild(el("polygon", { points: hexPts(T, t.x, t.y, .97), fill,
        stroke:"#0a111c", "stroke-width":.05, "stroke-linejoin":"round" }));
      const [cx, cy] = T(t.x, t.y);
      if (land && t.type !== "DESERT" && t.resource)
        g.appendChild(txt(cx, cy+.62, RES_LABEL[t.resource] || t.resource,
          { "font-size":.2, "font-weight":700, fill:"#ffffff", opacity:.5 }));
      if (t.type === "DESERT")
        g.appendChild(txt(cx, cy, "DESERT", { "font-size":.24, "font-weight":700, fill:"#7c6f4e" }));
      if (t.number != null) { const hot = t.number === 6 || t.number === 8;
        g.appendChild(el("circle", { cx, cy:cy-.05, r:.34, fill:"#f3ead2", stroke:"#cdbf9b", "stroke-width":.025 }));
        g.appendChild(txt(cx, cy-.07, t.number, { "font-size":.4, "font-weight":800, fill: hot?"#c0392b":"#2c2c2c" }));
        const n = PIPS[t.number] || 0;
        for (let k=0;k<n;k++) g.appendChild(el("circle",
          { cx:cx+(k-(n-1)/2)*.075, cy:cy+.26, r:.03, fill: hot?"#c0392b":"#6a6a6a" })); }
    }
    // ports: label the water hex (RES 2:1 / 3:1)
    for (const t of B.tiles) { if (t.type !== "PORT" || !t.port) continue; const [cx, cy] = T(t.x, t.y);
      if (t.port === "3:1") g.appendChild(txt(cx, cy, "3:1", { "font-size":.34, "font-weight":800, fill:"#cdd8e6" }));
      else { g.appendChild(txt(cx, cy-.1, RES_LABEL[t.port] || t.port, { "font-size":.24, "font-weight":800, fill:"#e7eef7" }));
             g.appendChild(txt(cx, cy+.18, "2:1", { "font-size":.2, "font-weight":700, fill:"#9fb0c4" })); } }
    // roads: dark casing under a bright bar
    for (const [k, color] of Object.entries(view.roads || {})) { const [a,b] = k.split("-");
      const na = B.nodes[a], nb = B.nodes[b]; if (!na || !nb) continue;
      const [x1,y1] = T(na.x,na.y), [x2,y2] = T(nb.x,nb.y);
      g.appendChild(el("line", { x1,y1,x2,y2, stroke:"#0a0f17", "stroke-width":.33, "stroke-linecap":"round" }));
      g.appendChild(el("line", { x1,y1,x2,y2, stroke:pcol(color), "stroke-width":.22, "stroke-linecap":"round" })); }
    // robber
    if (view.robber) { const t = B.tiles.find(t => t.coord.join() === view.robber.join());
      if (t) { const [cx,cy] = T(t.x,t.y);
        g.appendChild(el("circle", { cx, cy:cy+.02, r:.22, fill:"#11161e", opacity:.85, stroke:"#000", "stroke-width":.03 })); } }
    // optional highlight ring for the action being shown (replay)
    if (opts.highlight && (opts.highlight.type === "BUILD_SETTLEMENT" || opts.highlight.type === "BUILD_CITY")) {
      const n = B.nodes[String(opts.highlight.node)];
      if (n) { const [x,y] = T(n.x,n.y); g.appendChild(el("circle", { cx:x, cy:y, r:.42, fill:"none", stroke:"#f5c33b", "stroke-width":.09 })); } }
    // settlements = house; cities = bigger house + storey line
    const house = (x,y,s) => [[x-s,y+s],[x+s,y+s],[x+s,y-s*.25],[x,y-s*1.05],[x-s,y-s*.25]].map(p=>p.join(",")).join(" ");
    for (const [nid, bd] of Object.entries(view.buildings || {})) { const n = B.nodes[nid]; if (!n) continue;
      const [x,y] = T(n.x,n.y), city = bd.type === "CITY", s = city?.26:.19;
      g.appendChild(el("polygon", { points:house(x,y,s), fill:pcol(bd.color), stroke:"#fff", "stroke-width":.06, "stroke-linejoin":"round", "paint-order":"stroke" }));
      if (city) g.appendChild(el("line", { x1:x-s*.78, y1:y+s*.3, x2:x+s*.78, y2:y+s*.3, stroke:"#fff", "stroke-width":.05 })); }
    return g;   // caller may append more (e.g. play's selection overlay)
  }

  // ---- sidebar panels --------------------------------------------------------
  const costChips = cost => `<span class="costs">${Object.entries(cost)
    .map(([r,n]) => `<span class="cost">${RES_ICON[r]}${n}</span>`).join("")}</span>`;

  function renderVP(node, players, vp, opts = {}) {
    node.innerHTML = players.map(c => { const v = (vp||{})[c]||0;
      const role = opts.human ? `<span class="role2">${c===opts.human?"(you)":"(model)"}</span>` : "";
      const fin = opts.finalVp ? `<span class="muted"> · fin ${opts.finalVp[c] ?? "?"}</span>` : "";
      return `<div class="vprow"><div class="who"><span class="dot" style="background:${pcol(c)}"></span>`+
        `<b style="color:${pcol(c)}">${dname(c)}</b>${role}</div>`+
        `<div class="vpbar"><div class="vpfill" style="width:${Math.min(100,v/10*100)}%;background:${pcol(c)}"></div></div>`+
        `<div class="vpval">${v}${fin}</div></div>`; }).join("");
  }

  function renderHands(node, players, hands, opts = {}) {
    node.innerHTML = players.map(c => { const h = (hands||{})[c]||{};
      const total = RES5.reduce((s,r)=>s+(h[r]||0),0);
      const tiles = RES5.map(r => { const n = h[r]||0;
        return `<div class="rcard ${n?'':'z'}" style="--rc:${RES_COLOR[r]}"><div class="ic">${RES_ICON[r]}</div><div class="ct">${n}</div></div>`; }).join("");
      const dn = h.DEV||0;
      const devClick = opts.onToggleDev ? ` onclick="${opts.onToggleDev}('${c}')"` : "";
      const devTile = `<div class="rcard dev ${dn?'':'z'}" style="--rc:#6b5bd1"${devClick}><div class="ic">${RES_ICON.DEV}</div><div class="ct">${dn}</div></div>`;
      let block = `<div class="handblock"><div class="handhead"><span class="dot" style="background:${pcol(c)}"></span>`+
        `<b style="color:${pcol(c)}">${dname(c)}</b> · ${total} cards</div><div class="cards">${tiles}${devTile}</div>`;
      if (opts.expanded && opts.expanded.has(c)) {
        const dc = h.dev_cards || {};
        const parts = Object.entries(dc).filter(([,v])=>v>0).map(([k,v])=>`${k}×${v}`);
        block += `<div class="devdetail">${parts.length?parts.join(", "):"(no dev cards)"}</div>`; }
      return block + `</div>`; }).join("");
  }

  // dice bar OVER the board: big latest roll + a few recent totals.
  // accepts either a rolls[] list ({color,dice,total}) or a single {color,dice}.
  function renderDiceBar(node, rolls) {
    const list = Array.isArray(rolls) ? rolls : (rolls ? [rolls] : []);
    if (!list.length) { node.className = "empty"; node.innerHTML = ""; return; }
    node.className = "";
    const last = list[list.length-1];
    const recent = list.slice(-4, -1).reverse();
    node.innerHTML =
      `<span class="dwho"><span class="dot" style="width:11px;height:11px;border-radius:50%;background:${pcol(last.color)}"></span>`+
      `<span style="color:${pcol(last.color)}">${dname(last.color)}</span></span>`+
      `<span class="bigdie">${last.dice[0]}</span><span class="bigdie">${last.dice[1]}</span>`+
      `<span class="dtot">${last.total ?? (last.dice[0]+last.dice[1])}</span>`+
      (recent.length ? `<span class="recent">${recent.map(r=>
        `<span class="rchip"><span class="dot" style="width:8px;height:8px;border-radius:50%;background:${pcol(r.color)}"></span>${r.total ?? (r.dice[0]+r.dice[1])}</span>`).join("")}</span>` : "");
  }

  global.CatanUI = { NS, el, txt, RES_COLOR, RES_ICON, RES_LABEL, RES5, PCOL, DNAME, dname, pcol,
    ND, PORT_DIR, PIPS, COSTS, fit, hexPts, drawBoard, costChips, renderVP, renderHands, renderDiceBar };
})(window);
