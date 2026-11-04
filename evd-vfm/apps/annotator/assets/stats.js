"use strict";
// Progress statistics: read-only view of /api/stats (apps/annotator/stats.py).
// A unit is completed when all its frames are reviewed or excluded, in progress when
// at least one frame has a saved state. Contract in docs/20_annotatore_spaziale.md.
const ST = {data: null, unit: "planes", sort: {key: "order", dir: 1}};
const fmt = n => Number(n).toLocaleString("en-US");
const percent = (a, b) => b ? `${Math.round(100 * a / b)}%` : "—";
const emptyCount = () => ({total: 0, done: 0, progress: 0});
const UNITS = {planes: "Planes", events: "Events", videos: "Videos", frames: "Frames"};

function frameState(f) {
  if (f.total && f.reviewed + f.excluded === f.total) return "done";
  return f.reviewed + f.excluded + f.draft ? "progress" : "todo";
}
function add(target, source) { for (const k of Object.keys(source)) target[k] = (target[k] || 0) + source[k]; return target; }
function tallyState(target, state) { target.total++; if (state !== "todo") target[state]++; }

// Projects a video onto the chosen class: with a class filter only its events and planes count.
function projectVideo(v, cls) {
  const frames = cls ? v.frames_by_class[cls] : v.frames;
  return {
    source: v, frames, state: frameState(frames),
    events: cls ? v.events.filter(e => e.class === cls) : v.events,
    planes: cls ? v.planes_by_class[cls] || emptyCount() : v.planes,
    boxes: cls ? v.boxes.filter(b => b[1].includes(cls)) : v.boxes,
    completed: cls ? v.completed_planes.filter(c => c[2].includes(cls)) : v.completed_planes,
  };
}
function scopedVideos() {
  const dataset = $("statsDataset").value, cls = $("statsClass").value;
  return ST.data.videos.filter(v => (!dataset || v.dataset === dataset) && (!cls || v.classes.includes(cls)))
    .map(v => projectVideo(v, cls));
}
function totals(views) {
  const t = {videos: emptyCount(), events: emptyCount(), planes: emptyCount(),
    frames: {total: 0, reviewed: 0, excluded: 0, draft: 0}, boxes: 0, boxPlanes: 0, emptyPlanes: 0};
  for (const v of views) {
    tallyState(t.videos, v.state);
    for (const e of v.events) tallyState(t.events, e.state);
    add(t.planes, v.planes); add(t.frames, v.frames);
    for (const [n, , count] of v.boxes) { t.boxes += n * count; t.boxPlanes += count; if (!n) t.emptyPlanes += count; }
  }
  return t;
}

// --- Marks ------------------------------------------------------------------------------
function tipText(lines) { return lines.filter(Boolean).join("\n"); }
function meter(parts, total, label) {
  const bar = el("div", {class: "meter", role: "img", "aria-label": label});
  const shown = parts.filter(p => p.value > 0);
  const rest = Math.max(0, total - shown.reduce((s, p) => s + p.value, 0));
  if (rest || !shown.length) shown.push({key: "todo", value: rest || 1, name: "To do", count: rest});
  for (const p of shown) {
    const seg = el("span", {class: `seg ${p.key}`, "data-tip": tipText([`${fmt(p.count ?? p.value)} · ${percent(p.count ?? p.value, total)}`, p.name, label])});
    seg.style.flexGrow = String(p.value);
    bar.append(seg);
  }
  return bar;
}
function countMeter(c, label) {
  return meter([{key: "done", value: c.done, name: "Completed"}, {key: "progress", value: c.progress, name: "In progress"}], c.total, label);
}
function frameMeter(f, label) {
  return meter([{key: "done", value: f.reviewed, name: "Reviewed"}, {key: "excluded", value: f.excluded, name: "Excluded"},
    {key: "progress", value: f.draft, name: "Draft"}], f.total, label);
}
function emptyRow(body, span, text) {
  const tr = el("tr"); tr.append(el("td", {colspan: String(span), class: "empty"}, text)); body.append(tr);
}
function tile(label, value, detail, bar) {
  const node = el("div", {class: "stat-tile"});
  node.append(el("div", {class: "stat-label"}, label), el("div", {class: "stat-value"}, value), el("div", {class: "stat-detail"}, detail));
  if (bar) node.append(bar);
  return node;
}
function countTile(label, c) {
  return tile(label, `${fmt(c.done)} / ${fmt(c.total)}`,
    `${percent(c.done, c.total)} completed${c.progress ? ` · ${fmt(c.progress)} in progress` : ""}`, countMeter(c, label));
}

// Single-series histogram: columns ≤ 24 px, 4 px rounded top, value on top.
function histogram(title, subtitle, bins, note, available = 440) {
  const card = el("figure", {class: "stats-card"});
  card.append(el("figcaption", {}, title), el("p", {class: "stats-sub"}, subtitle));
  // Real card width: text stays at 10 px instead of scaling with the viewBox.
  const W = Math.max(320, Math.round(available)), H = 200, top = 20, bottom = 38, left = 6, right = 6, inner = H - top - bottom;
  const chart = svg("svg", {viewBox: `0 0 ${W} ${H}`, class: "hist", role: "img", "aria-label": `${title}: ${bins.map(b => `${b.label} ${b.value}`).join(", ")}`});
  const max = Math.max(1, ...bins.map(b => b.value)), slot = (W - left - right) / bins.length;
  const width = Math.min(24, slot * 0.62), labelEvery = Math.ceil(bins.length / 14);
  chart.append(svg("line", {x1: left, x2: W - right, y1: top + inner, y2: top + inner, class: "baseline"}));
  bins.forEach((b, i) => {
    const cx = left + slot * (i + 0.5), h = b.value ? Math.max(2, inner * b.value / max) : 0, x = cx - width / 2, y = top + inner - h;
    const r = Math.min(4, h, width / 2);
    if (h) chart.append(svg("path", {class: "bar", d: `M${x},${top + inner}V${y + r}Q${x},${y} ${x + r},${y}H${x + width - r}Q${x + width},${y} ${x + width},${y + r}V${top + inner}Z`}));
    if (b.value && bins.length <= 14) chart.append(svg("text", {x: cx, y: y - 5, class: "value"}, fmt(b.value)));
    if (i % labelEvery === 0) chart.append(svg("text", {x: cx, y: top + inner + 15, class: "tick"}, b.label));
    const hit = svg("rect", {x: left + slot * i, y: top - 14, width: slot, height: inner + 14, class: "hit", tabindex: "0",
      "data-tip": tipText([fmt(b.value), b.tip || b.label])});
    chart.append(hit);
  });
  card.append(chart);
  if (note) card.append(el("p", {class: "stats-sub"}, note));
  return card;
}
function binned(values, edges, unit = "") {
  // edges: increasing lower bounds; the last bin is open.
  return edges.map((low, i) => {
    const high = edges[i + 1];
    const value = values.filter(v => v >= low && (high === undefined || v < high)).length;
    const label = high === undefined ? `${fmt(low)}+` : high - low === 1 ? fmt(low) : `${fmt(low)}–${fmt(high - 1)}`;
    return {label: label + unit, value};
  });
}

// --- Sections ---------------------------------------------------------------------------
function renderKpis(views, t) {
  const kpis = $("statsKpis"); kpis.replaceChildren();
  kpis.append(countTile("Videos", t.videos), countTile("Events", t.events), countTile("VTCD planes", t.planes),
    tile("Event frames", `${fmt(t.frames.reviewed + t.frames.excluded)} / ${fmt(t.frames.total)}`,
      `${percent(t.frames.reviewed + t.frames.excluded, t.frames.total)} closed · ${fmt(t.frames.excluded)} excluded · ${fmt(t.frames.draft)} draft`,
      frameMeter(t.frames, "Event frames")),
    tile("Boxes in reviewed planes", fmt(t.boxes),
      t.boxPlanes ? `${(t.boxes / t.boxPlanes).toLocaleString("en-US", {maximumFractionDigits: 1})} per plane · ${fmt(t.emptyPlanes)} planes without boxes` : "No reviewed plane"));
  const normal = Object.entries(ST.data.normal).filter(([d]) => !$("statsDataset").value || d === $("statsDataset").value)
    .reduce((s, [, n]) => s + n, 0);
  $("statsScope").textContent = `${fmt(views.length)} videos with events${$("statsClass").value ? ` of class ${$("statsClass").value}` : ""}`
    + `; ${fmt(normal)} normal videos excluded (no planes to annotate). Computed at ${new Date(ST.data.generated_at).toLocaleTimeString("en-US")}.`;
}
function renderClasses() {
  const dataset = $("statsDataset").value, only = $("statsClass").value;
  const rows = new Map();
  for (const v of ST.data.videos) {
    if (dataset && v.dataset !== dataset) continue;
    for (const cls of v.classes) {
      if (only && cls !== only) continue;
      const r = rows.get(cls) || {videos: emptyCount(), events: emptyCount(), planes: emptyCount(), frames: {total: 0, reviewed: 0, excluded: 0, draft: 0}};
      tallyState(r.videos, frameState(v.frames_by_class[cls]));
      for (const e of v.events) if (e.class === cls) tallyState(r.events, e.state);
      add(r.planes, v.planes_by_class[cls] || emptyCount()); add(r.frames, v.frames_by_class[cls]);
      rows.set(cls, r);
    }
  }
  const body = $("statsClasses"); body.replaceChildren();
  const fraction = c => `${fmt(c.done)} / ${fmt(c.total)}`;
  for (const [cls, r] of [...rows].sort((a, b) => a[0].localeCompare(b[0], "en"))) {
    const tr = el("tr");
    const closed = {done: r.frames.reviewed + r.frames.excluded, total: r.frames.total};
    tr.append(el("th", {scope: "row"}, cls), el("td", {}, fraction(r.videos)), el("td", {}, fraction(r.events)),
      el("td", {}, fraction(r.planes)), el("td", {}, percent(closed.done, closed.total)));
    const cell = el("td", {class: "meter-cell"});
    cell.append(ST.unit === "frames" ? frameMeter(r.frames, `${cls} · frame`) : countMeter(r[ST.unit], `${cls} · ${UNITS[ST.unit].toLowerCase()}`));
    tr.append(cell); body.append(tr);
  }
  if (!rows.size) emptyRow(body, 6, "No videos with these filters.");
}
function renderDistributions(views) {
  const grid = $("statsHistograms"); grid.replaceChildren();
  const columns = getComputedStyle(grid).gridTemplateColumns.split(" ").length;
  const width = (grid.clientWidth - 12 * (columns - 1)) / columns - 30;
  const planes = views.map(v => v.planes.total);
  grid.append(histogram("Planes per video", "How many base VTCD planes each video has", binned(planes, [1, 6, 11, 26, 51, 101, 201, 401]), "", width));
  const completion = views.filter(v => v.planes.total).map(v => v.planes.done / v.planes.total);
  const bands = [["0%", c => c === 0], ["1–24%", c => c > 0 && c < .25], ["25–49%", c => c >= .25 && c < .5],
    ["50–74%", c => c >= .5 && c < .75], ["75–99%", c => c >= .75 && c < 1], ["100%", c => c === 1]];
  grid.append(histogram("Completion per video", "Share of completed planes in each video",
    bands.map(([label, test]) => ({label, value: completion.filter(test).length, tip: `${label} of the planes completed`})), "", width));
  const counts = new Map();
  for (const v of views) for (const [n, , count] of v.boxes) counts.set(Math.min(n, 10), (counts.get(Math.min(n, 10)) || 0) + count);
  const boxBins = Array.from({length: 11}, (_, n) => ({label: n === 10 ? "10+" : String(n), value: counts.get(n) || 0,
    tip: `reviewed planes with ${n === 10 ? "10 or more boxes" : n === 1 ? "1 box" : `${n} boxes`}`}));
  grid.append(histogram("Boxes per reviewed plane", "Boxes in the first frame of each reviewed plane", boxBins,
    boxBins.some(b => b.value) ? "" : "No reviewed plane with these filters yet.", width));
  const seconds = views.flatMap(v => v.events.map(e => e.frames / v.source.fps));
  grid.append(histogram("Event duration", "Seconds of each annotated interval",
    binned(seconds.map(s => Math.floor(s)), [0, 2, 5, 10, 30, 60, 120, 300], " s"), "", width));
}
function renderTimeline(views) {
  const box = $("statsTimeline"); box.replaceChildren();
  const perDay = new Map(), perReviewer = new Map();
  for (const v of views) for (const [day, who, , n] of v.completed) {
    perDay.set(day, (perDay.get(day) || 0) + n); perReviewer.set(who || "—", (perReviewer.get(who || "—") || 0) + n);
  }
  if (!perDay.size) { box.append(el("p", {class: "empty"}, "No completed plane with these filters yet.")); return; }
  const days = [...perDay.keys()].sort(), bins = [];
  let cumulative = 0;
  for (let d = new Date(`${days[0]}T00:00:00Z`); d <= new Date(`${days.at(-1)}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + 1)) {
    const key = d.toISOString().slice(0, 10), value = perDay.get(key) || 0; cumulative += value;
    bins.push({label: `${key.slice(8, 10)}/${key.slice(5, 7)}`, value, tip: `${key.slice(8, 10)}/${key.slice(5, 7)} · ${fmt(cumulative)} in total`});
  }
  box.append(histogram("Planes completed per day", "Day of the last change to their frames: a later correction moves the plane",
    bins, "", box.clientWidth - 30));
  if (perReviewer.size > 1) box.append(el("p", {class: "stats-sub"},
    `Per reviewer: ${[...perReviewer].sort((a, b) => b[1] - a[1]).map(([who, n]) => `${who} ${fmt(n)}`).join(" · ")}`));
}
const VIDEO_COLUMNS = [
  ["name", "Video", v => v.source.video_id], ["dataset", "Dataset", v => v.source.dataset],
  ["classes", "Classes", v => v.source.classes.join(" + ")],
  ["events", "Events", v => v.events.length ? v.events.filter(e => e.state === "done").length / v.events.length : -1],
  ["planes", "Planes", v => v.planes.total], ["done", "Completed", v => v.planes.total ? v.planes.done / v.planes.total : -1],
  ["boxes", "Box", v => v.boxes.reduce((s, [n, , c]) => s + n * c, 0)], ["updated", "Last modified", v => v.source.updated_at || ""],
];
function renderVideos(views) {
  const search = $("statsSearch").value.trim().toLowerCase();
  const order = new Map(ST.data.videos.map((v, i) => [v.id, i]));
  const column = VIDEO_COLUMNS.find(c => c[0] === ST.sort.key);
  const rows = views.filter(v => !search || `${v.source.video_id} ${v.source.classes.join(" ")}`.toLowerCase().includes(search))
    .sort((a, b) => {
      if (!column) return order.get(a.source.id) - order.get(b.source.id);
      const x = column[2](a), y = column[2](b);
      const primary = typeof x === "string" ? x.localeCompare(y, "en", {numeric: true}) : x - y;
      return ST.sort.dir * primary || order.get(a.source.id) - order.get(b.source.id);
    });
  for (const th of $("statsVideoHead").querySelectorAll("th[data-key]"))
    th.setAttribute("aria-sort", th.dataset.key === ST.sort.key ? (ST.sort.dir > 0 ? "ascending" : "descending") : "none");
  const body = $("statsVideos"); body.replaceChildren();
  for (const v of rows) {
    const tr = el("tr", {tabindex: "0", title: "Open the video"});
    const doneEvents = v.events.filter(e => e.state === "done").length;
    tr.append(el("th", {scope: "row"}, v.source.video_id.replace("_x264", "")),
      el("td", {}, v.source.dataset === "ucf_crime" ? "UCF" : "MIVIA"), el("td", {class: "classes"}, v.source.classes.join(" + ")),
      el("td", {}, `${fmt(doneEvents)} / ${fmt(v.events.length)}`), el("td", {}, `${fmt(v.planes.done)} / ${fmt(v.planes.total)}`),
      el("td", {}, percent(v.planes.done, v.planes.total)), el("td", {}, fmt(VIDEO_COLUMNS[6][2](v))),
      el("td", {}, v.source.updated_at ? new Date(v.source.updated_at).toLocaleString("en-GB", {dateStyle: "short", timeStyle: "short"}) : "—"));
    const cell = el("td", {class: "meter-cell"}); cell.append(countMeter(v.planes, `${v.source.video_id} · planes`)); tr.append(cell);
    const open = () => { $("statsDialog").close(); openVideo(v.source.id).catch(e => toast(e.message, true)); };
    tr.onclick = open; tr.onkeydown = event => { if (event.key === "Enter") open(); };
    body.append(tr);
  }
  $("statsVideoCount").textContent = `${fmt(rows.length)} videos`;
  if (!rows.length) emptyRow(body, 9, "No matching video.");
}
function renderStats() {
  if (!ST.data) return;
  const views = scopedVideos(), t = totals(views);
  renderKpis(views, t); renderClasses(); renderDistributions(views); renderTimeline(views); renderVideos(views);
}
function populateClassFilter() {
  const select = $("statsClass"), chosen = select.value;
  const classes = [...new Set(ST.data.videos.flatMap(v => v.classes))].sort((a, b) => a.localeCompare(b, "en"));
  select.replaceChildren(el("option", {value: ""}, "All classes"), ...classes.map(c => el("option", {value: c}, c)));
  select.value = classes.includes(chosen) ? chosen : "";
}
async function loadStats() {
  $("statsBody").classList.add("refreshing"); $("statsRefresh").disabled = true;
  try { ST.data = await api("/api/stats"); populateClassFilter(); renderStats(); }
  catch (error) { toast(error.message, true); }
  finally { $("statsBody").classList.remove("refreshing"); $("statsRefresh").disabled = false; }
}
function exportStatsCsv() {
  const header = ["video_id", "dataset", "classes", "events_total", "events_completed", "events_in_progress", "planes_total",
    "planes_completed", "planes_in_progress", "frames_total", "frames_reviewed", "frames_excluded", "frames_draft", "boxes_reviewed_planes", "last_modified"];
  const quote = value => /[",\n;]/.test(String(value)) ? `"${String(value).replaceAll('"', '""')}"` : String(value);
  const lines = [header.join(",")];
  for (const v of scopedVideos()) {
    const ev = emptyCount(); for (const e of v.events) tallyState(ev, e.state);
    lines.push([v.source.video_id, v.source.dataset, v.source.classes.join(" + "), ev.total, ev.done, ev.progress, v.planes.total,
      v.planes.done, v.planes.progress, v.frames.total, v.frames.reviewed, v.frames.excluded, v.frames.draft,
      v.boxes.reduce((s, [n, , c]) => s + n * c, 0), v.source.updated_at].map(quote).join(","));
  }
  const suffix = [$("statsDataset").value, $("statsClass").value].filter(Boolean).join("-");
  download(lines.join("\n") + "\n", `annotation-stats${suffix ? `-${suffix}` : ""}.csv`, "text/csv;charset=utf-8");
}

// --- Tooltip: a single element, text set with textContent ------------------------------
function showTip(target, x, y) {
  const tip = $("statsTip"), lines = target.getAttribute("data-tip").split("\n");
  tip.replaceChildren(el("strong", {}, lines[0]), ...lines.slice(1).map(line => el("span", {}, line)));
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(window.innerWidth - box.width - 8, x + 14)}px`;
  tip.style.top = `${Math.max(8, y - box.height - 10)}px`;
}
function initStats() {
  $("statsButton").onclick = () => { stopPlay(); $("statsDialog").showModal(); loadStats(); };
  $("statsRefresh").onclick = loadStats;
  $("statsCsv").onclick = () => ST.data && exportStatsCsv();
  for (const id of ["statsDataset", "statsClass"]) $(id).onchange = renderStats;
  $("statsSearch").oninput = () => ST.data && renderVideos(scopedVideos());
  for (const button of $("statsUnit").querySelectorAll("button")) button.onclick = () => {
    ST.unit = button.dataset.unit;
    for (const b of $("statsUnit").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
    renderClasses();
  };
  for (const th of $("statsVideoHead").querySelectorAll("th[data-key]")) th.onclick = () => {
    ST.sort = ST.sort.key === th.dataset.key ? {key: th.dataset.key, dir: -ST.sort.dir}
      : {key: th.dataset.key, dir: ["name", "dataset", "classes"].includes(th.dataset.key) ? 1 : -1};
    renderVideos(scopedVideos());
  };
  const dialog = $("statsDialog");
  dialog.addEventListener("pointermove", event => {
    const target = event.target.closest?.("[data-tip]");
    if (target) showTip(target, event.clientX, event.clientY); else $("statsTip").hidden = true;
  });
  dialog.addEventListener("pointerleave", () => { $("statsTip").hidden = true; });
  dialog.addEventListener("focusin", event => {
    const target = event.target.closest?.("[data-tip]");
    if (!target) { $("statsTip").hidden = true; return; }
    const r = target.getBoundingClientRect(); showTip(target, r.left + r.width / 2, r.top);
  });
  dialog.addEventListener("close", () => { $("statsTip").hidden = true; });
  dialog.addEventListener("scroll", () => { $("statsTip").hidden = true; }, {capture: true});
}
initStats();
