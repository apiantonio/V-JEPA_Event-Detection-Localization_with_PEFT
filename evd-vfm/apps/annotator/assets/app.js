"use strict";
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const copy = value => structuredClone(value);
const S = {catalog: [], token: "", video: null, state: null, frame: 0, sources: [], candidates: [],
  proposals: {}, units: [], unitMode: "planes", selected: new Set(), tool: "select", hidden: false, zoom: 1,
  grid: null, defaultGrid: null, gridVisible: false, tokenVisible: false, precision: false, context: null,
  undo: [], redo: [], pendingHistory: null, historyBusy: false, dirty: 0, saved: 0, saving: null,
  loadId: 0, openId: 0, sourceRequest: 0, loading: false, playing: false, playTimer: null, gesture: null, objectUrl: null};
let toastTimer;
function toast(message, error = false) {
  $("toast").textContent = message; $("toast").classList.toggle("error", error);
  $("toast").hidden = false; clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").hidden = true, error ? 7000 : 3200);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: "POST",
    headers: {"Content-Type": "application/json", "X-Annotation-Token": S.token}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Error ${response.status}`);
  return result;
}
const endpoint = suffix => `/api/video/${encodeURIComponent(S.video.id)}${suffix || ""}`;
const classesAt = frame => [...new Set(S.video.events.filter(e => e.start <= frame && frame <= e.end).map(e => e.class_name))];
const activeUnit = (frame = S.frame) => S.units.find(u => u.start <= frame && frame <= u.end);
const planeScope = () => S.unitMode === "planes" && activeUnit()?.known ? activeUnit() : null;
const keyframeSource = () => S.sources.find(s=>s.id===S.state?.source_id)?.metadata.source_type === "external_keyframes";
// Proposals are the editable base; a saved entry, even empty, always wins.
const current = () => S.state?.frames[String(S.frame)] || {boxes: S.proposals[String(S.frame)] || [], status: "draft", note: ""};
const currentBoxes = () => current().boxes;
const boxNumber = (box, index) => box.display_id ?? index + 1;
const nextBoxNumber = () => Math.max(0,...currentBoxes().map(boxNumber)) + 1;
const boxColor = box => box.color || "#8bffbd";
const writable = () => S.video && !S.loading && !S.playing && !S.historyBusy;
function time(frame, fps = S.video.fps) {
  const seconds = frame / fps;
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}
function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function svg(tag, attrs, text) {
  const node = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function showSave(label, pending = false) {
  $("saveStatus").textContent = label; $("saveStatus").classList.toggle("pending", pending);
}
function markDirty() {
  recordHistory();
  S.dirty++; showSave("Unsaved changes", true);
}
async function flush() {
  if (S.saving) await S.saving;
  if (!S.state || S.saved === S.dirty) return;
  if (!$("reviewer").value.trim()) {
    showSave("Enter your name to save", true);
    $("reviewer").focus(); throw new Error("Enter the reviewer name to save the changes.");
  }
  S.saving = (async () => {
    while (S.saved !== S.dirty) {
      const generation = S.dirty;
      const payload = {...copy(S.state), reviewer: $("reviewer").value.trim(), last_frame: S.frame};
      showSave("Saving…", true);
      const result = await api(endpoint(), payload);
      S.state.revision = result.revision;
      S.state.reviewer = result.reviewer;
      S.saved = generation;
      updateCatalogProgress();
    }
    showSave(`Saved · revision ${S.state.revision}`);
  })();
  try { await S.saving; }
  catch (error) {
    showSave("Save failed", true);
    $("errorText").textContent = error.message;
    if (!$("errorDialog").open) $("errorDialog").showModal();
    throw error;
  } finally { S.saving = null; }
}
function checkpoint() {
  S.pendingHistory = {frames: {...S.state.frames}, source: S.state.source_id || "", video: S.video.id,
    frame: S.frame, mode: S.unitMode, selected: [...S.selected]};
}
function annotationEqual(a, b) {
  const content = f => f && {boxes:f.boxes.map(({token_space,...b})=>b), status:f.status, note:f.note || "", review_unit:f.review_unit};
  return JSON.stringify(content(a)) === JSON.stringify(content(b));
}
function recordHistory() {
  const before = S.pendingHistory; S.pendingHistory = null;
  if (!before) return;
  const changes = [];
  for (const key of new Set([...Object.keys(before.frames), ...Object.keys(S.state.frames)])) {
    const a = before.frames[key], b = S.state.frames[key];
    if (a !== b && !annotationEqual(a,b)) changes.push({key, before:a, after:b});
  }
  const source = S.state.source_id || "";
  if (!changes.length && before.source === source) return;
  // Keeps only the changed frames, without the old 25-action limit.
  S.undo.push({video:before.video, frame:before.frame, modeBefore:before.mode, modeAfter:S.unitMode, selected:before.selected,
    sourceBefore:before.source, sourceAfter:source, changes});
  S.redo = [];
}
function unitConflict(unit = planeScope()) {
  if (!unit) return false;
  const existing = [];
  for (let n = unit.start; n <= unit.end; n++) if (S.state.frames[String(n)]) existing.push(S.state.frames[String(n)]);
  if (!existing.length) return false;
  const signature = f => JSON.stringify({boxes:f.boxes.map(({id,token_space,...b}) => b), status:f.status, note:f.note||""});
  return existing.length !== unit.end - unit.start + 1 || existing.some(f => signature(f) !== signature(existing[0]));
}
function applyScope(frame, unit = planeScope()) {
  if (unit) {
    frame.review_unit = {id:unit.id,source_id:S.state.source_id,start:unit.start,end:unit.end,mode:"plane_hold"};
    for (let n = unit.start; n <= unit.end; n++) S.state.frames[String(n)] = copy(frame);
  } else { delete frame.review_unit; S.state.frames[String(S.frame)] = frame; }
}
function scopeReady() {
  if (S.unitMode === "planes" && S.units.length && !activeUnit()?.known) {
    toast("The plane mapping is missing here. Switch to Frames for a single-frame correction.", true); return false;
  }
  if (unitConflict()) { toast("This plane contains different per-frame corrections. Use Frames or Unify plane.", true); return false; }
  return true;
}
function editFrame(change, force = false) {
  if (!writable()) return;
  if (!scopeReady()) return;
  const base = copy(current()); base.boxes.forEach((b,i)=>b.display_id=boxNumber(b,i));
  const frame = copy(base); change(frame);
  if (!force && annotationEqual(frame,base)) return;
  if (frame.status === current().status) frame.status = "draft";
  checkpoint(); applyScope(frame);
  markDirty(); renderAnnotations();
}
let historyQueue = Promise.resolve();
function undo(redo = false) {
  historyQueue = historyQueue.then(() => restoreAction(redo)).catch(error => toast(error.message,true));
  return historyQueue;
}
async function restoreAction(redo) {
  stopPlay();
  const from = redo ? S.redo : S.undo, to = redo ? S.undo : S.redo;
  if (!from.length || S.loading) return;
  const entry = from.at(-1), expected = redo ? "before" : "after", wanted = redo ? "after" : "before";
  S.historyBusy = true;
  try {
    if (entry.video !== S.video.id) await openVideo(entry.video);
    if ((S.state.source_id || "") !== (redo ? entry.sourceBefore : entry.sourceAfter)
        || entry.changes.some(c=>!annotationEqual(S.state.frames[c.key],c[expected]))) {
      throw new Error("The annotations changed in another session: the local history cannot overwrite them.");
    }
    const source = redo ? entry.sourceAfter : entry.sourceBefore;
    if ((S.state.source_id || "") !== source) await loadSource(source,false);
    for (const change of entry.changes) {
      if (change[wanted] === undefined) delete S.state.frames[change.key];
      else S.state.frames[change.key] = copy(change[wanted]);
    }
    S.unitMode = redo ? entry.modeAfter : entry.modeBefore; $("unitMode").value = S.unitMode;
    from.pop(); to.push(entry); markDirty();
    if (S.frame !== entry.frame) await goFrame(entry.frame);
    S.selected = new Set(entry.selected.filter(id=>currentBoxes().some(b=>b.id===id)));
    toast(redo ? "Change redone" : "Change undone");
  } finally { S.historyBusy = false; renderAnnotations(); }
}
function updateCatalogProgress() {
  if (!S.video) return;
  const row = S.catalog.find(v => v.id === S.video.id);
  row.reviewed = Object.values(S.state.frames).filter(f => f.status === "reviewed").length;
  row.drafts = Object.values(S.state.frames).filter(f => f.status === "draft").length;
  renderLibrary();
}
function eventFrameCount(video) {
  const events = [...video.events].sort((a, b) => a.start - b.start);
  let count = 0, end = -1;
  for (const e of events) { count += Math.max(0, e.end - Math.max(e.start, end + 1) + 1); end = Math.max(end, e.end); }
  return count || video.num_frames;
}
const STALE_SERVER = "The server is an older version: restart it to get the base VTCD planes and their count.";
// Class from the annotated events, never from the file name; videos without events go last.
const annotatedClass = v => [...new Set(v.events.map(e => e.class_name))].sort().join(" + ") || "￿";
function sortVideos(videos, mode) {
  const order = new Map(S.catalog.map((v, i) => [v.id, i]));
  const keys = {
    class: v => annotatedClass(v), "events-desc": v => -v.events.length, "events-asc": v => v.events.length,
    "planes-desc": v => -(v.planes ?? 0), "planes-asc": v => v.planes ?? 0,
  };
  const key = keys[mode];
  if (!key) return videos.sort((a, b) => order.get(a.id) - order.get(b.id));
  const name = (a, b) => a.video_id.localeCompare(b.video_id, "en", {numeric: true});
  return videos.sort((a, b) => {
    const x = key(a), y = key(b);
    const primary = typeof x === "string" ? x.localeCompare(y, "en") : x - y;
    return primary || (mode === "class" ? name(a, b) : 0) || order.get(a.id) - order.get(b.id);
  });
}
function renderLibrary() {
  const search = $("search").value.toLowerCase(), dataset = $("dataset").value, filter = $("filter").value;
  const videos = S.catalog.filter(v => (!dataset || dataset === v.dataset)
    && (!search || `${v.video_id} ${v.events.map(e => e.class_name).join(" ")}`.toLowerCase().includes(search))
    && (filter !== "events" || v.events.length) && (filter !== "proposals" || v.sources)
    && (filter !== "started" || v.reviewed || v.drafts));
  sortVideos(videos, $("sort").value);
  $("libraryCount").textContent = videos.length;
  $("totalProgress").textContent = `${S.catalog.reduce((sum, v) => sum + v.reviewed, 0)} reviewed`;
  $("videoList").replaceChildren();
  for (const v of videos) {
    const card = el("button", {class: `video-card${v.id === S.video?.id ? " active" : ""}`, "data-video": v.id});
    card.append(el("span", {class: "video-name"}, v.video_id.replace("_x264", "")));
    const detail = el("span", {class: "video-detail"});
    const planes = v.planes ? ` · ${v.planes} ${v.planes === 1 ? "plane" : "planes"}` : "";
    detail.append(el("span", {title: annotatedClass(v).replace("￿", "No events")},
      `${v.dataset === "ucf_crime" ? "UCF" : "MIVIA"} · ${v.events.length} ${v.events.length === 1 ? "event" : "events"}${planes}`));
    detail.append(el(v.sources ? "em" : "span", {}, v.available === false ? "File missing" : v.reviewed ? `${v.reviewed} ✓` : v.sources ? "VTCD" : "From scratch"));
    card.append(detail);
    if (v.reviewed) {
      const progress = el("div", {class: "video-progress"}), bar = el("span");
      bar.style.width = `${Math.min(100, 100 * v.reviewed / eventFrameCount(v))}%`;
      progress.append(bar); card.append(progress);
    }
    card.onclick = () => openVideo(v.id).catch(e => toast(e.message, true));
    $("videoList").append(card);
  }
  if (!videos.length) $("videoList").append(el("p", {class: "empty"}, "No video matches the filters."));
}
async function openVideo(id) {
  closeBoxMenu();
  const openRequest = ++S.openId;
  stopPlay(); await flush();
  if (openRequest !== S.openId) return;
  S.loadId++; S.sourceRequest++; S.loading = true;
  const data = await api(`/api/video/${encodeURIComponent(id)}`);
  if (openRequest !== S.openId) return;
  S.video = data.video; S.state = data.state; S.sources = data.sources; S.candidates = data.candidates;
  S.grid = S.defaultGrid = data.grid;
  resetFrameBuffer();
  S.selected.clear(); S.proposals = {}; S.units = []; S.dirty = 0; S.saved = 0; S.zoom = 1;
  S.unitMode = "frames"; $("unitMode").value = "frames";
  if (!$("reviewer").value && S.state.reviewer) $("reviewer").value = S.state.reviewer;
  $("videoTitle").textContent = S.video.video_id;
  $("videoTitle").title = S.video.video_id;
  $("datasetLabel").textContent = `${S.video.dataset === "ucf_crime" ? "UCF-CRIME" : "MIVIA-CRIME"} / TEST`;
  $("videoMeta").textContent = `${S.video.width} × ${S.video.height} px · ${S.video.fps.toFixed(2)} fps · ${S.video.num_frames.toLocaleString("en")} frames`;
  $("scrubber").max = S.video.num_frames - 1; $("frameNumber").max = S.video.num_frames - 1;
  $("durationLabel").textContent = time(S.video.num_frames - 1);
  renderSources(); renderEvents(); renderLibrary();
  if (!Object.keys(S.state.frames).length && data.preferred_source) await loadSource(data.preferred_source,false);
  else if (S.state.source_id) await loadSource(S.state.source_id, false);
  else if (!S.state.revision && S.sources.length) await loadSource(S.sources[0].id, false);
  else await loadSource("", false);
  showSave(S.state.revision ? `Saved · revision ${S.state.revision}` : "No changes");
  const first = S.state.revision ? S.state.last_frame : keyframeSource() ? S.units[0]?.start || 0 : S.video.events[0]?.start || 0;
  await goFrame(first);
  if(openRequest===S.openId)startPreload();
}
function renderSources() {
  const select = $("sourceSelect"); select.replaceChildren(el("option", {value: ""}, "No proposal / work from scratch"));
  for (const source of S.sources) select.append(el("option", {value: source.id}, source.label));
  select.value = S.state.source_id || "";
  const chosen = S.sources.find(s => s.id === select.value);
  $("sourceInfo").textContent = chosen
    ? keyframeSource() ? `${chosen.metadata.sorgente} · ${Object.keys(S.proposals).length} sparse samples. Boxes to review; no extension to the missing frames.` : `${chosen.metadata.sorgente || chosen.metadata.method || chosen.label} · ${chosen.metadata.grid?.join(" × ") || "Boxes in native pixels"} · ID ${chosen.id.slice(0, 10)}`
    : (S.units.length ? `Base VTCD planes: ${S.units.length} planes on the events, no proposed boxes. ` : "")
      + (S.candidates.length ? `${S.candidates.length} local sources available: choose Import.` : "You can annotate from scratch or import a new configuration.");
  $("sourceSamples").hidden = !keyframeSource();
}
async function loadSource(id, dirty = true) {
  closeBoxMenu();
  const request = ++S.sourceRequest, videoId = S.video.id;
  // Without a source: base VTCD planes from the protocol, no proposals.
  const view = id || !S.staleServer ? await api(endpoint(id ? `/source/${id}` : "/units")) : {frames:{},units:[]};
  if (request !== S.sourceRequest || videoId !== S.video.id) return;
  if (dirty && id !== (S.state.source_id || "")) checkpoint();
  const hadUnits = S.units.length > 0;
  S.state.source_id = id;
  S.proposals = view.frames; S.units = (view.units || []).filter(u=>!keyframeSource() || u.known);
  $("unitMode").querySelector('[value="planes"]').textContent=keyframeSource()?"Source samples":"VTCD planes";
  S.grid = view.grid || S.defaultGrid;
  $("unitMode").disabled = !S.units.length;
  S.unitMode = S.units.length ? (hadUnits ? $("unitMode").value : "planes") : "frames";
  $("unitMode").value = S.unitMode;
  if (dirty) markDirty();
  renderSources(); renderAnnotations();
  if (S.units.length && !view.complete && !keyframeSource()) toast("Stretches without mapping are flagged: no plane is invented.");
}
function renderEvents() {
  $("eventBands").replaceChildren(); $("eventChips").replaceChildren();
  S.video.events.forEach((event, index) => {
    const band = el("span", {class: "event-band", title: `${event.class_name} · ${event.start}–${event.end}`});
    band.style.left = `${event.start / S.video.num_frames * 100}%`;
    band.style.width = `${(event.end - event.start + 1) / S.video.num_frames * 100}%`;
    $("eventBands").append(band);
    const chip = el("button", {"data-event": index}, `${event.class_name} · ${time(event.start)}–${time(event.end)}`);
    chip.onclick = () => navigate(event.start); $("eventChips").append(chip);
  });
}
function renderReviewBands() {
  const entries = Object.entries(S.state.frames).sort((a, b) => Number(a[0]) - Number(b[0]));
  const groups = [];
  for (const [key, frame] of entries) {
    const n = Number(key), last = groups.at(-1);
    if (last && last.end + 1 === n && last.status === frame.status) last.end = n;
    else groups.push({start: n, end: n, status: frame.status});
  }
  $("reviewBands").replaceChildren();
  for (const group of groups) {
    const band = el("span", {class: `review-band ${group.status}`});
    band.style.left = `${group.start / S.video.num_frames * 100}%`;
    band.style.width = `${(group.end - group.start + 1) / S.video.num_frames * 100}%`;
    $("reviewBands").append(band);
  }
}
function fitCanvas() {
  if (!S.video) return;
  const area = $("viewport");
  const scale = Math.min((area.clientWidth - 8) / S.video.width, (area.clientHeight - 8) / S.video.height) * S.zoom;
  const width = Math.max(1, S.video.width * scale), height = Math.max(1, S.video.height * scale);
  $("canvas").setAttribute("viewBox", `0 0 ${S.video.width} ${S.video.height}`);
  $("canvas").setAttribute("width", width); $("canvas").setAttribute("height", height);
  $("canvasWrap").style.width = `${width}px`; $("canvasWrap").style.height = `${height}px`;
  $("frameImage").setAttribute("width", S.video.width); $("frameImage").setAttribute("height", S.video.height);
  $("zoomFit").textContent = S.zoom === 1 ? "Fit" : `${Math.round(S.zoom * 100)}%`;
  renderBoxes();
}
function setZoom(value) {
  if (!S.video || S.gesture) return;
  const area = $("viewport"), bounds = area.getBoundingClientRect(), old = $("canvas").getBoundingClientRect();
  const x = (bounds.left + area.clientWidth / 2 - old.left) / old.width;
  const y = (bounds.top + area.clientHeight / 2 - old.top) / old.height;
  S.zoom = Math.max(.5,Math.min(8,value)); fitCanvas();
  if (S.zoom === 1) { area.scrollTo(0,0); return; }
  const next = $("canvas").getBoundingClientRect();
  area.scrollLeft += next.left + x * next.width - bounds.left - area.clientWidth / 2;
  area.scrollTop += next.top + y * next.height - bounds.top - area.clientHeight / 2;
}
async function goFrame(frame) {
  closeBoxMenu();
  if (!S.video) return;
  frame = Math.max(0, Math.min(S.video.num_frames - 1, Math.round(frame)));
  const request = ++S.loadId;
  S.loading = true; S.gesture = null; $("gestureLayer").replaceChildren();
  $("canvasMessage").textContent = `Loading frame ${frame}…`; $("canvasMessage").hidden = false;
  $("confirmFrame").disabled = true;
  try {
    const blob = await frameBlob(frame);
    if (request !== S.loadId) return;
    const objectUrl = URL.createObjectURL(blob);
    const probe = new Image(); probe.src = objectUrl; await probe.decode();
    if (request !== S.loadId) { URL.revokeObjectURL(objectUrl); return; }
    $("frameImage").setAttribute("href", objectUrl);
    if (S.objectUrl) URL.revokeObjectURL(S.objectUrl);
    S.objectUrl = objectUrl; S.frame = frame; S.selected.clear(); S.loading = false;
    $("canvasWrap").style.visibility = "visible"; $("canvasMessage").hidden = true;
    $("scrubber").value = frame; $("frameNumber").value = frame;
    $("timeLabel").textContent = time(frame);
    $("framePosition").textContent = `Frame ${frame} / ${S.video.num_frames - 1} · native pixels`;
    fitCanvas(); renderAnnotations();warmFrames();
  } catch (error) {
    if (request !== S.loadId) return;
    S.loading = false; stopPlay();
    $("canvasMessage").textContent = error.message; $("canvasWrap").style.visibility = "hidden";
    // Blocks edits when the requested frame was not actually shown.
    S.loading = true; toast(error.message, true);
  }
}
function navigate(frame) { stopPlay(); goFrame(frame); }
function stepTarget(amount) {
  if (S.unitMode !== "planes" || !S.units.length) return Math.max(0,Math.min(S.video.num_frames-1,S.frame+amount));
  const index = S.units.findIndex(u => u.start <= S.frame && S.frame <= u.end);
  if (index < 0) {
    const next = amount > 0 ? S.units.find(u => u.start > S.frame) : [...S.units].reverse().find(u => u.end < S.frame);
    return next?.start ?? S.frame;
  }
  return S.units[Math.max(0,Math.min(S.units.length-1,index+amount))].start;
}
function renderBoxes() {
  if (!S.video || !S.state) return;
  renderGrid();
  const layer = $("boxLayer"); layer.replaceChildren(); $("proposalLayer").replaceChildren();$("tokenLayer").replaceChildren();
  const scale = S.video.width / Math.max(1, $("canvas").getBoundingClientRect().width);
  const rect = (box, cls) => {
    const [x0, y0, x1, y1] = box.xyxy;
    return svg("rect", {x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1, class: cls, "data-box": box.id});
  };
  const display = S.gesture?.boxes || currentBoxes();
  if (!S.hidden) for (const original of S.proposals[String(S.frame)] || []) {
    const edited = display.find(b=>b.source_id===original.source_id && b.window_id===original.window_id
      && b.plane===original.plane && b.source_box===original.source_box);
    const differs = !edited || original.xyxy.some((v,i)=>v!==edited.xyxy[i]);
    if (differs) $("proposalLayer").append(rect(original, "proposal-box"));
  }
  if(S.tokenVisible)for(const box of display){
    const token=tokenProjection(box);if(token)$("tokenLayer").append(rect({...box,xyxy:token.pixel_xyxy},"token-box"));
  }
  const handles=[];
  for (const [index, box] of display.entries()) {
    const selected = S.selected.has(box.id);
    const outline = rect(box, `box${selected ? " selected" : ""}`);
    outline.style.stroke = boxColor(box); layer.append(outline);
    const caption = svg("text", {x: box.xyxy[0] + 3 * scale, y: Math.max(12 * scale, box.xyxy[1] - 4 * scale),
      "font-size": 12 * scale}, String(boxNumber(box,index)));
    caption.style.fill = boxColor(box); layer.append(caption);
    if (selected && S.selected.size === 1) {
      const [x0, y0, x1, y1] = box.xyxy, midX = (x0 + x1 + 1) / 2, midY = (y0 + y1 + 1) / 2;
      const points = {nw: [x0, y0], n: [midX, y0], ne: [x1 + 1, y0], e: [x1 + 1, midY],
        se: [x1 + 1, y1 + 1], s: [midX, y1 + 1], sw: [x0, y1 + 1], w: [x0, midY]};
      for (const [direction, [x, y]] of Object.entries(points)) {
        const handle = svg("rect", {x: x - 3.5 * scale, y: y - 3.5 * scale,
          width: 7 * scale, height: 7 * scale, class: "handle", "data-box": box.id, "data-handle": direction});
        handle.style.cursor = `${direction}-resize`; handles.push(handle);
      }
    }
  }
  // Selection handles must receive clicks even under other boxes.
  layer.append(...handles);
}
function renderAnnotations() {
  if (!S.video || !S.state) return;
  renderBoxes(); renderReviewBands();
  const frame = current(), exists = String(S.frame) in S.state.frames, classes = classesAt(S.frame);
  const unit = planeScope(), mapped = activeUnit(), conflict = unitConflict(unit);
  $("unitMode").disabled = !S.units.length;
  $("confirmFrame").replaceChildren(document.createTextNode("Confirm and advance "), el("kbd",{},shortcutLabel("confirm")));
  $("confirmFrame").title = `Mark the ${unit ? "plane" : "frame"} as reviewed, save and move to the next. ${shortcutLabel("confirm")}`;
  $("uniformUnit").hidden = !conflict;
  $("unitInfo").textContent = unit
    ? keyframeSource() ? `Sample ${S.units.indexOf(unit)+1}/${S.units.length} · frame ${unit.start}` : `Plane ${unit.plane} · ${S.units.indexOf(unit)+1}/${S.units.length} · frames ${unit.start}–${unit.end}`
    : mapped && !mapped.known ? `Frames ${mapped.start}–${mapped.end}: plane mapping not available` : "Single-frame review";
  $("unitInfo").title = unit ? `${unit.end-unit.start+1} frames with the same boxes · ${unit.window_id}` : "";
  $("prevFrame").title = unit ? "Previous plane (←)" : "Previous frame (←)";
  $("nextFrame").title = unit ? "Next plane (→)" : "Next frame (→)";
  $("copyPreviousUnit").hidden = !unit || keyframeSource();
  $("applyPlanes").hidden = !unit || keyframeSource();
  $("useSourceSample").disabled = !S.proposals[String(S.frame)]?.length || S.loading;
  $("rangeButton").disabled = !!unit;
  $("rangeButton").title = unit ? "To interpolate or copy frames, choose Frames mode." : "";
  $("excludeFrame").textContent = unit ? "Exclude plane" : "Exclude frame";
  $("planeBands").replaceChildren();
  if (S.unitMode === "planes") for (const u of S.units) {
    const tick = el("span",{class:`plane-tick${u===unit?" active":""}`});
    tick.style.left = `${100*u.start/S.video.num_frames}%`;
    tick.style.width = u===unit ? `${100*(u.end-u.start+1)/S.video.num_frames}%` : "0";
    $("planeBands").append(tick);
  }
  const label = conflict ? "Different corrections" : !exists ? "To review" : {draft: "Draft", reviewed: "Reviewed", excluded: "Excluded"}[frame.status];
  $("reviewBadge").textContent = label; $("reviewBadge").className = `badge ${exists ? frame.status : ""}`;
  $("confirmFrame").disabled = !writable();
  $("frameSummary").textContent = `${frame.boxes.length} boxes · ${unit ? `plane ${unit.plane} · ` : ""}${classes.join(" + ") || "Outside the events"}`;
  $("actionHint").textContent = frame.status === "reviewed" ? "Frame reviewed. An edit moves it back to draft."
    : unit ? `Constant corrections on frames ${unit.start}–${unit.end}.` : "Check the whole scene before confirming.";
  if (unit && frame.status === "reviewed") $("actionHint").textContent = `Plane reviewed; constant boxes on frames ${unit.start}–${unit.end}.`;
  if (conflict) $("actionHint").textContent = "Existing per-frame corrections are protected: choose how to proceed.";
  $("actionHint").classList.toggle("warning",conflict);
  $("frameNote").value = frame.note || "";
  $("boxCount").textContent = frame.boxes.length;
  $("reassignBoxIds").disabled = !frame.boxes.length || S.loading || S.historyBusy;
  $("reassignBoxIds").title = `Renumber all boxes from 1 in list order, in the current ${unit ? "plane" : "frame"}. Ctrl+Z undoes.`;
  $("eventLabel").textContent = classes.join(" + ") || "No event in this frame";
  document.querySelectorAll("[data-event]").forEach(node => {
    const e = S.video.events[Number(node.dataset.event)]; node.classList.toggle("active", e.start <= S.frame && S.frame <= e.end);
  });
  $("boxList").replaceChildren();
  frame.boxes.forEach((box, i) => {
    const row = el("button", {class: `box-row${S.selected.has(box.id) ? " active" : ""}`});
    const swatch = el("span", {class: "swatch"}, "□"); swatch.style.color=boxColor(box); row.append(swatch);
    const title = el("span", {}, String(boxNumber(box,i)));
    title.append(el("span", {class: "box-subtitle"}, `${box.class_name}${box.class_ambiguous ? " · choose class" : ""}`));
    row.append(title);
    row.onclick = event => { selectBox(box.id, event.shiftKey); renderAnnotations(); };
    row.oncontextmenu = event => showBoxMenu(event,box.id);
    $("boxList").append(row);
  });
  if (!frame.boxes.length) $("boxList").append(el("div", {class: "empty"}, "No boxes. Press B and drag on the scene."));
  renderProperties();
  $("undo").disabled = !S.undo.length; $("redo").disabled = !S.redo.length;
  $("deleteBox").disabled = !S.selected.size;
}
function renderProperties() {
  const box = S.selected.size === 1 ? currentBoxes().find(b => S.selected.has(b.id)) : null;
  $("noSelection").hidden = !!box; $("boxProperties").hidden = !box;
  $("noSelection").textContent = S.selected.size > 1 ? `${S.selected.size} boxes selected: move and delete as a group.` : "Select a box to edit it.";
  if (!box) return;
  const token=tokenProjection(box);
  $("tokenCoordinates").textContent=token?`Token-space: [${token.xyxy.join(", ")}) · ${S.grid.columns}×${S.grid.rows} · end bounds excluded`:"";
  $("trackId").value = box.track_id || "";
  $("eventClass").replaceChildren();
  if (box.class_ambiguous) $("eventClass").append(el("option", {value: ""}, "Choose the event…"));
  for (const name of classesAt(S.frame)) $("eventClass").append(el("option", {value: name}, name));
  $("eventClass").value = box.class_ambiguous ? "" : box.class_name;
  ["x0", "y0", "x1", "y1"].forEach((name, i) => {
    $(name).value = box.xyxy[i]; $(name).max = i % 2 ? S.video.height - 1 : S.video.width - 1;
  });
}
function selectBox(id, multi = false) {
  if (!multi) { if (!S.selected.has(id)) S.selected = new Set([id]); }
  else if (S.selected.has(id)) S.selected.delete(id); else S.selected.add(id);
}
function setTool(tool) {
  S.tool = tool; $("drawTool").classList.toggle("active", tool === "draw");
  $("selectTool").classList.toggle("active", tool === "select"); $("canvas").classList.toggle("drawing", tool === "draw");
}
function point(event) {
  const p = new DOMPoint(event.clientX, event.clientY).matrixTransform($("canvas").getScreenCTM().inverse());
  return [Math.max(0, Math.min(S.video.width, p.x)), Math.max(0, Math.min(S.video.height, p.y))];
}
function shiftBoxes(boxes, dx, dy) {
  const selected = boxes.filter(b => S.selected.has(b.id));
  if (!selected.length) return;
  dx = Math.round(Math.max(-Math.min(...selected.map(b => b.xyxy[0])), Math.min(dx,
    S.video.width - 1 - Math.max(...selected.map(b => b.xyxy[2])))));
  dy = Math.round(Math.max(-Math.min(...selected.map(b => b.xyxy[1])), Math.min(dy,
    S.video.height - 1 - Math.max(...selected.map(b => b.xyxy[3])))));
  for (const box of selected) box.xyxy = box.xyxy.map((v, i) => v + (i % 2 ? dy : dx));
}
let lastCanvasTap=null;
$("canvas").addEventListener("pointerdown", event => {
  if(event.button!==0)return;
  stopPlay();if(!writable())return;
  commitEditor(); $("canvas").focus({preventScroll:true});
  const start=point(event),target=event.target,id=target.getAttribute("data-box");
  // Selection redraws the SVG nodes: detect a double click even when the first
  // clicked box was replaced in the DOM before the second click.
  const double=!target.hasAttribute("data-handle")&&lastCanvasTap&&lastCanvasTap.video===S.video.id&&lastCanvasTap.frame===S.frame
    &&event.timeStamp-lastCanvasTap.time<500&&Math.hypot(event.clientX-lastCanvasTap.x,event.clientY-lastCanvasTap.y)<6;
  if(double){S.gesture={kind:"double",start};lastCanvasTap=null;}
  else if(id){
    setTool("select");selectBox(id,event.shiftKey);
    S.gesture={kind:target.getAttribute("data-handle")||"move",start,id,
      original:copy(currentBoxes()),boxes:copy(currentBoxes())};renderAnnotations();
  }else if(S.tool==="draw"){
    if(!classesAt(S.frame).length){toast("Draw in the frames of the annotated events.",true);return;}
    S.gesture={kind:"draw",start};
  }else{S.selected.clear();renderAnnotations();S.gesture={kind:"tap",start};}
  $("canvas").setPointerCapture(event.pointerId); event.preventDefault();
});
$("canvas").addEventListener("pointermove", event => {
  const g = S.gesture; if (!g) return;
  const p = point(event), dx = Math.round(p[0] - g.start[0]), dy = Math.round(p[1] - g.start[1]);
  g.moved = Math.abs(dx) + Math.abs(dy) > 0;
  if(g.kind==="tap"||g.kind==="double")return;
  if (g.kind === "draw") {
    g.end = p;
    $("gestureLayer").replaceChildren(svg("rect", {x: Math.min(g.start[0], p[0]), y: Math.min(g.start[1], p[1]),
      width: Math.abs(p[0] - g.start[0]), height: Math.abs(p[1] - g.start[1]), class: "box selected"}));
  } else {
    g.boxes = copy(g.original);
    if (g.kind === "move") shiftBoxes(g.boxes, dx, dy);
    else {
      const b = g.boxes.find(b => b.id === g.id), q = b.xyxy;
      if (g.kind.includes("w")) q[0] = Math.max(0, Math.min(q[2], q[0] + dx));
      if (g.kind.includes("e")) q[2] = Math.max(q[0], Math.min(S.video.width - 1, q[2] + dx));
      if (g.kind.includes("n")) q[1] = Math.max(0, Math.min(q[3], q[1] + dy));
      if (g.kind.includes("s")) q[3] = Math.max(q[1], Math.min(S.video.height - 1, q[3] + dy));
    }
    renderBoxes();
  }
});
$("canvas").addEventListener("pointerup", event => {
  const g = S.gesture; if (!g) return;
  S.gesture = null; $("gestureLayer").replaceChildren();
  if ($("canvas").hasPointerCapture(event.pointerId)) $("canvas").releasePointerCapture(event.pointerId);
  if(g.kind==="double"){
    if(!g.moved){S.lastDouble=event.timeStamp;baseBoxAt(event);}return;
  }
  if (!g.moved) {
    lastCanvasTap=g.kind==="move"||g.kind==="tap"||g.kind==="draw"
      ?{video:S.video.id,frame:S.frame,time:event.timeStamp,x:event.clientX,y:event.clientY}:null;
    renderBoxes(); return;
  }
  lastCanvasTap=null;if(g.kind==="tap")return;
  if (g.kind === "draw") {
    const end = g.end || point(event), classes = classesAt(S.frame);
    const [x0, y0] = [Math.floor(Math.min(g.start[0], end[0])), Math.floor(Math.min(g.start[1], end[1]))];
    const [x1, y1] = [Math.min(S.video.width - 1, Math.ceil(Math.max(g.start[0], end[0])) - 1),
      Math.min(S.video.height - 1, Math.ceil(Math.max(g.start[1], end[1])) - 1)];
    if (x1 < x0 || y1 < y0) return;
    const box = {id: crypto.randomUUID(), xyxy: [x0, y0, x1, y1], class_name: classes[0],
      class_ambiguous: classes.length > 1, display_id:nextBoxNumber(), track_id: "", origin: "manual"};
    S.selected = new Set([box.id]); editFrame(f => f.boxes.push(box));
  } else editFrame(f => f.boxes = g.boxes);
});
$("canvas").addEventListener("pointercancel", () => { S.gesture = null; $("gestureLayer").replaceChildren(); renderBoxes(); });
function deleteSelected() {
  if (!S.selected.size) return;
  editFrame(f => f.boxes = f.boxes.filter(b => !S.selected.has(b.id))); S.selected.clear(); renderAnnotations();
}
function hideProposals() {
  S.hidden = !S.hidden; $("hideProposals").textContent = S.hidden ? "○ Original hidden" : "◉ Original";
  $("hideProposals").setAttribute("aria-pressed", String(!S.hidden)); renderBoxes();
}
function pendingFrame(after = S.frame) {
  if (S.unitMode === "planes" && S.units.length) {
    const currentIndex = S.units.findIndex(u => u.start <= after && after <= u.end);
    for (let offset = 1; offset <= S.units.length; offset++) {
      const unit = S.units[(Math.max(-1,currentIndex)+offset) % S.units.length];
      for (let frame = unit.start; frame <= unit.end; frame++) {
        if (!["reviewed","excluded"].includes(S.state.frames[String(frame)]?.status)) return unit.start;
      }
    }
    return null;
  }
  const events = [...S.video.events].sort((a, b) => a.start - b.start);
  const ranges = events.length ? events : [{start: 0, end: S.video.num_frames - 1}];
  for (const wrap of [false, true]) {
    for (const event of ranges) {
      const first = Math.max(event.start, wrap ? 0 : after + 1), last = Math.min(event.end, wrap ? after : S.video.num_frames - 1);
      for (let n = first; n <= last; n++) {
        const status = S.state.frames[String(n)]?.status;
        if (status !== "reviewed" && status !== "excluded") return n;
      }
    }
  }
  return null;
}
async function confirmFrame() {
  stopPlay();if(!applyBoxMenu())return;commitEditor();
  if (!writable()) return;
  if (!scopeReady()) return;
  if (!$("reviewer").value.trim()) { $("reviewer").focus(); toast("Enter the reviewer name.", true); return; }
  if (currentBoxes().some(b => b.class_ambiguous)) { toast("Choose the event for the boxes with an ambiguous class.", true); return; }
  checkpoint(); applyScope({...copy(current()), status: "reviewed"});
  markDirty(); renderAnnotations();
  try { await flush(); }
  catch (error) { toast(error.message, true); return; }
  const next = pendingFrame();
  if (next !== null) await goFrame(next); else toast(keyframeSource()&&S.unitMode==="planes"?"All source samples reviewed. The frames that were not sampled still need annotating.":"All event stretches in this video reviewed.");
}
function jumpEvent(direction) {
  if (!S.video) return;
  const starts = [...new Set(S.video.events.map(e => e.start))].sort((a, b) => a - b);
  const next = direction > 0 ? starts.find(n => n > S.frame) : starts.reverse().find(n => n < S.frame);
  if (next !== undefined) navigate(next);
}
function updateProperty(name, value) {
  if (S.selected.size !== 1 || !writable()) return;
  editFrame(f => { const box = f.boxes.find(b => S.selected.has(b.id)); box[name] = value;
    if (name === "class_name") box.class_ambiguous = false; });
}
for (const [id, key] of [["eventClass", "class_name"], ["trackId", "track_id"]]) {
  $(id).onchange = () => { if (key === "class_name" && !$(id).value) return; updateProperty(key, $(id).value); };
}
for (const [i, id] of ["x0", "y0", "x1", "y1"].entries()) $(id).onchange = () => {
  const box = currentBoxes().find(b => S.selected.has(b.id)); if (!box) return;
  const xyxy = [...box.xyxy]; xyxy[i] = Number($(id).value);
  if (!xyxy.every(Number.isInteger) || xyxy[0] < 0 || xyxy[1] < 0 || xyxy[2] >= S.video.width || xyxy[3] >= S.video.height || xyxy[0] > xyxy[2] || xyxy[1] > xyxy[3]) {
    toast("Invalid coordinates: the edges must stay inside the frame.", true); renderProperties(); return;
  }
  updateProperty("xyxy", xyxy);
};
async function importSource(body) {
  await flush();
  const result = await api(endpoint("/import"), body);
  S.sources = result.sources;
  S.catalog.find(v => v.id === S.video.id).sources = S.sources.length;
  await loadSource(result.id); $("importDialog").close(); renderLibrary(); toast("Source checked and imported.");
}
$("importButton").onclick = () => {
  if (!S.video) return;
  stopPlay(); $("importError").textContent = ""; $("candidateList").replaceChildren();
  for (const candidate of S.candidates) {
    const button = el("button", {type: "button", class: "wide"}, `↓ Import local source · ${candidate.label}`);
    button.onclick = async () => { button.disabled = true;
      try { await importSource({candidate: candidate.index}); }
      catch (e) { $("importError").textContent = e.message; }
      finally { button.disabled = false; } };
    $("candidateList").append(button);
  }
  $("importDialog").showModal();
};
$("importForm").onsubmit = async event => {
  event.preventDefault(); const button = event.submitter;
  const meta = $("metadataFile").files[0], csv = $("csvFile").files[0];
  if (!meta || !csv || !$("sourceLabel").value.trim()) { $("importError").textContent = "Choose both files and name the source."; return; }
  button.disabled = true;
  try { await importSource({metadata: await meta.text(), csv: await csv.text(), label: $("sourceLabel").value}); }
  catch (e) { $("importError").textContent = e.message; }
  finally { button.disabled = false; }
};
$("rangeButton").onclick = () => {
  if (!writable()) return;
  const event = S.video.events.find(e => e.start <= S.frame && S.frame <= e.end);
  if (!event) { toast("Choose a frame inside an event.", true); return; }
  $("rangeStart").value = S.frame; $("rangeEnd").value = Math.min(S.frame + 30, event.end);
  $("rangeError").textContent = ""; $("rangeDialog").showModal();
};
$("rangeForm").onsubmit = event => {
  event.preventDefault();
  try {
    const start = Number($("rangeStart").value), end = Number($("rangeEnd").value), mode = $("rangeMode").value;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end >= S.video.num_frames) throw new Error("Enter two valid frames in increasing order.");
    if (end - start > 10000) throw new Error("Work on at most 10,000 frames per operation.");
    const covering = S.video.events.filter(e => e.start <= start && end <= e.end);
    if (!covering.length) throw new Error("The range must lie inside a single event.");
    const boxes = mode === "copy" ? currentBoxes().filter(b => S.selected.has(b.id)) : S.state.frames[String(start)]?.boxes;
    if (!boxes?.length) throw new Error("Select the boxes to copy, or correct the start frame first.");
    if (mode === "copy" && (S.frame < start || S.frame > end)) throw new Error("The copy range must include the current frame.");
    if (boxes.some(b => b.class_ambiguous || !covering.some(e => e.class_name === b.class_name))) throw new Error("Choose classes compatible with the event of the range.");
    if (mode === "interpolate") {
      if (boxes.some(b => !b.track_id) || new Set(boxes.map(b => b.track_id)).size !== boxes.length) throw new Error("Assign a unique track to each box of the start frame.");
    }
    const ends = S.state.frames[String(end)]?.boxes || [];
    const pairs = boxes.map(b => [b, ends.filter(e => e.track_id === b.track_id)]);
    if (mode === "interpolate" && pairs.some(([b, matched]) => matched.length !== 1 || matched[0].class_name !== b.class_name)) throw new Error("The end frame must contain a unique match for each track, with the same event.");
    const generated = {}; let count = 0, protectedCount = 0;
    for (let frame = start; frame <= end; frame++) {
      if (mode === "copy" && frame === S.frame) continue;
      if (mode === "interpolate" && (frame === start || frame === end)) continue;
      const existing = S.state.frames[String(frame)];
      if (existing && existing.status !== "draft") { protectedCount++; continue; }
      const item = copy(existing || {boxes: [], note: "", status: "draft"});
      item.boxes.forEach((b,i)=>b.display_id=boxNumber(b,i));
      for (const [b, matched] of pairs) {
        const box = copy(b); box.id = crypto.randomUUID();
        box.origin = mode === "copy" ? "propagated" : "interpolated";
        box.derived_from = {start: mode === "copy" ? S.frame : start, end: mode === "copy" ? S.frame : end, box_id: b.id};
        if (mode === "interpolate") box.xyxy = b.xyxy.map((v, i) => Math.round(v + (matched[0].xyxy[i] - v) * (frame - start) / (end - start)));
        if (box.track_id) item.boxes = item.boxes.filter(other => other.track_id !== box.track_id);
        if (!box.display_id || item.boxes.some((b,i)=>boxNumber(b,i)===box.display_id)) box.display_id=Math.max(0,...item.boxes.map(boxNumber))+1;
        item.boxes.push(box);
      }
      generated[String(frame)] = item; count++;
    }
    if (!count) throw new Error("No editable frame: the ends and the frames already reviewed or excluded are protected.");
    checkpoint(); Object.assign(S.state.frames, generated); markDirty(); renderAnnotations();
    $("rangeDialog").close(); toast(`${count} drafts generated${protectedCount ? ` · ${protectedCount} frames protected` : ""}.`);
  } catch (error) { $("rangeError").textContent = error.message; }
};
$("confirmFrame").onclick = confirmFrame;
$("nextPending").onclick = () => { if (!S.video) return; const next = pendingFrame(); next === null ? toast("No frame to review in the events.") : navigate(next); };
$("sourceSelect").onchange = () => loadSource($("sourceSelect").value).catch(e => toast(e.message, true));
$("hideProposals").onclick = hideProposals;
$("selectTool").onclick = () => setTool("select"); $("drawTool").onclick = () => setTool("draw");
$("undo").onclick = () => undo(); $("redo").onclick = () => undo(true); $("deleteBox").onclick = deleteSelected;
$("prevFrame").onclick = () => S.video && navigate(stepTarget(-1)); $("nextFrame").onclick = () => S.video && navigate(stepTarget(1));
$("unitMode").onchange = () => { stopPlay(); S.unitMode = $("unitMode").value; renderAnnotations(); };
$("uniformUnit").onclick = () => {
  const unit = planeScope(); if (!unit) return;
  $("uniformDescription").textContent = `Plane ${unit.plane}, frames ${unit.start}–${unit.end}. The boxes of frame ${S.frame} will be used.`;
  $("uniformDialog").showModal();
};
$("cancelUniform").onclick = () => $("uniformDialog").close();
$("applyUniform").onclick = () => {
  const unit = planeScope(); if (!unit || !writable()) return;
  checkpoint(); applyScope({...copy(current()),status:"draft"},unit);markDirty();renderAnnotations();$("uniformDialog").close();
};
async function copyPreviousUnit() {
  const unit = planeScope(), index = S.units.indexOf(unit);
  stopPlay();commitEditor();
  if (!unit || index < 1) {toast("There is no previous plane to copy.");return;}
  if(!writable()||!scopeReady())return;
  const previous = S.units[index-1];
  if (previous.end+1 !== unit.start || !previous.known) {toast("The previous plane is not contiguous: no copy across gaps or separate events.",true);return;}
  const frame = S.state.frames[String(previous.start)] || {boxes:copy(S.proposals[String(previous.start)]||[])};
  if (unitConflict(previous)) {toast("The previous plane contains different corrections: unify them before copying it.",true);return;}
  if (frame.boxes.some(b=>!classesAt(S.frame).includes(b.class_name))) {toast("The previous plane contains classes incompatible with the current one.",true);return;}
  if(currentBoxes().length){
    const choice=await askChoice("Copy from the previous plane",`Delete the current boxes to import those of the previous plane? The previous one contains ${frame.boxes.length} boxes. The replacement affects the current plane and can be undone.`,[
      {value:"replace",label:"Replace with the previous boxes",primary:true},{value:null,label:"Keep the current boxes"}]);
    if(choice!=="replace")return;
  }
  editFrame(f=>f.boxes=frame.boxes.map(b=>({...copy(b),id:crypto.randomUUID(),origin:"propagated",derived_from:{start:previous.start,end:previous.end,box_id:b.id}})));
};
$("copyPreviousUnit").onclick=copyPreviousUnit;
$("prevEvent").onclick = () => jumpEvent(-1); $("nextEvent").onclick = () => jumpEvent(1); $("play").onclick = togglePlay;
$("frameNumber").onchange = () => { if (Number.isFinite($("frameNumber").valueAsNumber)) navigate($("frameNumber").valueAsNumber); };
let scrubTimer;
$("scrubber").oninput = () => { stopPlay(); clearTimeout(scrubTimer); scrubTimer = setTimeout(() => goFrame(Number($("scrubber").value)), 100); };
$("zoomIn").onclick = () => setZoom(S.zoom * 1.4);
$("zoomOut").onclick = () => setZoom(S.zoom / 1.4);
$("zoomFit").onclick = () => setZoom(1);
$("detailsToggle").onclick = () => {
  const expanded = document.querySelector("main").classList.toggle("show-details");
  $("detailsToggle").textContent = expanded ? "Minimize" : "Details";
  $("detailsToggle").setAttribute("aria-expanded", String(expanded));
};
$("emptyFrame").onclick = () => { editFrame(f => f.boxes = [], true); S.selected.clear(); renderAnnotations(); };
$("frameNote").onchange = () => { const note = $("frameNote").value; editFrame(f => f.note = note); };
$("excludeFrame").onclick = () => {
  if (!writable()) return;
  const note = $("frameNote").value.trim();
  if (!note) { $("frameNote").focus(); toast("Write the reason for the exclusion in the note.", true); return; }
  editFrame(f => { f.status = "excluded"; f.note = note; });
};
$("reviewer").value = localStorage.getItem("forma-reviewer") || "";
$("reviewer").onchange = () => { localStorage.setItem("forma-reviewer", $("reviewer").value.trim()); };
for (const id of ["search", "dataset", "filter"]) $(id).addEventListener(id === "search" ? "input" : "change", renderLibrary);
// Browser-only preference: without storage the name order applies.
try { const saved = localStorage.getItem("forma-sort"); if (saved && $("sort").querySelector(`option[value="${saved}"]`)) $("sort").value = saved; } catch {}
$("sort").addEventListener("change", () => { try { localStorage.setItem("forma-sort", $("sort").value); } catch {} renderLibrary(); });
$("helpButton").onclick = openGuide;
$("closeImport").onclick = () => $("importDialog").close(); $("closeRange").onclick = () => $("rangeDialog").close();
$("closeError").onclick = () => $("errorDialog").close();
$("retrySave").onclick = async () => { $("errorDialog").close(); try { await flush(); } catch (e) { toast(e.message, true); } };
function download(data, name, mime) {
  const url = URL.createObjectURL(new Blob([data], {type: mime})), link = el("a", {href: url, download: name});
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
}
$("recoveryButton").onclick = () => download(JSON.stringify({video: S.video, state: {...S.state, reviewer: $("reviewer").value.trim(), last_frame: S.frame}}, null, 2), `recovery-${S.video.video_id}.json`, "application/json");
$("exportButton").onclick = async () => {
  stopPlay(); const button = $("exportButton"); button.disabled = true; button.textContent = "Exporting…";
  try { await flush(); const response = await fetch("/api/export");
    if (!response.ok) throw new Error((await response.json()).error);
    download(await response.blob(), "spatial-annotations.zip", "application/zip");
    toast("Exported project, sources and only the reviewed frames as ground truth.");
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = "Export project ↗"; }
};
function isAnnotationInput(node) {
  return node?.id === "frameNote" || !!node?.closest("#boxProperties");
}
function commitEditor() {
  const node = document.activeElement;
  if (isAnnotationInput(node)) node.dispatchEvent(new Event("change"));
}
async function saveEdits() { await saveWithReview(); }
$("saveButton").onclick = () => saveEdits().catch(e=>toast(e.message,true));
document.addEventListener("keydown", event => {
  const action=shortcutAction(event);if(!action)return;
  if(action==="escape"&&!$("boxMenu").hidden){event.preventDefault();closeBoxMenu();return;}
  if(action==="escape"&&$("choiceDialog").open){event.preventDefault();$("choiceDialog").dispatchEvent(new Event("cancel",{cancelable:true}));return;}
  if(document.querySelector("dialog[open]")){if(["save","confirm"].includes(action))event.preventDefault();return;}
  const active=document.activeElement,typing=["INPUT","TEXTAREA","SELECT"].includes(active?.tagName)||active?.isContentEditable;
  const nativeUndo=["INPUT","TEXTAREA"].includes(active?.tagName)&&!isAnnotationInput(active);
  if(["save","confirm"].includes(action)){
    event.preventDefault();
    if(action==="save")saveEdits().catch(e=>toast(e.message,true));
    else if(applyBoxMenu()){commitEditor();confirmFrame();}return;
  }
  if(["undo","redo"].includes(action)&&!nativeUndo){
    event.preventDefault();commitEditor();closeBoxMenu();undo(action==="redo");return;
  }
  if(typing)return;
  event.preventDefault();
  if(action==="help"){openGuide();return;}
  if(!S.video)return;
  const actions={
    all:()=>{S.selected=new Set(currentBoxes().map(b=>b.id));renderAnnotations();},
    select:()=>setTool("select"),draw:()=>setTool("draw"),grid:toggleGrid,original:hideProposals,
    tokens:toggleTokenSpace,precision:togglePrecision,unitMode:toggleUnitMode,copy:copyPreviousUnit,
    copyBoxes:copySelectedBoxes,pasteBoxes:pasteCopiedBoxes,applyPlanes:openApplyPlanes,
    previous:()=>navigate(stepTarget(-1)),next:()=>navigate(stepTarget(1)),
    previous10:()=>navigate(stepTarget(-10)),next10:()=>navigate(stepTarget(10)),
    previousEvent:()=>jumpEvent(-1),nextEvent:()=>jumpEvent(1),play:togglePlay,
    zoomIn:()=>setZoom(S.zoom*1.4),zoomOut:()=>setZoom(S.zoom/1.4),zoomFit:()=>setZoom(1),
    delete:deleteSelected,
    escape:()=>{S.gesture=null;S.selected.clear();$("gestureLayer").replaceChildren();renderAnnotations();},
    menu:()=>{const id=[...S.selected][0];if(id){const r=$("canvas").getBoundingClientRect();showBoxMenu({preventDefault(){},clientX:r.left+r.width/2,clientY:r.top+r.height/2},id);}},
  };
  if(action==="move"||action==="move10"){
    const n=action==="move10"?10:1,key=event.key.toLowerCase();
    const dx=key==="arrowright"?n:key==="arrowleft"?-n:0,dy=key==="arrowdown"?n:key==="arrowup"?-n:0;
    if(S.selected.size)editFrame(f=>shiftBoxes(f.boxes,dx,dy));
  } else actions[action]?.();
});
initBoxTools();initCollection();initWorkflow();initPlayback();
window.addEventListener("beforeunload", event => { if (S.dirty !== S.saved) { event.preventDefault(); event.returnValue = ""; } });
new ResizeObserver(() => fitCanvas()).observe($("viewport"));
(async () => {
  try {
    const catalog = await api("/api/catalog"); S.catalog = catalog.videos; S.token = catalog.token;
    // Assets are read from disk, Python is not: a server started before the update does not
    // count the planes. Say so instead of sorting on zeros.
    S.staleServer = S.catalog.some(v => v.planes === undefined);
    for (const option of $("sort").querySelectorAll('option[value^="planes"]')) option.disabled = S.staleServer;
    if (S.staleServer && $("sort").value.startsWith("planes")) $("sort").value = "name";
    if (S.staleServer) toast(STALE_SERVER, true);
    renderLibrary();
    const first = S.catalog.find(v => v.sources && v.events.length) || S.catalog.find(v => v.events.length) || S.catalog[0];
    if (first) await openVideo(first.id);
  } catch (error) { $("canvasMessage").textContent = error.message; toast(error.message, true); }
})();
