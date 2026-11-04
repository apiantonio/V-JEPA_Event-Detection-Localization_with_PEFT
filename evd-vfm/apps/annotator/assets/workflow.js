"use strict";
function askChoice(title,message,choices) {
  stopPlay();closeBoxMenu();
  if($("choiceDialog").open)return Promise.resolve(null);
  $("choiceTitle").textContent=title;$("choiceMessage").textContent=message;$("choiceButtons").replaceChildren();
  return new Promise(resolve=>{
    let done=false;
    const finish=value=>{if(done)return;done=true;$("choiceDialog").close();resolve(value);};
    for(const choice of choices){
      const button=el("button",{class:choice.primary?"primary wide":"wide"},choice.label);
      if(choice.default)button.autofocus=true;
      button.disabled=!!choice.disabled;button.onclick=()=>finish(choice.value);$("choiceButtons").append(button);
    }
    $("choiceDialog").oncancel=event=>{event.preventDefault();finish(null);};$("choiceDialog").showModal();
  });
}
function toggleUnitMode() {
  stopPlay();commitEditor();
  if(!S.units.length){toast("No VTCD plane: the video has no annotated events.");return;}
  S.unitMode=S.unitMode==="planes"?"frames":"planes";$("unitMode").value=S.unitMode;renderAnnotations();warmFrames();
}
function togglePrecision() {
  S.precision=!S.precision;$("canvas").classList.toggle("precision",S.precision);
  $("precisionButton").setAttribute("aria-pressed",String(S.precision));$("precisionButton").classList.toggle("active",S.precision);
}
function toggleTokenSpace() {
  S.tokenVisible=!S.tokenVisible;$("tokenButton").setAttribute("aria-pressed",String(S.tokenVisible));
  $("tokenButton").classList.toggle("active",S.tokenVisible);renderBoxes();
}
const tokenProjections=new Map();
function coveringTokenAxis(start,end,edges) {
  // Same intersection of inclusive pixels and half-open cells as geometry.token_box.
  const cells=[];
  for(let i=0;i<edges.length-1;i++)if(edges[i]<edges[i+1]&&edges[i]<=end&&edges[i+1]>start)cells.push(i);
  return [cells[0],cells.at(-1)+1];
}
function tokenProjection(box) {
  if(!S.grid)return null;
  const key=JSON.stringify([box.xyxy,S.grid.x_edges,S.grid.y_edges]);
  if(tokenProjections.has(key))return tokenProjections.get(key);
  const x=coveringTokenAxis(box.xyxy[0],box.xyxy[2],S.grid.x_edges),y=coveringTokenAxis(box.xyxy[1],box.xyxy[3],S.grid.y_edges);
  const value={xyxy:[x[0],y[0],x[1],y[1]],pixel_xyxy:[S.grid.x_edges[x[0]],S.grid.y_edges[y[0]],S.grid.x_edges[x[1]]-1,S.grid.y_edges[y[1]]-1]};
  if(tokenProjections.size>1000)tokenProjections.clear();tokenProjections.set(key,value);return value;
}
function baseBoxAt(event) {
  stopPlay();if(!writable())return;
  commitEditor();setTool("draw");
  if(!classesAt(S.frame).length){toast("Draw in the frames of the annotated events.",true);return;}
  const [x,y]=point(event),side=Math.max(1,Math.round(Math.min(S.video.width,S.video.height)*.15));
  const x0=Math.max(0,Math.min(S.video.width-side,Math.round(x-side/2))),y0=Math.max(0,Math.min(S.video.height-side,Math.round(y-side/2)));
  const classes=classesAt(S.frame),box={id:crypto.randomUUID(),display_id:nextBoxNumber(),xyxy:[x0,y0,x0+side-1,y0+side-1],class_name:classes[0],class_ambiguous:classes.length>1,track_id:"",origin:"manual"};
  S.selected=new Set([box.id]);editFrame(f=>f.boxes.push(box));
}
async function saveWithReview() {
  stopPlay();if(!applyBoxMenu())return;commitEditor();
  if(!S.state)return;
  const drafts=Object.entries(S.state.frames).filter(([,f])=>f.status==="draft");
  const planes=new Set(drafts.map(([,f])=>f.review_unit?.id).filter(Boolean)).size;
  const singles=drafts.filter(([,f])=>!f.review_unit).length;
  const summary=planes?`${planes} planes and ${singles} single frames (${drafts.length} frames in total)`:`${drafts.length} frames`;
  const choice=await askChoice("Save the annotations",`${S.video.video_id}: reassign all the IDs of the video and save everything? Every annotated plane/frame restarts from 1, in list order, keeping geometry and review state. Drafts in ${summary}: you can also confirm them. Proposals not yet edited stay out; excluded frames stay excluded.`,[
    {value:"renumber",label:"Reassign all IDs and save",primary:true,default:true},
    {value:"renumberReview",label:"Reassign IDs, save and confirm the drafts",disabled:!drafts.length},
    {value:"save",label:"Save only"},
    {value:"review",label:`Save and confirm ${drafts.length} drafts`,disabled:!drafts.length},
    {value:null,label:"Cancel"},
  ]);
  if(!choice)return;
  if(!$("reviewer").value.trim()){toast("Enter the reviewer name.",true);$("reviewer").focus();return;}
  const renumber=choice==="renumber"||choice==="renumberReview",review=choice==="review"||choice==="renumberReview";
  if(review){
    if(drafts.some(([,f])=>f.boxes.some(b=>b.class_ambiguous))){toast("First choose the class of the ambiguous boxes. You can use Save only.",true);return;}
  }
  const changes=Object.entries(S.state.frames).filter(([,frame])=>
    (review&&frame.status==="draft")||(renumber&&frame.boxes.some((box,index)=>boxNumber(box,index)!==index+1)));
  if(changes.length){
    checkpoint();
    for(const [key,frame] of changes){
      const next=copy(frame);
      if(renumber)assignConsecutiveBoxIds(next.boxes);
      if(review&&next.status==="draft")next.status="reviewed";
      S.state.frames[key]=next;
    }
    markDirty();renderAnnotations();
  }
  // Even without edits, Save refreshes the derived fields of boxes created with older rules.
  if(S.saved===S.dirty&&Object.values(S.state.frames).some(f=>f.boxes.length))markDirty();
  await flush();toast(renumber?(review?"IDs reassigned, annotations saved and drafts confirmed.":"IDs reassigned and annotations saved."):review?"Annotations saved and confirmed.":"Changes saved.");
}
// Applies the current plane boxes to a range of planes, writing through applyScope.
function planeRun(index, step) {
  let i = index;
  while (S.units[i+step]?.known && (step > 0 ? S.units[i].end+1 === S.units[i+1].start : S.units[i-1].end+1 === S.units[i].start)) i += step;
  return i;
}
function describePlanes() {
  const from = Number($("planesFrom").value), to = Number($("planesTo").value);
  const ok = Number.isInteger(from) && Number.isInteger(to) && from >= 1 && to <= S.units.length && from <= to;
  $("planesRange").textContent = ok ? `Planes ${from}–${to} · frames ${S.units[from-1].start}–${S.units[to-1].end} · ${to-from+1} planes` : "Enter a valid range of planes.";
}
function openApplyPlanes() {
  stopPlay();commitEditor();
  const unit = planeScope();
  if (!unit || keyframeSource()) { toast("Available in VTCD planes mode.", true); return; }
  if (!writable() || !scopeReady()) return;
  if (!currentBoxes().length) { toast("The current plane has no boxes to apply.", true); return; }
  const index = S.units.indexOf(unit);
  $("planesSource").textContent = `Source: plane ${index+1} of ${S.units.length} (frames ${unit.start}–${unit.end}), ${currentBoxes().length} boxes${S.selected.size ? `, ${S.selected.size} selected` : ""}.`;
  $("planesBoxes").value = S.selected.size ? "selected" : "all";
  $("planesBoxes").querySelector("[value=selected]").disabled = !S.selected.size;
  $("planesFrom").max = $("planesTo").max = S.units.length;
  $("planesFrom").value = index+1; $("planesTo").value = Math.min(planeRun(index, 1), index+10)+1;
  $("planesError").textContent = ""; describePlanes(); $("planesDialog").showModal();
}
function applyBoxesToPlanes(boxes, from, to, mode, overwrite) {
  const source = planeScope(), result = {applied: 0, protected: 0, incompatible: 0, conflict: 0, full: 0};
  const changed = [];
  for (let i = from; i <= to; i++) {
    const unit = S.units[i];
    if (unit === source || !unit.known) continue;
    const classes = unit.classes || classesAt(unit.start);
    if (boxes.some(b => !classes.includes(b.class_name))) { result.incompatible++; continue; }
    if (unitConflict(unit)) { result.conflict++; continue; }
    const existing = S.state.frames[String(unit.start)];
    if (existing && existing.status !== "draft" && !overwrite) { result.protected++; continue; }
    const base = mode === "add" ? copy(existing ? existing.boxes : S.proposals[String(unit.start)] || []) : [];
    if (base.length + boxes.length > 500) { result.full++; continue; }
    base.forEach((b,k) => b.display_id = boxNumber(b,k));
    const used = new Set(base.map(boxNumber));
    for (const original of boxes) {
      const box = {...copy(original), id: crypto.randomUUID(), origin: "propagated",
        derived_from: {start: source.start, end: source.end, box_id: original.id}};
      delete box.token_space;
      if (used.has(box.display_id)) box.display_id = Math.max(0, ...used) + 1;
      used.add(box.display_id); base.push(box);
    }
    changed.push([unit, {boxes: base, status: "draft", note: existing?.note || ""}]);
  }
  // One checkpoint: Ctrl+Z reverts all the planes together.
  if (changed.length) { checkpoint(); for (const [unit, item] of changed) applyScope(item, unit); markDirty(); renderAnnotations(); }
  result.applied = changed.length;
  return result;
}
function submitApplyPlanes(event) {
  event.preventDefault();
  if (!writable() || !planeScope()) return;
  const from = Number($("planesFrom").value), to = Number($("planesTo").value);
  if (!Number.isInteger(from) || !Number.isInteger(to) || from < 1 || to > S.units.length || from > to) { $("planesError").textContent = "Enter two valid planes in increasing order."; return; }
  const all = currentBoxes().map((b,i) => ({...b, display_id: boxNumber(b,i)}));
  const boxes = $("planesBoxes").value === "selected" ? all.filter(b => S.selected.has(b.id)) : all;
  if (!boxes.length) { $("planesError").textContent = "No boxes to apply."; return; }
  if (boxes.some(b => b.class_ambiguous)) { $("planesError").textContent = "First choose the event of the boxes with an ambiguous class."; return; }
  const r = applyBoxesToPlanes(boxes, from-1, to-1, $("planesMode").value, $("planesOverwrite").checked);
  const skipped = [r.protected && `${r.protected} confirmed/excluded protected`, r.incompatible && `${r.incompatible} with incompatible classes`,
    r.conflict && `${r.conflict} with per-frame corrections`, r.full && `${r.full} over 500 boxes`].filter(Boolean);
  if (!r.applied) { $("planesError").textContent = `No plane changed${skipped.length ? ": " + skipped.join(", ") : ""}.`; return; }
  $("planesDialog").close();
  toast(`Boxes applied to ${r.applied} ${r.applied === 1 ? "plane" : "planes"} as draft${skipped.length ? " · skipped: " + skipped.join(", ") : ""}. Ctrl+Z undoes.`);
}
function initApplyPlanes() {
  $("applyPlanes").onclick = openApplyPlanes; $("planesForm").onsubmit = submitApplyPlanes;
  $("closePlanes").onclick = () => $("planesDialog").close();
  $("planesFrom").oninput = $("planesTo").oninput = describePlanes;
  const index = () => S.units.indexOf(planeScope());
  $("planesRunStart").onclick = () => { $("planesFrom").value = planeRun(index(), -1)+1; describePlanes(); };
  $("planesRunEnd").onclick = () => { $("planesTo").value = planeRun(index(), 1)+1; describePlanes(); };
  $("planesNext10").onclick = () => { $("planesTo").value = Math.min(S.units.length, Number($("planesTo").value || index()+1)+10); describePlanes(); };
}
function initWorkflow() {
  initApplyPlanes();
  $("firstSourceSample").onclick=()=>{if(S.units.length)navigate(S.units[0].start);};
  $("useSourceSample").onclick=async()=>{
    stopPlay();commitEditor();
    const boxes=S.proposals[String(S.frame)];
    if(!keyframeSource()||!boxes?.length||!writable()||!scopeReady())return;
    if(S.state.frames[String(S.frame)]){
      const choice=await askChoice("Use the source sample",`Replace the boxes of frame ${S.frame} only with the Liu & Ma ones? The other annotations stay unchanged. The frame goes back to draft; Ctrl+Z undoes.`,[
        {value:"replace",label:"Replace this frame",primary:true},{value:null,label:"Keep my annotations"}]);
      if(choice!=="replace")return;
    }
    editFrame(f=>{f.boxes=copy(boxes);f.note="Source Liu & Ma 2019: original sample to review. Index conventions not documented by the authors.";});
    S.selected=new Set(currentBoxes().map(b=>b.id));renderAnnotations();
  };
  $("tokenButton").onclick=toggleTokenSpace;$("precisionButton").onclick=togglePrecision;
  $("canvas").addEventListener("dblclick",event=>{
    if(event.button===0&&!event.target.hasAttribute("data-handle")){event.preventDefault();if(!S.lastDouble||event.timeStamp-S.lastDouble>600)baseBoxAt(event);}
  });
  $("viewport").addEventListener("wheel",event=>{
    // Pinch arrives as wheel+ctrl; two-finger scrolling stays native on both axes.
    if(!S.video||!event.ctrlKey)return;
    event.preventDefault();const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?200:1);
    setZoom(S.zoom*Math.exp(-Math.max(-180,Math.min(180,delta))*.004));
  },{passive:false});
}
