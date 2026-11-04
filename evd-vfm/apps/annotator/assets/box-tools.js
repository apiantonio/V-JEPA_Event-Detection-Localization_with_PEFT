"use strict";
// Grid and menu use native coordinates; zoom is only a view.
function renderGrid() {
  const layer = $("tokenGrid"); layer.replaceChildren();
  $("gridButton").setAttribute("aria-pressed",String(S.gridVisible));
  $("gridButton").classList.toggle("active",S.gridVisible);
  if (!S.grid) return;
  $("gridButton").title = `${S.grid.columns} × ${S.grid.rows} cells · token ${S.grid.patch_px} × ${S.grid.patch_px} on the model (${S.grid.input_size} px). ${shortcutLabel("grid")}`;
  if (!S.gridVisible) return;
  const lines = [...S.grid.x_edges.map(x=>`M${x},0V${S.video.height}`),
    ...S.grid.y_edges.map(y=>`M0,${y}H${S.video.width}`)].join(" ");
  layer.append(svg("path",{d:lines,class:"token-grid-shadow"}),svg("path",{d:lines,class:"token-grid-lines"}));
}
function toggleGrid() { S.gridVisible=!S.gridVisible; renderBoxes(); }
function assignConsecutiveBoxIds(boxes) {
  boxes.forEach((box,index)=>box.display_id=index+1);
}
function reassignBoxIds() {
  stopPlay();if(!writable()||S.gesture)return;
  commitEditor();if(!scopeReady()||!currentBoxes().length)return;
  closeBoxMenu();
  if(currentBoxes().every((box,index)=>boxNumber(box,index)===index+1)){
    toast("The IDs are already consecutive from 1.");return;
  }
  editFrame(frame=>assignConsecutiveBoxIds(frame.boxes));
  toast(`IDs reassigned from 1 to ${currentBoxes().length}. Ctrl+Z undoes; Ctrl+S saves.`);
}
// Tab clipboard: a copy stays available while navigating the planes of the video.
let boxClipboard=null;
function copySelectedBoxes() {
  stopPlay();if(!writable()||S.gesture)return;
  commitEditor();
  const boxes=currentBoxes().filter(box=>S.selected.has(box.id));
  if(!boxes.length){toast("Select at least one box to copy.");return;}
  boxClipboard={video:S.video.id,frame:S.frame,boxes:copy(boxes)};
  toast(`${boxes.length===1?"Box copied":"Boxes copied: "+boxes.length}. Ctrl+V pastes into the video.`);
}
function pasteCopiedBoxes() {
  stopPlay();if(!writable()||S.gesture)return;
  commitEditor();if(!scopeReady())return;
  if(!boxClipboard){toast("Select a box and copy it first with Ctrl+C.");return;}
  if(boxClipboard.video!==S.video.id){toast("The boxes in the clipboard belong to another video. Go back to that video to paste them.",true);return;}
  const classes=classesAt(S.frame);
  if(!classes.length){toast("Paste into the planes or frames of the annotated events.",true);return;}
  if(boxClipboard.boxes.some(box=>!classes.includes(box.class_name))){toast("The classes of the copied boxes do not belong to the current event.",true);return;}
  if(currentBoxes().length+boxClipboard.boxes.length>500){toast("A plane or frame can contain at most 500 boxes.",true);return;}
  const used=new Set(currentBoxes().map(boxNumber));let number=nextBoxNumber();
  const duplicates=boxClipboard.boxes.map(original=>{
    if(number>999999)number=1;
    while(used.has(number))number++;
    used.add(number);
    const box={...copy(original),id:crypto.randomUUID(),display_id:number++,origin:"propagated",
      derived_from:{frame:boxClipboard.frame,box_id:original.id}};
    delete box.token_space;return box;
  });
  editFrame(frame=>frame.boxes.push(...duplicates));
  S.selected=new Set(duplicates.map(box=>box.id));setTool("select");closeBoxMenu();renderAnnotations();
  toast(`${duplicates.length===1?"Box pasted":"Boxes pasted: "+duplicates.length}. You can move the selection; Ctrl+Z undoes.`);
}
function snapAxis(start, end, edges) {
  // Nearest edges over all non-empty pairs: avoids collapsed boxes when both
  // edges fall near the same grid line.
  let best=null, distance=Infinity;
  for(let a=0;a<edges.length-1;a++) for(let b=a+1;b<edges.length;b++) {
    if(edges[b]<=edges[a]) continue;
    const d=Math.abs(edges[a]-start)+Math.abs(edges[b]-(end+1));
    if(d<distance) {distance=d;best=[edges[a],edges[b]-1];}
  }
  return best;
}
function closeBoxMenu() { if($("boxMenu")) $("boxMenu").hidden=true; S.context=null; }
function contextBox() {
  if(!S.context || S.context.video!==S.video?.id || S.context.frame!==S.frame) return null;
  return currentBoxes().find(b=>b.id===S.context.id);
}
function applyBoxMenu() {
  if($("boxMenu").hidden) return true;
  const box=contextBox(); if(!box) return false;
  const number=Number($("menuBoxId").value),color=$("menuBoxColor").value;
  if(!Number.isInteger(number)||number<1||number>999999 || currentBoxes().some((b,i)=>b.id!==box.id&&boxNumber(b,i)===number)) {
    $("boxMenuError").textContent="Choose a free numeric ID between 1 and 999999.";return false;
  }
  editFrame(f=>{const b=f.boxes.find(b=>b.id===box.id);b.display_id=number;b.color=color;});closeBoxMenu();
  return true;
}
function showBoxMenu(event,id) {
  event.preventDefault(); if(!writable()) return;
  commitEditor(); stopPlay(); S.selected=new Set([id]); renderAnnotations();
  const box=currentBoxes().find(b=>b.id===id); if(!box) return;
  S.context={video:S.video.id,frame:S.frame,id};
  $("boxMenuTitle").textContent=`Box ${boxNumber(box,currentBoxes().indexOf(box))}`;
  $("menuBoxId").value=boxNumber(box,currentBoxes().indexOf(box));
  $("menuBoxColor").value=boxColor(box); $("boxMenuError").textContent="";
  $("snapBox").disabled=!S.grid; $("boxMenu").hidden=false;
  const rect=$("boxMenu").getBoundingClientRect();
  $("boxMenu").style.left=`${Math.max(8,Math.min(event.clientX,innerWidth-rect.width-8))}px`;
  $("boxMenu").style.top=`${Math.max(8,Math.min(event.clientY,innerHeight-rect.height-8))}px`;
  $("snapBox").focus({preventScroll:true});
}
function initBoxTools() {
$("canvas").addEventListener("contextmenu",event=>{
  const id=event.target.getAttribute("data-box");
  if(id) showBoxMenu(event,id); else {event.preventDefault();closeBoxMenu();}
});
$("gridButton").onclick=toggleGrid;
$("reassignBoxIds").onclick=reassignBoxIds;
$("closeBoxMenu").onclick=closeBoxMenu;
$("snapBox").onclick=()=>{
  const box=contextBox(); if(!box || !S.grid) return;
  const x=snapAxis(box.xyxy[0],box.xyxy[2],S.grid.x_edges);
  const y=snapAxis(box.xyxy[1],box.xyxy[3],S.grid.y_edges);
  if(x&&y) editFrame(f=>f.boxes.find(b=>b.id===box.id).xyxy=[x[0],y[0],x[1],y[1]]);
  closeBoxMenu();
};
$("boxMenuForm").onsubmit=event=>{
  event.preventDefault();applyBoxMenu();
};
$("menuDeleteBox").onclick=()=>{if(contextBox()) deleteSelected();closeBoxMenu();};
$("menuDuplicateBox").onclick=()=>{
  const original=contextBox(); if(!original) return;
  const duplicate={...copy(original),id:crypto.randomUUID(),display_id:nextBoxNumber(),origin:"manual",
    derived_from:{frame:S.frame,box_id:original.id}};
  editFrame(f=>f.boxes.push(duplicate));closeBoxMenu();
};
document.addEventListener("pointerdown",e=>{if(!$("boxMenu").hidden&&!$("boxMenu").contains(e.target)&&!e.target.closest("#saveButton,#confirmFrame"))closeBoxMenu();});
window.addEventListener("resize",()=>{if(typeof S!=="undefined")closeBoxMenu();});
$("viewport").addEventListener("scroll",()=>{if(typeof S!=="undefined")closeBoxMenu();});
}
