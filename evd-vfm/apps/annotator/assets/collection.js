"use strict";
const Batch = {items:[],busy:false,cancel:false};
function batchVisible(item) {
  const filter=$("batchFilter").value.toLowerCase();
  return !filter || item.name.toLowerCase().includes(filter);
}
function renderBatch() {
  $("batchList").replaceChildren();
  for(const item of Batch.items.filter(batchVisible)) {
    const row=el("label",{class:"batch-row"}),check=el("input",{type:"checkbox"});
    check.checked=item.selected;check.disabled=Batch.busy;
    check.onchange=()=>{item.selected=check.checked;item.result=null;renderBatch();};
    row.append(check,el("span",{},item.name),el("small",{class:item.error?"error":"muted"},item.error||item.status||"To check"));
    $("batchList").append(row);
  }
  const selected=Batch.items.filter(i=>i.selected);
  $("batchCount").textContent=`${selected.length} pairs selected · ${Batch.items.length} available`;
  $("batchPreview").disabled=Batch.busy||!selected.length;
  $("batchImport").disabled=Batch.busy||!selected.some(i=>i.result&&!i.error);
  $("batchFiles").disabled=Batch.busy;$("batchLocal").disabled=Batch.busy;
  $("batchDataset").disabled=Batch.busy;$("batchSelectAll").disabled=Batch.busy;
  $("batchDefaults").disabled=Batch.busy;$("batchLabel").disabled=Batch.busy;
  $("batchStop").hidden=!Batch.busy;
}
async function batchBody(item) {
  const label=$("batchLabel").value.trim()||item.label;
  return item.files ? {metadata:await item.files.metadata.text(),csv:await item.files.csv.text(),label,dataset:$("batchDataset").value}
    : {video:item.video,candidate:item.candidate,label};
}
async function loadLocalCollection() {
  const response=await api("/api/import-candidates");
  Batch.items=response.items.map(i=>({...i,selected:false}));renderBatch();
  $("batchProgress").textContent="Filter the collection, select the pairs and press Check.";
}
async function openCollection() {
  stopPlay();closeBoxMenu();commitEditor();await flush();
  $("batchDialog").showModal();
  try{await loadLocalCollection();}catch(e){$("batchProgress").textContent=e.message;}
}
async function runBatch(commit) {
  if(Batch.busy)return;
  const items=Batch.items.filter(i=>i.selected&&(!commit||i.result&&!i.error));
  if(!items.length)return;
  const selectedSources=new Map();
  if(commit)for(const {result} of items){
    if(!selectedSources.has(result.video))selectedSources.set(result.video,new Set());
    selectedSources.get(result.video).add(result.source_id);
  }
  Batch.busy=true;Batch.cancel=false;renderBatch();
  const completed=[];
  try{
    for(let i=0;i<items.length;i++){
      if(Batch.cancel)break;
      const item=items[i];item.error="";
      $("batchProgress").textContent=`${commit?"Importing":"Checking"} ${i+1}/${items.length} · ${item.name}`;
      try{
        const result=await api(commit?"/api/import-pair":"/api/import-preview",await batchBody(item));
        item.result=result;item.status={ready:"Ready",existing:"Already present",imported:"Imported"}[result.status];
        if(commit)completed.push(result);
      }catch(e){item.error=e.message;item.result=null;}
      renderBatch();
    }
    let defaults="";
    if(commit&&$("batchDefaults").checked&&completed.length){
      const perVideo=new Map();
      for(const result of completed){if(!perVideo.has(result.video))perVideo.set(result.video,new Map());perVideo.get(result.video).set(result.source_id,result);}
      const unambiguous=[...perVideo].filter(([video,m])=>m.size===1&&selectedSources.get(video).size===1).map(([,m])=>[...m.values()][0]);
      const chosen=await api("/api/import-defaults",{items:unambiguous.map(({video,source_id})=>({video,source_id}))});
      defaults=` · ${chosen.applied.length} starting points set · ${chosen.skipped.length} videos with corrections preserved`;
      const ambiguous=[...selectedSources.values()].filter(s=>s.size>1).length;
      if(ambiguous)defaults+=` · ${ambiguous} videos with several sources: choose the one to use in the video`;
    }
    const errors=items.filter(i=>i.error).length;
    $("batchProgress").textContent=`${Batch.cancel?"Stopped":"Completed"}: ${commit?completed.length:items.filter(i=>i.result).length} pairs ${commit?"imported or already present":"checked"}, ${errors} errors${defaults}.`;
    if(commit){
      const catalog=await api("/api/catalog");S.catalog=catalog.videos;renderLibrary();
      if(S.video){
        const data=await api(endpoint());S.sources=data.sources;S.candidates=data.candidates;
        if(!Object.keys(S.state.frames).length&&data.preferred_source)await loadSource(data.preferred_source,false);
        else renderSources();
      }
    }
  }catch(e){$("batchProgress").textContent=e.message;}
  finally{Batch.busy=false;renderBatch();}
}
function initCollection(){
  $("collectionImport").onclick=()=>openCollection().catch(e=>toast(e.message,true));
  $("batchLocal").onclick=()=>loadLocalCollection().catch(e=>$("batchProgress").textContent=e.message);
  $("batchFilter").oninput=renderBatch;
  $("batchSelectAll").onclick=()=>{
    const visible=Batch.items.filter(batchVisible),all=visible.every(i=>i.selected);
    for(const item of visible)item.selected=!all;renderBatch();
  };
  for(const id of ["batchDataset","batchLabel"])$(id).onchange=()=>{for(const item of Batch.items){item.result=null;item.status="";item.error="";}renderBatch();};
  $("batchPreview").onclick=()=>runBatch(false);$("batchImport").onclick=()=>runBatch(true);
  $("batchStop").onclick=()=>{Batch.cancel=true;$("batchProgress").textContent="Stopping after the current pair…";};
  $("batchDialog").addEventListener("cancel",e=>{if(Batch.busy){e.preventDefault();Batch.cancel=true;}});
  $("closeBatch").onclick=()=>{if(Batch.busy)Batch.cancel=true;else $("batchDialog").close();};
  $("batchFiles").onchange=()=>{
    const groups=new Map();
    for(const file of $("batchFiles").files){
      if(!["boxes.json","boxes_per_frame.csv"].includes(file.name))continue;
      const relative=file.webkitRelativePath||file.name,folder=relative.slice(0,relative.lastIndexOf("/"));
      if(!groups.has(folder))groups.set(folder,{});
      groups.get(folder)[file.name==="boxes.json"?"metadata":"csv"]=file;
    }
    Batch.items=[...groups].map(([name,files])=>({name,label:name.split("/").at(-1)||"Imported collection",files,
      selected:!!(files.metadata&&files.csv),error:files.metadata&&files.csv?"":"Incomplete pair: JSON and CSV are needed in the same folder"}));
    renderBatch();$("batchProgress").textContent="Automatic matching by video ID; choose the dataset if names are ambiguous. Videos are not uploaded.";
  };
}
