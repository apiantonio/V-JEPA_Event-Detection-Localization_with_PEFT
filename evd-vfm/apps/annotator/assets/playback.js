"use strict";
// The pace follows the native frames, not a fixed delay per plane.
const FrameBuffer={video:"",generation:0,cache:new Map(),pending:new Map(),bytes:0,poll:null,warm:0};
const FRAME_BUFFER_LIMIT=48*1024*1024;
function resetFrameBuffer() {
  FrameBuffer.video=S.video.id;FrameBuffer.generation++;FrameBuffer.cache.clear();
  FrameBuffer.pending.clear();FrameBuffer.bytes=0;clearTimeout(FrameBuffer.poll);clearTimeout(FrameBuffer.warm);
}
async function frameBlob(index) {
  if(FrameBuffer.cache.has(index)) {
    const blob=FrameBuffer.cache.get(index);FrameBuffer.cache.delete(index);FrameBuffer.cache.set(index,blob);return blob;
  }
  if(FrameBuffer.pending.has(index))return FrameBuffer.pending.get(index);
  const generation=FrameBuffer.generation,url=endpoint(`/frame?n=${index}`);
  const task=(async()=>{
    const response=await fetch(url);
    if(!response.ok)throw new Error((await response.json()).error);
    const blob=await response.blob();
    if(generation===FrameBuffer.generation){
      while(FrameBuffer.bytes+blob.size>FRAME_BUFFER_LIMIT&&FrameBuffer.cache.size){
        const [key,old]=FrameBuffer.cache.entries().next().value;FrameBuffer.cache.delete(key);FrameBuffer.bytes-=old.size;
      }
      if(blob.size<=FRAME_BUFFER_LIMIT){FrameBuffer.cache.set(index,blob);FrameBuffer.bytes+=blob.size;}
    }
    return blob;
  })();
  FrameBuffer.pending.set(index,task);
  try{return await task;}finally{if(generation===FrameBuffer.generation)FrameBuffer.pending.delete(index);}
}
function warmFrames() {
  clearTimeout(FrameBuffer.warm);
  const generation=FrameBuffer.generation,frame=S.frame;
  FrameBuffer.warm=setTimeout(async()=>{
    for(let i=1;i<=4;i++){
      if(generation!==FrameBuffer.generation||frame!==S.frame)return;
      try{await frameBlob(stepTarget(i));}catch{return;}
    }
  },20);
}
function startPreload() {
  const generation=FrameBuffer.generation,url=endpoint("/preload");
  $("cacheStatus").textContent="Preloading events…";
  const report=status=>{
    if(generation!==FrameBuffer.generation)return;
    const tail={complete:"ready",loading:"preloaded",limited:"partial cache · 1 GiB limit",inactive:"cache in use by another video",error:"preload interrupted"};
    $("cacheStatus").textContent=status.total ? `${status.ready.toLocaleString("en")}/${status.total.toLocaleString("en")} frames · ${tail[status.state]}` : tail[status.state]||"";
    $("cacheStatus").title=status.error||"Event frames loaded in the background. Disk cache limited to 1 GiB; browser RAM to 48 MiB.";
    if(status.state==="loading")FrameBuffer.poll=setTimeout(()=>api(url).then(report).catch(fail),1000);
  };
  const fail=e=>{if(generation===FrameBuffer.generation){$("cacheStatus").textContent="Preload not available";$("cacheStatus").title=e.message;}};
  api(url,{frame:S.frame}).then(report).catch(fail);
}
function stopPlay() {
  const wasPlaying=S.playing;S.playing=false;S.playEpoch=(S.playEpoch||0)+1;
  clearTimeout(S.playTimer);$("play").textContent="▶";
  if(wasPlaying&&S.loading){S.loadId++;S.loading=false;$("canvasMessage").hidden=true;}
}
function playbackDelay() {
  const unit=S.unitMode==="planes"?activeUnit():null;
  const count=unit?unit.end-S.frame+1:1;
  return 1000*count/S.video.fps/Number($("playRate").value);
}
function togglePlay() {
  if(!S.video)return;
  if(S.playing){stopPlay();return;}
  if(S.loading)return;
  closeBoxMenu();commitEditor();S.playing=true;$("play").textContent="Ⅱ";
  const epoch=S.playEpoch=(S.playEpoch||0)+1;
  const anchorFrame=S.frame,anchorTime=performance.now(),rate=Number($("playRate").value);
  const frameDuration=1000/S.video.fps/rate;
  const tick=async()=>{
    if(!S.playing||epoch!==S.playEpoch)return;
    const next=S.unitMode==="frames"?Math.min(S.video.num_frames-1,
      Math.max(S.frame+1,anchorFrame+Math.floor((performance.now()-anchorTime)/frameDuration))):stepTarget(1);
    if(next===S.frame){stopPlay();return;}
    const started=performance.now();await goFrame(next);
    if(!S.playing||epoch!==S.playEpoch)return;
    const delay=playbackDelay(),spent=performance.now()-started;
    $("playRate").title=spent>delay ? "Speed limited by loading: wait for the events to preload." : "Speed relative to the native video time";
    const wait=S.unitMode==="frames"?anchorTime+(S.frame-anchorFrame+1)*frameDuration-performance.now():delay-spent;
    S.playTimer=setTimeout(tick,Math.max(0,wait));
  };
  warmFrames();S.playTimer=setTimeout(tick,playbackDelay());
}
function initPlayback() {
  $("playRate").onchange=()=>{if(S.playing){stopPlay();togglePlay();}};
}
