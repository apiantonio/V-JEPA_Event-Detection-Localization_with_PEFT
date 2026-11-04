/* 1x player and incremental results. The server only receives the time already played. */
window.EVDLive = (() => {
  const el = id => document.getElementById(id);
  let active = false, send = null, id = null, timer = null, sending = false;
  let info = null, points = [], finalUrl = null, position = 0;
  const video = () => el('rt-video');
  const seconds = value => Number(value || 0).toFixed(1) + ' s';

  function draw() {
    if (!info) return;
    const canvas = el('rt-timeline'), ctx = canvas.getContext('2d');
    const width = canvas.clientWidth || 800, height = 190, ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio; canvas.height = height * ratio;
    ctx.scale(ratio, ratio);
    const left = 42, right = width - 16, top = 18, bottom = height - 28;
    const x = t => left + Math.min(t / info.duration_s, 1) * (right - left);
    const y = s => bottom - s * (bottom - top);
    ctx.fillStyle = '#101419'; ctx.fillRect(0, 0, width, height);
    ctx.font = '12px Segoe UI'; ctx.fillStyle = '#a2afbe';
    for (const score of [0, 0.5, 1]) {
      ctx.fillText(String(score), 8, y(score) + 4);
      ctx.strokeStyle = '#2c3541'; ctx.beginPath();ctx.moveTo(left,y(score));ctx.lineTo(right,y(score));ctx.stroke();
    }
    ctx.strokeStyle = '#e5b56d';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(left,y(info.threshold));ctx.lineTo(right,y(info.threshold));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle = '#e5b56d';ctx.fillText('Threshold '+info.threshold.toFixed(3),left+8,top+12);
    ctx.strokeStyle = '#72dfbc';ctx.lineWidth = 2;ctx.beginPath();
    points.forEach((p,i) => {if(i)ctx.lineTo(x(p.end_s),y(p.score));else ctx.moveTo(x(p.end_s),y(p.score));});ctx.stroke();
    for (const p of points) {ctx.fillStyle=p.alarm?'#ff9c8d':'#72dfbc';ctx.beginPath();ctx.arc(x(p.end_s),y(p.score),3,0,Math.PI*2);ctx.fill();}
    ctx.strokeStyle='#edf2f7';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(position),top);ctx.lineTo(x(position),bottom);ctx.stroke();
    ctx.fillStyle='#a2afbe';ctx.fillText('0 s',left,height-7);ctx.fillText(seconds(info.duration_s),Math.max(left,right-65),height-7);
    const latest = points.at(-1);
    el('rt-clock').textContent=seconds(position)+' / '+seconds(info.duration_s);
    el('rt-coverage').textContent=latest ? seconds(latest.end_s) : 'Waiting';
    el('rt-lag').textContent=seconds(Math.max(0,position-(latest?.end_s||0)));
  }

  async function heartbeat() {
    if (!active || sending) return;
    position = Math.max(position, video().ended && info ? info.duration_s : video().currentTime || 0);
    draw(); sending=true;
    try {await send('/api/playback',{id,position_s:position});}
    catch(e) {video().pause();el('rt-state').textContent='Connection lost: '+e.message;}
    finally {sending=false;}
  }

  function stop() {active=false;clearInterval(timer);timer=null;video().pause();}

  function start(result, api) {
    stop();el('rt-metrics-dialog').close();id=result.id;send=api;info=null;points=[];position=0;el('rt-metrics').replaceChildren();finalUrl=null;active=true;
    el('realtime-panel').hidden=false;el('rt-list').replaceChildren();
    el('rt-name').textContent='Local live';el('rt-state').textContent='Loading the model…';
    el('rt-alert').textContent='Waiting for the first window';el('rt-alert').classList.remove('anomaly');
    el('rt-score').textContent='—';el('rt-count').textContent='0';el('rt-compute').textContent='—';
    el('rt-clock').textContent='0 s';el('rt-coverage').textContent='Waiting';el('rt-lag').textContent='0 s';
    el('rt-back').disabled=true;el('rt-stop').hidden=false;el('rt-play').disabled=true;el('rt-play').textContent='Start video';el('rt-final').hidden=true;
    const ctx=el('rt-timeline').getContext('2d');ctx.clearRect(0,0,el('rt-timeline').width,el('rt-timeline').height);
    video().src=result.video_url;video().muted=true;video().playbackRate=1;
    timer=setInterval(heartbeat,250);heartbeat();
  }

  function update(events) {
    for (const event of events || []) {
      if(event.type==='loading')el('rt-state').textContent=event.message;
      if(event.type==='ready') {
        info=event;EVDMetrics.render(el('rt-metrics'),event.video_metrics,event.metrics);el('rt-name').textContent=event.video_id;
        el('rt-state').textContent='Model ready · hop '+seconds(event.hop_s)+' / window '+seconds(event.window_s)+' · press Start video';
        el('rt-play').disabled=false;el('rt-count').textContent='0 / '+event.total;
      }
      if(event.type==='score' && !points.some(p=>p.index===event.index)) {
        points.push(event);EVDMetrics.render(el('rt-metrics'),event.video_metrics,info.metrics);el('rt-score').textContent=event.score.toFixed(3);
        el('rt-count').textContent=event.index+' / '+info.total;
        el('rt-compute').textContent=event.compute_ms.toFixed(0)+' ms';
        el('rt-alert').classList.toggle('anomaly',event.alarm);
        el('rt-alert').textContent=event.alarm?'ANOMALY DETECTED':'Last window below threshold';
        const row=document.createElement('tr');
        for(const value of [event.index,seconds(event.start_s)+' – '+seconds(event.end_s),event.score.toFixed(3),event.alarm?'Anomaly':'Below threshold']) {
          const cell=document.createElement('td');cell.textContent=value;row.append(cell);
        }
        row.classList.toggle('anomaly',event.alarm);el('rt-list').prepend(row);
        if(el('rt-list').children.length>30)el('rt-list').lastElementChild.remove();
      }
    }
    draw();
  }

  function finish(data) {
    stop();el('rt-back').disabled=false;el('rt-stop').hidden=true;el('rt-play').disabled=true;
    el('rt-state').textContent=data.state==='done'?'Analysis completed':data.state==='cancelled'?'Analysis cancelled · partial results':'Analysis interrupted · partial results';
    if(data.video_metrics)EVDMetrics.render(el('rt-metrics'),data.video_metrics,data.metrics||info?.metrics);
    if(data.url){finalUrl=data.url;el('rt-final').hidden=false;}
    draw();
  }
  el('rt-play').onclick=async()=>{
    if(video().paused){try{await video().play();el('rt-play').textContent='Pause';el('rt-state').textContent='Live · 1× playback';}catch(e){el('rt-state').textContent='Video cannot be played: '+e.message;}}
    else{video().pause();el('rt-play').textContent='Resume';el('rt-state').textContent='Video paused · completing the windows already available';}
  };
  video().addEventListener('ended',()=>{heartbeat();el('rt-play').disabled=true;el('rt-state').textContent='Video ended · completing the queued windows';});
  video().addEventListener('ratechange',()=>{if(active&&video().playbackRate!==1)video().playbackRate=1;});
  video().addEventListener('error',()=>{if(active)el('rt-state').textContent='The browser cannot play this format. Cancel and choose an H.264 MP4.';});
  el('rt-final').onclick=()=>{el('realtime-panel').hidden=true;el('result').hidden=false;el('result-title').textContent=info.video_id;el('result-mode').textContent='Full result of the local live analysis';el('viewer').src=finalUrl;el('rt-back').hidden=true;el('back').hidden=false;};
  for (const [button, cls] of [['rt-side-toggle','no-side'],['rt-deck-toggle','no-deck']]) {
    el(button).onclick=()=>{const hidden=el('rt-workspace').classList.toggle(cls);el(button).setAttribute('aria-pressed',String(!hidden));draw();};
  }
  el('rt-fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await el('realtime-panel').requestFullscreen();}catch(e){el('rt-state').textContent='Full screen not available: '+e.message;}};
  if(typeof ResizeObserver !== 'undefined')new ResizeObserver(draw).observe(el('rt-timeline'));
  window.addEventListener('resize',draw);
  EVDMetrics.bindPopup(el('rt-metrics-dialog'),el('rt-metrics-open'),el('rt-metrics-close'));
  return {start,update,finish,stop};
})();
