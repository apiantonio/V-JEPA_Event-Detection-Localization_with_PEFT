/* Shared presentation of the metrics, used by replay, analysis and live mode. */
window.EVDMetrics = {
  bindPopup(dialog, trigger, close) {
    trigger.addEventListener('click', () => { if (!dialog.open) dialog.showModal(); });
    close.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      const box = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
    });
  },
  render(root, video, global = {}) {
    root.replaceChildren();
    const number = (v, digits=3, unit='') => typeof v==='number' && Number.isFinite(v) ? v.toFixed(digits)+unit : 'n/a';
    function section(title, rows) {
      const heading=document.createElement('h3');heading.textContent=title;
      const table=document.createElement('dl');table.className='metric-values';
      for(const [label,value] of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=String(value);table.append(dt,dd);}
      root.append(heading,table);
    }
    const globalRows=[];
    for(const [key,label] of [['chunk_auc','Chunk AUC'],['chunk_ap','Chunk AP'],['video_f1','Video F1']])
      if(typeof global[key]==='number')globalRows.push([label,number(global[key],4)]);
    if(globalRows.length)section('Model · '+(global.eval_split||'test'),globalRows);
    if(!video || !Object.keys(video).length){section('This video',[['Metrics','Reopen the video to compute them']]);return;}
    const rows=[['Analysed windows',video.n_windows],['Windows above threshold',video.above_threshold]];
    const chunk=video.chunk;
    if(chunk){
      for(const [key,label] of [['precision','Precision (windows)'],['recall','Recall (windows)'],['f1','F1 (windows)'],['auc','AUC (windows)'],['ap','AP (windows)'],['accuracy','Accuracy (windows)']])rows.push([label,number(chunk[key])]);
      rows.push(['TP / FP / FN / TN',[chunk.tp,chunk.fp,chunk.fn,chunk.tn].join(' / ')]);
      rows.push(['Normal windows with a false alarm',number(chunk.false_positive_rate==null?null:chunk.false_positive_rate*100,2,'%')]);
    } else rows.push(['Metrics against the labels',video.ground_truth?'Waiting for scorable windows':'n/a · annotations not available']);
    const events=video.events;
    if(events){
      rows.push(['Detected / annotated events',events.n_eventi_rilevati+' / '+events.n_eventi]);
      rows.push([video.complete?'Missed events':'Events not yet detected',events.n_eventi-events.n_eventi_rilevati]);
      rows.push(['Event precision / recall',number(events.precision)+' / '+number(events.recall)]);
      rows.push(['Event F1',number(events.f1)]);
    }
    section(video.complete?'This video':'This video · provisional',rows);
    const latency=video.latency_grid, timing=video.timing||{};
    section('Video latencies',[
      ['Detection, median (grid)',number(latency?.mediana_s,3,' s')],
      ['Detection, p90 (grid)',number(latency?.p90_s,3,' s')],
      ['Decoding + inference, mean',number(timing.compute_mean_ms,1,' ms')],
      ['Decoding + inference, p95',number(timing.compute_p95_ms,1,' ms')],
      [timing.analysis_s!=null?'Video processing, loading excluded':'Decoding + inference, total',number(timing.analysis_s??timing.compute_total_s,3,' s')],
      ...(timing.mean_window_ms!=null?[['Mean cost per window (amortized)',number(timing.mean_window_ms,1,' ms')]]:[]),
    ]);
    const note=document.createElement('p');note.className='hint';
    note.textContent='The grid latency runs from the annotated event onset to the first endpoint above threshold intersecting it: it excludes compute and UI updates, and refers to detected events only. n/a marks a measure that is unavailable or undefined; AUC/AP need both classes. Per-video metrics are diagnostic; the global ones refer to the original geometry.';
    root.append(note);
  }
};
