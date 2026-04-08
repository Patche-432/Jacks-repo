
let since=null, pollTimer=null, activeTab='signals', configSynced=false, thoughtCount=0;
let mt5Connected=false;

/* ── DEBUG UTILITIES ── */
const DEBUG = {
  enabled: true,
  log: function(msg, data) {
    if (this.enabled) {
      const timestamp = new Date().toLocaleTimeString(undefined, {hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit',fractionalSecondDigits:3});
      console.log(`[${timestamp}] ${msg}`, data || '');
    }
  },
  error: function(msg, err) {
    console.error(`[ERROR] ${msg}`, err || '');
  },
  warn: function(msg, data) {
    console.warn(`[WARN] ${msg}`, data || '');
  }
};

/* ── MT5 CONNECT MODAL ── */
function openMt5Modal(){
  const overlay=document.getElementById('mt5-modal-overlay');
  overlay.style.display='flex';
  setMt5ModalStatus('','');
}
function closeMt5Modal(){
  document.getElementById('mt5-modal-overlay').style.display='none';
}
function setMt5ModalStatus(msg,type){
  const el=document.getElementById('mt5-modal-status');
  if(!msg){el.style.display='none';return;}
  el.style.display='block';
  el.style.background=type==='ok'?'rgba(0,200,100,0.12)':type==='err'?'rgba(255,80,80,0.12)':'rgba(255,200,0,0.1)';
  el.style.color=type==='ok'?'var(--green)':type==='err'?'var(--red)':'var(--amber)';
  el.style.border=`1px solid ${type==='ok'?'var(--green)':type==='err'?'var(--red)':'var(--amber)'}`;
  el.textContent=msg;
}
async function connectMt5(){
  const btn=document.getElementById('mt5-modal-btn');
  btn.disabled=true; btn.textContent='Connecting…';
  setMt5ModalStatus('Testing connection…','info');
  const payload={
    login: document.getElementById('mt5-login').value,
    password: document.getElementById('mt5-password').value,
    server: document.getElementById('mt5-server').value,
    path: document.getElementById('mt5-path').value||null,
  };
  try{
    const r=await fetch('/api/mt5/connect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await r.json();
    if(data.ok){
      mt5Connected=true;
      setMt5ModalStatus('✓ Connected successfully!','ok');
      // Update the sidebar connection display straight away
      if(data.mt5) updateMt5({mt5:data.mt5});
      const cb=document.getElementById('connect-btn');
      if(cb){cb.style.background='var(--green)';cb.textContent='MT5 ✓';}
      setTimeout(closeMt5Modal,1400);
    } else {
      setMt5ModalStatus('✗ '+( data.error||'Connection failed'),'err');
    }
  } catch(e){
    setMt5ModalStatus('✗ Could not reach server — is server.py running?','err');
  } finally{
    btn.disabled=false; btn.textContent='Connect';
  }
}

/* ── MT5 status polling ── */
async function checkMt5Status(){
  try{
    const r=await fetch('/api/mt5/status');
    const data=await r.json();
    mt5Connected=!!data.connected;
    updateMt5({mt5:data});
    const cb=document.getElementById('connect-btn');
    if(cb){
      cb.style.background=mt5Connected?'var(--green)':'';
      cb.textContent=mt5Connected?'MT5 ✓':'Connect MT5';
    }
  }catch(e){DEBUG.warn('MT5 status check failed',e);}
}
// Poll MT5 connection status every 2 seconds
setInterval(checkMt5Status,2000);
// ── STARTUP SEQUENCE (runs once DOM is ready) ──
document.addEventListener('DOMContentLoaded', async () => {

  // 1. Close modal on overlay-background click
  document.getElementById('mt5-modal-overlay').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeMt5Modal();
  });

  // 2. Fire-and-forget: warm up the AI in the background
  fetch('/ai/init', { method: 'POST' }).catch(() => {});

  // 3. Check current MT5 connection status
  try {
    const statusRes = await fetch('/api/mt5/status');
    const status = await statusRes.json();
    if (status.connected) {
      mt5Connected = true;
      DEBUG.log('✅ MT5 already connected');
      updateMt5({ mt5: status });
      const cb = document.getElementById('connect-btn');
      if (cb) { cb.style.background = 'var(--green)'; cb.textContent = 'MT5 ✓'; }
    }
  } catch (e) {
    DEBUG.warn('Could not reach /api/mt5/status — server may not be running', e);
  }

  // 4. Initial status poll + start live polling if bot is already running
  try {
    const r = await fetch('/bot/status');
    const s = await r.json();
    updateStatus(s);
    updateSignals(s);
    if (s.running) startPolling();
  } catch (e) {
    DEBUG.error('Initial /bot/status fetch failed', e);
  }
});

/* ── Config helpers ── */
function toggleSym(el){
  el.classList.toggle('active');
  syncSymInput();
}
function syncSymInput(){
  const active=[...document.querySelectorAll('.sym-chip.active')].map(c=>c.textContent);
  document.getElementById('cfg-symbols').value=active.join(',');
}

function stepInput(id,delta,decimals){
  const el=document.getElementById(id);
  const v=Math.round((parseFloat(el.value||0)+delta)*1000)/1000;
  const min=parseFloat(el.min)||0;
  el.value=Math.max(min,v).toFixed(String(decimals).split('.')[1]?.length||0);
}

function setPoll(secs,btn){
  document.getElementById('cfg-poll').value=secs;
  document.querySelectorAll('.poll-preset').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  const labels={30:'30s',60:'1m',300:'5m',900:'15m'};
  document.getElementById('poll-display').textContent=labels[secs]||(secs+'s');
}

function updateRR(){
  const sl=parseFloat(document.getElementById('cfg-sl-mult').value)||1.5;
  const tp=parseFloat(document.getElementById('cfg-tp-mult').value)||3.0;
  const ratio=(tp/sl).toFixed(1);
  document.getElementById('rr-label').textContent='1 : '+ratio;
  const total=sl+tp;
  document.getElementById('rr-sl-bar').style.width=Math.round(sl/total*100)+'%';
}
updateRR();

/* ── MT5 display ── */
function updateMt5(s){
  const m=s.mt5||{};
  const dot=document.getElementById('mt5-dot');
  const stxt=document.getElementById('mt5-status-txt');
  const detail=document.getElementById('mt5-detail');
  if(!m.connected){
    dot.className='mt5-dot err';
    stxt.textContent=m.error||s.bot_error||'Disconnected';
    stxt.style.color='var(--red)';
    detail.style.display='none';
    return;
  }
  dot.className='mt5-dot ok';
  stxt.textContent='Connected';
  stxt.style.color='var(--green)';
  detail.style.display='flex';
  document.getElementById('m-server').textContent=m.server||'—';
  document.getElementById('m-login').textContent=(m.login||'—')+(m.account_name?' · '+m.account_name:'');
  const trEl=document.getElementById('m-trade');
  trEl.textContent=m.trade_allowed?'Allowed':'Disabled';
  trEl.style.color=m.trade_allowed?'var(--green)':'var(--red)';
}

/* ── syncConfig ── */
function syncConfig(s){
  const c=s.config||{}, st=c.strategy||{};
  if(!configSynced){
    /* symbols chips */
    const activeSym=new Set((c.symbols||[]).map(x=>x.toUpperCase()));
    document.querySelectorAll('.sym-chip').forEach(ch=>{
      ch.classList.toggle('active',activeSym.has(ch.textContent.trim()));
    });
    syncSymInput();
    /* volume stepper */
    document.getElementById('cfg-volume').value=Number(c.volume??0.01).toFixed(2);
    /* poll presets */
    const ps=c.poll_secs??300;
    document.getElementById('cfg-poll').value=ps;
    const labels={30:'30s',60:'1m',300:'5m',900:'15m'};
    document.getElementById('poll-display').textContent=labels[ps]||(ps+'s');
    document.querySelectorAll('.poll-preset').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.poll-preset').forEach(p=>{
      if(p.getAttribute('onclick')?.includes('setPoll('+ps+','))p.classList.add('active');
    });
    /* toggles */
    document.getElementById('cfg-dry').checked=!!c.dry_run;
    document.getElementById('cfg-ai').checked=!!c.use_ai;
    document.getElementById('cfg-auto').checked=!!c.auto_trade;
    /* strategy sliders */
    const setSlider=(id,valId,val,fmt)=>{
      document.getElementById(id).value=val;
      document.getElementById(valId).textContent=fmt(val);
    };
    setSlider('cfg-atr-mult','v-atr-mult',st.atr_tolerance_multiplier??1.5,v=>parseFloat(v).toFixed(1)+'×');
    setSlider('cfg-sl-mult','v-sl-mult',st.sl_atr_mult??1.5,v=>parseFloat(v).toFixed(1)+'×');
    setSlider('cfg-tp-mult','v-tp-mult',st.tp_atr_mult??3.0,v=>parseFloat(v).toFixed(1)+'×');
    setSlider('cfg-pc-rr','v-pc-rr',st.partial_close_rr??1.0,v=>parseFloat(v).toFixed(1)+'R');
    setSlider('cfg-be-buf','v-be-buf',st.breakeven_buffer_pips??1.0,v=>parseFloat(v).toFixed(1)+'p');
    updateRR();
    configSynced=true;
  }
  document.getElementById('apply-btn').style.display=s.running?'inline-block':'none';
}

/* ── Tab nav ── */
function showTab(tab){
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{
    b.classList.toggle('active',['signals','thoughts','positions','history','performance'][i]===tab);
  });
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  activeTab=tab;
  if(tab==='history')loadHistory();
  if(tab==='positions')poll();
  if(tab==='performance')updatePerformancePanel();
}

/* ── Performance Metrics ── */
function updatePerformancePanel(){
  updatePerformanceKPIs();
  drawEquityCurve();
}

function updatePerformanceKPIs(){
  fetch('/bot/status')
    .then(r=>r.json())
    .then(s=>{
      const trades=s.trades||[];
      const winning=trades.filter(t=>t.pnl>0).length;
      const total=trades.length||1;
      const winRate=(winning/total*100).toFixed(1);
      
      const grossProfit=trades.filter(t=>t.pnl>0).reduce((a,t)=>a+t.pnl,0);
      const grossLoss=Math.abs(trades.filter(t=>t.pnl<0).reduce((a,t)=>a+t.pnl,0));
      const profitFactor=(grossProfit/(grossLoss||1)).toFixed(2);
      
      const totalPnL=trades.reduce((a,t)=>a+t.pnl,0);
      const returnPct=(totalPnL/10000*100).toFixed(2);
      
      const equity=trades.reduce((a,t)=>a+t.pnl,10000);
      const peak=trades.reduce((p,t,i)=>Math.max(p,trades.slice(0,i+1).reduce((a,x)=>a+x.pnl,10000)),10000);
      const drawdown=((peak-equity)/peak*100).toFixed(2);
      
      const avgWin=winning>0?(trades.filter(t=>t.pnl>0).reduce((a,t)=>a+t.pnl,0)/winning).toFixed(2):0;
      const avgLoss=total-winning>0?-(trades.filter(t=>t.pnl<0).reduce((a,t)=>a+t.pnl,0)/(total-winning)).toFixed(2):0;
      const avgratio=(avgLoss>0?(avgWin/avgLoss).toFixed(2):0);
      
      document.getElementById('kpi-winrate').textContent=winRate+'%';
      document.getElementById('kpi-profitfactor').textContent=profitFactor;
      document.getElementById('kpi-trades').textContent=total;
      document.getElementById('kpi-return').textContent=returnPct+'%';
      document.getElementById('kpi-drawdown').textContent=drawdown+'%';
      document.getElementById('kpi-sharpe').textContent=(winRate/10).toFixed(2);
      document.getElementById('kpi-avgratio').textContent=avgratio;
      document.getElementById('kpi-currentdd').textContent=drawdown+'%';
    })
    .catch(e=>console.log('Performance fetch error:',e));
}

function drawEquityCurve(){
  const canvas=document.getElementById('equityCanvas');
  const ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth;
  canvas.height=canvas.offsetHeight;
  
  fetch('/bot/status')
    .then(r=>r.json())
    .then(s=>{
      const trades=s.trades||[];
      const equity=[];
      let cum=10000;
      equity.push(cum);
      trades.forEach(t=>{
        cum+=t.pnl||0;
        equity.push(cum);
      });
      
      const w=canvas.width,h=canvas.height;
      const max=Math.max(...equity,10000);
      const min=Math.min(...equity,10000);
      const range=max-min||1;
      
      ctx.strokeStyle='#4dd9e5';
      ctx.lineWidth=2;
      ctx.beginPath();
      
      for(let i=0;i<equity.length;i++){
        const x=(i/(equity.length-1||1))*w;
        const y=h-((equity[i]-min)/range)*h*0.9-h*0.05;
        if(i===0)ctx.moveTo(x,y);
        else ctx.lineTo(x,y);
      }
      ctx.stroke();
      
      ctx.fillStyle='rgba(77,217,229,0.1)';
      ctx.lineTo(w,h*0.95);
      ctx.lineTo(0,h*0.95);
      ctx.closePath();
      ctx.fill();
      
      ctx.strokeStyle='var(--line)';
      ctx.lineWidth=1;
      ctx.setLineDash([4,4]);
      ctx.beginPath();
      ctx.moveTo(0,h*0.95);
      ctx.lineTo(w,h*0.95);
      ctx.stroke();
      ctx.setLineDash([]);
    })
    .catch(e=>console.log('Equity curve error:',e));
}

/* ── Status ── */
function updateStatus(s){
  const live=!!s.running;
  syncConfig(s);updateMt5(s);
  document.getElementById('dot').className='dot'+(live?' live':'');
  const t=document.getElementById('status-text');
  t.textContent=live?'live':'idle';
  t.style.color=live?'var(--green)':'var(--txt3)';
  document.getElementById('start-btn').style.display=live?'none':'inline-block';
  document.getElementById('stop-btn').style.display=live?'inline-block':'none';
  if(!live)stopPolling();
}

/* ── Signals ── */
function updateSignals(s){
  if (!s) { DEBUG.warn('updateSignals: empty status object'); return; }
  const all=s.all_decisions||{}, el=document.getElementById('signal-cards');
  DEBUG.log(`📊 Updating signals (${Object.keys(all).length} symbols)`);
  const entries=Object.entries(all);
  if(!entries.length){
    el.innerHTML='<div class="empty-state"><div class="icon">◈</div><div class="msg">Waiting for first scan…</div></div>';
    return;
  }
  el.className='sym-grid';
  el.innerHTML=entries.map(([sym,d])=>{
    const dir=(d.action||'neutral').toUpperCase();
    const dc=dir==='BUY'?'buy':dir==='SELL'?'sell':'neutral';
    const conf=Math.round((d.confidence||0)*100);
    const aiCls=d.approve===true?'ok':d.approve===false?'no':'wait';
    const aiTxt=d.approve===true?'AI ✓':d.approve===false?'AI ✗':'AI —';
    const env=(d.reason||'').match(/ENV\d[^.|]*/)?((d.reason||'').match(/(ENV\d[^.|]*)/)||[''])[0].substring(0,28):'—';
    const confColor=conf>=70?'var(--green)':conf>=50?'var(--amber)':'var(--txt3)';
    return `<div class="sig-card ${dc}">
      <div class="sig-header">
        <span class="sig-sym">${sym}</span>
        <div style="display:flex;gap:6px;align-items:center">
          <span class="badge ${dc}">${dir==='BUY'?'▲ BUY':dir==='SELL'?'▼ SELL':'— NEUTRAL'}</span>
          <span class="badge ${aiCls}">${aiTxt}</span>
        </div>
      </div>
      <div class="sig-body">
        <div class="sig-row">
          <div class="sig-stat"><span class="lbl">Confidence</span><span class="val" style="color:${confColor}">${conf}%</span></div>
          <div class="sig-stat"><span class="lbl">Source</span><span class="val" style="font-size:10px;color:var(--txt2)">${env||'—'}</span></div>
          <div class="sig-stat"><span class="lbl">AI Note</span><span class="val" style="font-size:10px;color:var(--txt3)">${(d.ai_reason||'—').substring(0,28)}</span></div>
        </div>
        <div class="conf-bar"><div class="conf-fill" style="width:${conf}%;background:${confColor}"></div></div>
        ${d.sl?`<div class="level-row">
          <div class="lvl"><span class="lbl">Stop Loss</span><span class="val r">${d.sl}</span></div>
          <div class="lvl"><span class="lbl">Take Profit</span><span class="val g">${d.tp}</span></div>
        </div>`:''}
      </div>
      <div class="sig-reason">${(d.reason||'Waiting…').substring(0,200)}</div>
    </div>`;
  }).join('');
}

/* ── Thoughts ── */
function appendThoughts(items){
  const c=document.getElementById('thoughts');
  items.forEach(t=>{
    thoughtCount++;
    const d=document.createElement('div');
    d.className='thought';d.dataset.src=t.source;
    const ts=new Date(t.ts).toLocaleTimeString(undefined,{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
    const conf=t.confidence!=null?`<span class="t-conf">${Math.round(t.confidence*100)}%</span>`:'';
    d.innerHTML=`<div class="t-head">
      <div class="t-meta"><span class="t-src">${t.source}</span><span class="t-sym">${t.symbol||'—'}</span><span class="t-stage">${t.stage}</span>${conf}</div>
      <span class="t-time">${ts}</span>
    </div>
    <div class="t-sum">${t.summary}</div>
    ${t.detail?`<div class="t-detail">${String(t.detail).substring(0,200)}</div>`:''}`;
    c.insertBefore(d,c.firstChild);
  });
  document.getElementById('thought-count').textContent=thoughtCount+' entries';
}

async function clearThoughts(){
  await fetch('/bot/thoughts/clear',{method:'POST'});
  document.getElementById('thoughts').innerHTML='';
  thoughtCount=0;document.getElementById('thought-count').textContent='0 entries';since=null;
}

/* ── Positions ── */
function updatePositions(positions){
  const el=document.getElementById('positions-content');
  
  /* Update symbol chips with position status */
  document.querySelectorAll('.sym-chip').forEach(chip=>chip.classList.remove('position-active'));
  if(positions&&positions.length){
    const activeSymbols=new Set(positions.map(p=>p.symbol?.toUpperCase()));
    document.querySelectorAll('.sym-chip').forEach(chip=>{
      if(activeSymbols.has(chip.textContent.trim().toUpperCase())){
        chip.classList.add('position-active');
      }
    });
  }
  
  if(!positions||!positions.length){el.innerHTML='<div class="pos-empty">No open positions</div>';return;}
  el.innerHTML=`<table class="pos-table"><thead><tr>
    <th>Ticket</th><th>Symbol</th><th>Dir</th><th>Vol</th><th>Entry</th><th>Current</th><th>SL</th><th>TP</th><th>P&L</th>
  </tr></thead><tbody>${positions.map(p=>{
    const pc=p.pnl>0?'g':p.pnl<0?'r':'';
    const dc=p.direction==='BUY'?'buy':'sell';
    return `<tr><td style="color:var(--txt3)">#${p.ticket}</td><td style="font-weight:500">${p.symbol}</td>
      <td><span class="badge ${dc}">${p.direction==='BUY'?'▲':'▼'} ${p.direction}</span></td>
      <td>${p.volume}</td><td>${p.entry}</td><td>${p.current||'—'}</td>
      <td class="r">${p.sl||'—'}</td><td class="g">${p.tp||'—'}</td>
      <td class="${pc}" style="font-size:12px">${p.pnl>0?'+':''}${p.pnl}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── History ── */
async function loadHistory(){
  const r=await fetch('/bot/history'), data=await r.json();
  const trades=data.trades||[], bal=data.account_balance;
  const sEl=document.getElementById('history-summary'), lEl=document.getElementById('history-list');
  if(!trades.length){sEl.innerHTML='';lEl.innerHTML='<div class="pos-empty">No trades recorded yet</div>';return;}
  const wins=trades.filter(t=>t.outcome==='WIN').length;
  const loss=trades.filter(t=>t.outcome==='LOSS').length;
  const total=trades.reduce((s,t)=>s+Number(t.pnl||0),0);
  const rrArr=trades.map(t=>Number(t.rr)).filter(v=>Number.isFinite(v)&&v>0);
  const avgRr=rrArr.length?rrArr.reduce((a,v)=>a+v,0)/rrArr.length:null;
  const wr=trades.length?(wins/trades.length*100).toFixed(0):0;
  sEl.innerHTML=(bal!=null?`<div class="stat-box"><div class="lbl">Balance</div><div class="val">$${bal.toFixed(2)}</div></div>`:'')+
    `<div class="stat-box"><div class="lbl">Net P&L</div><div class="val" style="color:${total>=0?'var(--green)':'var(--red)'}">${total>=0?'+':''}${total.toFixed(2)}</div></div>
    <div class="stat-box"><div class="lbl">Win Rate</div><div class="val" style="color:${wr>=50?'var(--green)':'var(--txt)'}">${wr}%</div></div>
    <div class="stat-box"><div class="lbl">Wins</div><div class="val g">${wins}</div></div>
    <div class="stat-box"><div class="lbl">Losses</div><div class="val r">${loss}</div></div>
    <div class="stat-box"><div class="lbl">Avg R:R</div><div class="val">${avgRr!=null?avgRr.toFixed(2)+'R':'—'}</div></div>`;
  lEl.innerHTML=trades.map(t=>{
    const dc=t.direction==='BUY'?'buy':'sell';
    const pc=t.pnl>0?'g':t.pnl<0?'r':'';
    const oc=t.outcome==='WIN'?'win':t.outcome==='LOSS'?'loss':'be';
    const ts=new Date(t.ts).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    return `<div class="hist-item ${oc}">
      <div class="hist-left">
        <div class="hist-sym">${t.symbol}<span class="badge ${dc}" style="font-size:9px">${t.direction}</span></div>
        <div class="hist-sub">${t.source||'strategy'} · ${ts}</div>
        <div class="hist-sub">${t.entry} → ${t.exit} · vol ${t.volume}</div>
      </div>
      <div><div class="hist-pnl ${pc}">${t.pnl>0?'+':''}${t.pnl}</div><div class="hist-rr">${t.rr?'R:R '+t.rr:'—'}</div></div>
    </div>`;
  }).join('');
}

/* ── Bot controls ── */
function getConfig(){
  const symbols=[...document.querySelectorAll('.sym-chip.active')].map(c=>c.textContent.trim());
  return{symbols,
    volume:parseFloat(document.getElementById('cfg-volume').value),
    poll_secs:parseFloat(document.getElementById('cfg-poll').value),
    dry_run:document.getElementById('cfg-dry').checked,
    use_ai:document.getElementById('cfg-ai').checked,
    auto_trade:document.getElementById('cfg-auto').checked,
    strategy:{
      atr_tolerance_multiplier:parseFloat(document.getElementById('cfg-atr-mult').value),
      sl_atr_mult:parseFloat(document.getElementById('cfg-sl-mult').value),
      tp_atr_mult:parseFloat(document.getElementById('cfg-tp-mult').value),
      partial_close_rr:parseFloat(document.getElementById('cfg-pc-rr').value),
      breakeven_buffer_pips:parseFloat(document.getElementById('cfg-be-buf').value)
    }};
}

async function startBot(){
  const r=await fetch('/bot/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(getConfig())});
  const d=await r.json();
  if(d.ok){startPolling();poll();}else alert('Error: '+(d.error||JSON.stringify(d)));
}
async function applyConfig(){
  const r=await fetch('/bot/update_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(getConfig())});
  const d=await r.json();
  if(d.ok)await poll();else alert('Error: '+(d.error||JSON.stringify(d)));
}
async function stopBot(){
  await fetch('/bot/stop',{method:'POST'});stopPolling();await poll();
}
function startPolling(){if(pollTimer)clearInterval(pollTimer);pollTimer=setInterval(poll,3000);}
function stopPolling(){if(pollTimer){clearInterval(pollTimer);pollTimer=null;}}

async function poll(){
  try{
    const[sr,tr]=await Promise.all([fetch('/bot/status'),fetch('/bot/ai_thoughts'+(since?'?since='+since:''))]);
    const s=await sr.json(),th=await tr.json();
    updateStatus(s);
    if(th.ok&&th.thoughts.length){appendThoughts(th.thoughts);since=th.thoughts[th.thoughts.length-1].ts;}
    if(activeTab==='signals')updateSignals(s);
    if(activeTab==='positions')updatePositions(s.open_positions||[]);
  }catch(e){console.error(e);}
}

// Initial bot status is now handled inside DOMContentLoaded above.

setInterval(()=>{
  if(activeTab==='performance')updatePerformancePanel();
}, 2000);

/* ── INITIALIZATION LOGGING ── */
(function(){
  DEBUG.log('🎯 Dashboard initialized');
  
  // Log tab switches
  const originalShowTab = window.showTab;
  window.showTab = function(tab) {
    DEBUG.log(`📑 Switching to tab: ${tab}`);
    return originalShowTab.call(this, tab);
  };
  
  // Log config changes
  document.addEventListener('change', function(e) {
    if (e.target.id?.startsWith('cfg-')) {
      DEBUG.log(`\u26a1 Config changed: ${e.target.id} = ${e.target.value}`, e.target);
    }
  });
  
  DEBUG.log('✅ All logging initialized');
})();
