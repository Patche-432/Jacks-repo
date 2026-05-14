/**
 * agents.js — Fortis Multi-Agent Portfolio Manager Dashboard
 *
 * A war-room-style operator interface for the 4-pair multi-agent system.
 * All 10 panels, 3 data sources, live updates every 3 s when tab is active.
 *
 * Endpoints consumed:
 *   /api/agent/matrix      → agent status, bots, heatmap, memory, directives
 *   /api/agent/portfolio   → risk dial, alignment, correlation, exposure
 *   /api/ollama/health     → LLM health panel
 *   /bot/history           → KPI computation (win rate, profit factor, etc.)
 *   /bot/signal/<sym>      → market environment map (CHoCH, env type, bias)
 *   /bot/status            → open positions for trade lifecycle
 *   /bot/ai_thoughts       → thought log for decision timeline
 */

/* ═══════════════════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════════════════ */
let _ag_init = false;
let _ag_timer = null;
let _ag_abortCtrl = null;   // AbortController for in-flight fetch batch
let _ag_matrix = null;
let _ag_port   = null;
let _ag_ollama = null;
let _ag_kpi    = null;           // computed from /bot/history
let _ag_signals = {};            // sym → signal
let _ag_timeline = [];           // rolling event buffer
let _ag_tl_seen  = new Set();

const PAIRS = ['EURUSD', 'GBPUSD', 'GBPJPY', 'EURJPY'];
const BOT_N  = { EURUSD: 'Bot 1', GBPUSD: 'Bot 2', GBPJPY: 'Bot 3', EURJPY: 'Bot 4' };
const FLAG   = { EURUSD: 'EU/US', GBPUSD: 'GB/US', GBPJPY: 'GB/JP', EURJPY: 'EU/JP' };
const ENV_LABELS = {
  'CHoCH-BUY@PDL':       'CHoCH BUY · PDL',
  'CHoCH-SELL@PDH':      'CHoCH SELL · PDH',
  'Continuation-BUY@PDH':'Cont. BUY · PDH',
  'Continuation-SELL@PDL':'Cont. SELL · PDL',
};

/* ═══════════════════════════════════════════════════════════════════════
   ENTRY POINT
══════════════════════════════════════════════════════════════════════ */
function initAgentsTab() {
  if (_ag_init) { _startPoll(); return; }
  _ag_init = true;
  const root = document.getElementById('tab-agents');
  if (!root) return;
  root.innerHTML = _shell();
  _wireForm();
  _startPoll();
}

function _startPoll() {
  if (_ag_timer) return;
  _fetchAll();
  _ag_timer = setInterval(_fetchAll, 3000);
}
function _stopPoll() {
  clearInterval(_ag_timer);
  _ag_timer = null;
  if (_ag_abortCtrl) { _ag_abortCtrl.abort(); _ag_abortCtrl = null; }
}

/* Pause when leaving Agents tab */
const _origShow = window.showTab;
window.showTab = function(tab, btn) {
  if (tab !== 'agents') _stopPoll();
  if (_origShow) _origShow(tab, btn);
};

/* ═══════════════════════════════════════════════════════════════════════
   DATA FETCH
══════════════════════════════════════════════════════════════════════ */
function _fetchAll() {
  if (_ag_abortCtrl) _ag_abortCtrl.abort();
  _ag_abortCtrl = new AbortController();
  const sig = _ag_abortCtrl.signal;

  const _f = url => fetch(url, {signal: sig}).then(r=>r.json()).catch(()=>null);

  const p = [
    _f('/api/agent/matrix'),
    _f('/api/agent/portfolio'),
    _f('/api/ollama/health'),
    _f('/bot/history'),
    _f('/bot/status'),
  ];
  const sigFetches = PAIRS.map(sym => _f(`/bot/signal/${sym}`));

  Promise.allSettled([...p, ...sigFetches]).then(results => {
    if (sig.aborted) return;   // user left the tab — discard results

    const [mx, pf, ol, hist, status, ...sigs] = results.map(r =>
      r.status === 'fulfilled' ? r.value : null
    );

    if (mx)   { _ag_matrix = mx; }
    if (pf)   { _ag_port   = pf; }
    if (ol)   { _ag_ollama = ol; }

    PAIRS.forEach((sym, i) => {
      if (sigs[i]) _ag_signals[sym] = sigs[i];
    });

    if (hist?.trades) {
      _ag_kpi = _computeKPIs(hist.trades, hist.account_balance);
    }

    _ingestTimeline(mx, status);
    _render();
  });
}

/* ═══════════════════════════════════════════════════════════════════════
   KPI COMPUTATION
══════════════════════════════════════════════════════════════════════ */
function _computeKPIs(trades, balance) {
  if (!trades || !trades.length) return null;
  const closed = trades.filter(t => t.pnl != null);
  if (!closed.length) return null;

  const wins   = closed.filter(t => (t.pnl || 0) > 0);
  const losses = closed.filter(t => (t.pnl || 0) <= 0);
  const winRate = wins.length / closed.length;

  const grossWin  = wins.reduce((s, t) => s + (t.pnl || 0), 0);
  const grossLoss = Math.abs(losses.reduce((s, t) => s + (t.pnl || 0), 0));
  const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0;

  // Net PnLs for Sharpe / Sortino
  const pnls = closed.map(t => t.pnl || 0);
  const mean = pnls.reduce((s, v) => s + v, 0) / pnls.length;
  const variance = pnls.reduce((s, v) => s + (v - mean) ** 2, 0) / pnls.length;
  const std = Math.sqrt(variance);
  const sharpe = std > 0 ? mean / std : 0;

  const downsidePnls = pnls.filter(p => p < 0);
  const downsideVar = downsidePnls.length
    ? downsidePnls.reduce((s, v) => s + v ** 2, 0) / downsidePnls.length : 0;
  const sortino = downsideVar > 0 ? mean / Math.sqrt(downsideVar) : 0;

  // Max drawdown (sequential)
  let peak = 0, maxDD = 0, running_pnl = 0;
  pnls.forEach(p => {
    running_pnl += p;
    if (running_pnl > peak) peak = running_pnl;
    const dd = peak > 0 ? (peak - running_pnl) / peak : 0;
    if (dd > maxDD) maxDD = dd;
  });

  const totalPnl = pnls.reduce((s, v) => s + v, 0);
  const recovery = maxDD > 0 ? totalPnl / (peak * maxDD || 1) : 0;

  return {
    winRate: winRate * 100,
    profitFactor,
    sharpe,
    sortino,
    maxDrawdownPct: maxDD * 100,
    recoveryFactor: recovery,
    totalTrades: closed.length,
    netPnl: totalPnl,
  };
}

/* ═══════════════════════════════════════════════════════════════════════
   TIMELINE INGESTION
══════════════════════════════════════════════════════════════════════ */
function _ingestTimeline(mx, status) {
  if (!mx) return;
  const hm = mx.heatmap || {};
  const ts = Date.now();

  // Detect new veto/override events from running tallies
  PAIRS.forEach(sym => {
    const nv = (hm.vetoes    || {})[sym] || 0;
    const no = (hm.overrides || {})[sym] || 0;
    const kv = `${sym}:v:${nv}`, ko = `${sym}:o:${no}`;
    if (nv > 0 && !_ag_tl_seen.has(kv)) {
      _ag_tl_seen.add(kv);
      _ag_timeline.unshift({ ts, sym, verdict: 'VETO',     conf: 0,  phase: 'orchestrator' });
    }
    if (no > 0 && !_ag_tl_seen.has(ko)) {
      _ag_tl_seen.add(ko);
      _ag_timeline.unshift({ ts, sym, verdict: 'OVERRIDE', conf: 0,  phase: 'orchestrator' });
    }
  });

  // Ingest open position entries as lifecycle events
  const positions = status?.open_positions || status?.bot?.open_trades || [];
  positions.forEach(pos => {
    const k = `entry:${pos.ticket}`;
    if (!_ag_tl_seen.has(k)) {
      _ag_tl_seen.add(k);
      _ag_timeline.unshift({ ts, sym: pos.symbol, verdict: 'ENTRY',
        conf: null, phase: 'strategy', sl: pos.sl, tp: pos.tp });
    }
  });

  if (_ag_timeline.length > 60) _ag_timeline.length = 60;
}

/* ═══════════════════════════════════════════════════════════════════════
   MAIN RENDER DISPATCH
══════════════════════════════════════════════════════════════════════ */
function _render() {
  if (!document.getElementById('tab-agents')?.classList.contains('active')) return;
  _renderHeader();
  _renderMatrix(_ag_matrix);
  _renderRiskDial(_ag_port);
  _renderBotDiag(_ag_matrix);
  _renderTimeline();
  _renderHeatmap(_ag_matrix);
  _renderEnvMap(_ag_signals, _ag_port);
  _renderLLMHealth(_ag_ollama, _ag_matrix);
  _renderDirectives(_ag_matrix);
  _renderKPIs(_ag_kpi, _ag_port);
  _renderLifecycle(_ag_timeline);
}

/* ═══════════════════════════════════════════════════════════════════════
   1 ─ HEADER STATUS BAR
══════════════════════════════════════════════════════════════════════ */
function _renderHeader() {
  const el = document.getElementById('ag-hdr-status');
  if (!el || !_ag_matrix) return;
  const running = _ag_matrix.running;
  const oll = _ag_ollama || {};
  const orchOk = running && oll.reachable && oll.model_loaded;
  const botsOk = running;
  el.innerHTML = [
    _hdrPill('BOT', running ? 'on' : 'off', running ? 'green' : 'dim'),
    _hdrPill('ORCHESTRATOR', orchOk ? 'LLM·READY' : oll.reachable ? 'NO MODEL' : 'OFFLINE', orchOk ? 'green' : 'amber'),
    _hdrPill('BOTS 1–4', botsOk ? 'DETERMINISTIC·READY' : 'IDLE', botsOk ? 'cyan' : 'dim'),
    _hdrPill('PAIRS', PAIRS.length + ' active', 'cyan'),
  ].join('');
}

function _hdrPill(label, val, col) {
  const colors = { green: '#50d963', amber: '#ffc107', cyan: '#4dd9e5', dim: '#6a7280' };
  const c = colors[col] || '#6a7280';
  return `<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
    border:1px solid ${c}22;border-radius:3px;margin-right:6px;font-size:9px;
    letter-spacing:.08em;background:${c}0d">
    <span style="color:${c}88;text-transform:uppercase">${label}</span>
    <span style="color:${c};font-weight:700">${val}</span>
  </span>`;
}

/* ═══════════════════════════════════════════════════════════════════════
   2 ─ AGENT MATRIX
══════════════════════════════════════════════════════════════════════ */
function _renderMatrix(d) {
  const el = document.getElementById('ag-matrix-body');
  if (!el || !d) return;

  const running  = d.running;
  const oll      = _ag_ollama || {};
  const verdicts = d.latest_verdicts || {};
  const orchConf = d.orch_confidence;

  const agents = [
    { id: 'orch', name: 'AGENT 0', role: 'ORCHESTRATOR', type: 'LLM',
      ok: running && oll.reachable && oll.model_loaded,
      warn: running && !(oll.reachable && oll.model_loaded),
      action: running ? 'orchestrating' : '—',
      conf: orchConf != null ? `${(orchConf*100).toFixed(0)}%` : '—',
      code: '⧡' },
    ...PAIRS.map((sym, i) => {
      const v = verdicts[sym] || {};
      const conf = v.confidence != null ? (v.confidence*100).toFixed(0)+'%' : '—';
      return { id: sym, name: `BOT ${i+1}`, role: sym, type: 'DET',
        ok: running, warn: false,
        action: v.action || 'hold',
        conf, atr: v.atr_profit?.toFixed(2),
        code: v.reason_code || '—' };
    }),
  ];

  el.innerHTML = agents.map((a, idx) => {
    const dotC = a.ok ? '#50d963' : a.warn ? '#ffc107' : '#6a7280';
    const dotGlow = a.ok ? '0 0 6px #50d96366' : '';
    const typeBg  = a.type === 'LLM' ? 'rgba(218,127,247,.15)' : 'rgba(77,217,229,.12)';
    const typeC   = a.type === 'LLM' ? '#da7ff7' : '#4dd9e5';
    const actionC = { close:'#ff6b6b', move_sl:'#4dd9e5', hold:'#6a7280',
                      orchestrating:'#50d963' }[a.action] || '#6a7280';
    return `<div class="ag-mrow ${idx===0?'ag-mrow-orch':''}" style="animation-delay:${idx*40}ms">
      <div style="display:flex;align-items:center;gap:8px">
        <span style="width:7px;height:7px;border-radius:50%;background:${dotC};
          box-shadow:${dotGlow};flex-shrink:0;display:inline-block"></span>
        <span class="ag-mrow-name">${a.name}</span>
      </div>
      <span class="ag-mrow-role">${a.role}</span>
      <span style="padding:2px 8px;border-radius:3px;font-size:9px;font-weight:700;
        background:${typeBg};color:${typeC}">${a.type}</span>
      <span style="font-weight:700;color:${actionC};font-size:11px;font-family:var(--mono)">${a.action}</span>
      <span style="font-family:var(--mono);font-size:11px;color:var(--txt2)">${a.conf}</span>
      <span style="font-size:9px;color:var(--txt3);font-family:var(--mono)">${a.code}</span>
    </div>`;
  }).join('');
}

/* ═══════════════════════════════════════════════════════════════════════
   3 ─ RISK DIAL
══════════════════════════════════════════════════════════════════════ */
function _renderRiskDial(pf) {
  const canvas = document.getElementById('ag-dial-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const pct = Math.min(pf?.risk_budget_pct || 0, 100);
  const cx = W/2, cy = H-14, R = Math.min(cx, cy+4) - 10;

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI + (i/10) * Math.PI;
    const inner = i % 5 === 0 ? R-12 : R-7;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a)*(R-2), cy + Math.sin(a)*(R-2));
    ctx.lineTo(cx + Math.cos(a)*inner, cy + Math.sin(a)*inner);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = i % 5 === 0 ? 1.5 : 1;
    ctx.stroke();
  }

  // Zone arcs
  const zones = [
    [0,    0.60, '#50d96366', '#50d963'],
    [0.60, 0.80, '#ffc10766', '#ffc107'],
    [0.80, 1.00, '#ff6b6b66', '#ff6b6b'],
  ];
  zones.forEach(([from, to, bg, stroke]) => {
    const aS = Math.PI + from*Math.PI;
    const aE = Math.PI + to*Math.PI;
    ctx.beginPath();
    ctx.arc(cx, cy, R-4, aS, aE);
    ctx.strokeStyle = bg;
    ctx.lineWidth = 14;
    ctx.stroke();
  });

  // Value arc
  const valFrac = pct / 100;
  const fillColor = pct >= 80 ? '#ff6b6b' : pct >= 60 ? '#ffc107' : '#50d963';
  ctx.beginPath();
  ctx.arc(cx, cy, R-4, Math.PI, Math.PI + valFrac*Math.PI);
  ctx.strokeStyle = fillColor;
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Needle
  const needleA = Math.PI + valFrac * Math.PI;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.cos(needleA)*(R-16), cy + Math.sin(needleA)*(R-16));
  ctx.strokeStyle = '#e8eef7';
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, 2*Math.PI);
  ctx.fillStyle = '#e8eef7';
  ctx.fill();

  // Centre text
  ctx.fillStyle = fillColor;
  ctx.font = `700 ${R*0.28}px var(--mono, monospace)`;
  ctx.textAlign = 'center';
  ctx.fillText(`${pct.toFixed(0)}%`, cx, cy - R*0.22);
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = `500 ${R*0.13}px var(--sans, sans-serif)`;
  ctx.fillText('RISK BUDGET', cx, cy - R*0.06);

  // Metric rows below dial
  const metricsEl = document.getElementById('ag-dial-metrics');
  if (!metricsEl || !pf) return;
  const pl = pf.daily_pl || 0;
  const plSign = pl >= 0 ? '+' : '';
  const align = pf.alignment_score || 0;
  const alignLabel = align > 0.3 ? '↑ Bull' : align < -0.3 ? '↓ Bear' : '⟶ Flat';
  const alignCol = align > 0.2 ? '#50d963' : align < -0.2 ? '#ff6b6b' : '#ffc107';
  const corrPct = Math.round((pf.correlation_load || 0) * 100);

  const metrics = [
    { k:'Daily P/L',  v: `${plSign}${pl.toFixed(2)}`,     c: pl>=0?'#50d963':'#ff6b6b' },
    { k:'Equity',     v: `${(pf.equity||0).toFixed(0)}`,  c: '#e8eef7' },
    { k:'Open',       v: `${pf.open_count||0}`,            c: '#4dd9e5' },
    { k:'Alignment',  v: alignLabel,                       c: alignCol },
    { k:'Correl.',    v: `${corrPct}%`,                    c: corrPct>60?'#ffc107':'#a0aab8' },
    { k:'Loss Limit', v: pf.daily_loss_limit ? `-${pf.daily_loss_limit.toFixed(0)}` : 'Off',
                      c: pf.daily_loss_breach ? '#ff6b6b' : '#6a7280' },
  ];

  metricsEl.innerHTML = metrics.map(m =>
    `<div class="ag-dial-metric">
      <span class="ag-dial-key">${m.k}</span>
      <span class="ag-dial-val" style="color:${m.c}">${m.v}</span>
    </div>`
  ).join('');
}

/* ═══════════════════════════════════════════════════════════════════════
   4 ─ PAIR-BOT DIAGNOSTICS
══════════════════════════════════════════════════════════════════════ */
function _renderBotDiag(d) {
  const el = document.getElementById('ag-bot-diag');
  if (!el || !d) return;
  const bots    = d.bots    || {};
  const verdicts = d.latest_verdicts || {};
  const running  = d.running;

  el.innerHTML = PAIRS.map((sym, i) => {
    const b   = bots[sym]    || {};
    const v   = verdicts[sym] || {};
    const conf = v.confidence != null ? v.confidence : null;
    const confPct = conf != null ? Math.round(conf*100) : null;
    const confColor = confPct==null?'#6a7280':confPct>=80?'#50d963':confPct>=60?'#ffc107':'#ff6b6b';
    const action = v.action || 'hold';
    const actionColors = { close:'#ff6b6b',move_sl:'#4dd9e5',hold:'#6a7280' };
    const aC = actionColors[action] || '#6a7280';
    const dotC = running ? '#50d963' : '#6a7280';
    const dotGlow = running ? '0 0 6px #50d96344' : '';

    // Trend / structure signals
    const sig = _ag_signals[sym] || {};
    const trend = sig.trend_intact != null ? (sig.trend_intact ? '✓ Intact' : '✗ Broken') : '—';
    const struct = sig.structure_broken != null ? (sig.structure_broken ? '✗ Broken' : '✓ OK') : '—';
    const trendC = sig.trend_intact ? '#50d963' : sig.trend_intact===false ? '#ff6b6b' : '#6a7280';
    const structC = sig.structure_broken ? '#ff6b6b' : '#50d963';

    const tunedBadge = b.tuned
      ? `<span style="padding:1px 6px;border-radius:3px;font-size:8px;font-weight:700;background:var(--cyan-bg);color:var(--cyan);letter-spacing:.05em">TUNED</span>`
      : `<span style="padding:1px 6px;border-radius:3px;font-size:8px;font-weight:700;background:var(--amber-bg);color:var(--amber);letter-spacing:.05em">DEFAULTS</span>`;

    return `<div class="ag-bot-card">
      <div class="ag-bot-hdr">
        <span style="width:6px;height:6px;border-radius:50%;background:${dotC};
          box-shadow:${dotGlow};flex-shrink:0;display:inline-block"></span>
        <span class="ag-bot-title">BOT ${i+1}</span>
        <span class="ag-bot-sym">${sym}</span>
        ${tunedBadge}
        <span style="padding:2px 8px;border-radius:3px;font-size:9px;font-weight:700;
          background:${aC}20;color:${aC};margin-left:auto">${action.toUpperCase()}</span>
      </div>
      <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.05em;margin:8px 0 4px;font-weight:600">Backtest Tuned Params</div>
      ${_diagRows([
        ['SL Mult',        b.sl_atr_mult!=null      ? b.sl_atr_mult+'× ATR'      : '—'],
        ['TP Mult',        b.tp_atr_mult!=null      ? b.tp_atr_mult+'× ATR'      : '—'],
        ['Trail Mult',     b.trail_atr_mult!=null   ? b.trail_atr_mult+'× ATR'   : '—'],
        ['Min ATR Tighten',b.min_atr_to_tighten!=null ? b.min_atr_to_tighten+'× ATR' : '—'],
        ['B/E Trigger',    b.partial_close_rr!=null ? b.partial_close_rr+'R'     : '—'],
        ['B/E Buffer',     b.be_buffer_pips!=null   ? b.be_buffer_pips+' pips'   : '—'],
      ])}
      <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.05em;margin:8px 0 4px;font-weight:600">Live Position</div>
      ${_diagRows([
        ['Last Action',    v.reason_code || '—'],
        ['Profit ATR',     v.atr_profit!=null ? v.atr_profit.toFixed(2)+'×' : '—'],
        ['Trend',          trend, trendC],
        ['Structure',      struct, structC],
      ])}
      ${confPct != null ? `
        <div class="ag-conf-wrap">
          <div class="ag-conf-track">
            <div class="ag-conf-fill" style="width:${confPct}%;background:${confColor}"></div>
          </div>
          <span class="ag-conf-label" style="color:${confColor}">${confPct}%</span>
        </div>` : ''}
    </div>`;
  }).join('');
}

function _diagRows(rows) {
  return `<div class="ag-diag-rows">` +
    rows.map(([k, v, c]) =>
      `<div class="ag-diag-row">
        <span class="ag-diag-key">${k}</span>
        <span class="ag-diag-val" style="${c?`color:${c}`:''}">
          ${v}
        </span>
      </div>`
    ).join('') +
  `</div>`;
}

/* ═══════════════════════════════════════════════════════════════════════
   5 ─ ORCHESTRATOR DECISION TIMELINE
══════════════════════════════════════════════════════════════════════ */
function _renderTimeline() {
  const el = document.getElementById('ag-timeline');
  if (!el) return;
  if (!_ag_timeline.length) {
    el.innerHTML = `<div class="ag-tl-empty">No events recorded — start the bot with open positions</div>`;
    return;
  }
  const verdictMeta = {
    VETO:     { c:'#ff6b6b',  icon:'⊗', label:'VETO' },
    OVERRIDE: { c:'#ffc107',  icon:'↺', label:'OVERRIDE' },
    APPROVE:  { c:'#50d963',  icon:'✓', label:'APPROVE' },
    ENTRY:    { c:'#4dd9e5',  icon:'↗', label:'ENTRY' },
    EXIT:     { c:'#a0aab8',  icon:'↙', label:'EXIT' },
  };

  el.innerHTML = _ag_timeline.slice(0, 40).map((ev, i) => {
    const m   = verdictMeta[ev.verdict] || { c:'#6a7280', icon:'·', label: ev.verdict };
    const t   = new Date(ev.ts);
    const ts  = `${String(t.getHours()).padStart(2,'0')}:${String(t.getMinutes()).padStart(2,'0')}:${String(t.getSeconds()).padStart(2,'0')}`;
    const phase = ev.phase || 'orchestrator';
    const phaseC = { strategy:'#4dd9e5', orchestrator:'#ffc107', execution:'#50d963' }[phase] || '#6a7280';
    return `<div class="ag-tl-row" style="animation-delay:${i*12}ms">
      <span class="ag-tl-ts">${ts}</span>
      <span class="ag-tl-phase" style="color:${phaseC}">${phase.toUpperCase().slice(0,4)}</span>
      <span class="ag-tl-sym">${ev.sym||'—'}</span>
      <span style="font-weight:700;color:${m.c};font-family:var(--mono)">${m.icon} ${m.label}</span>
      ${ev.sl ? `<span class="ag-tl-detail">SL ${ev.sl}</span>` : '<span></span>'}
    </div>`;
  }).join('');
}

/* ═══════════════════════════════════════════════════════════════════════
   6 ─ DISAGREEMENT HEATMAP + MEMORY
══════════════════════════════════════════════════════════════════════ */
function _renderHeatmap(d) {
  const el = document.getElementById('ag-heatmap');
  if (!el || !d) return;
  const hm  = d.heatmap || {};
  const vet = hm.vetoes    || {};
  const ovr = hm.overrides || {};
  const maxV = Math.max(1, ...Object.values(vet));
  const maxO = Math.max(1, ...Object.values(ovr));

  const _cell = (n, max, hue) => {
    const a = n === 0 ? 0.03 : 0.08 + (n/max)*0.65;
    const c = n === 0 ? '#6a7280' : hue;
    return `<div class="ag-hm-cell" style="background:${hue}${n?Math.round(a*255).toString(16).padStart(2,'0'):'0a'};color:${c};font-weight:${n>0?'700':'400'}">${n||'·'}</div>`;
  };

  el.innerHTML = `
    <div class="ag-hm-grid">
      <div class="ag-hm-corner"></div>
      ${PAIRS.map(p=>`<div class="ag-hm-head">${p}</div>`).join('')}
      <div class="ag-hm-row-label" style="color:#ff6b6b">VETOES</div>
      ${PAIRS.map(p=>_cell(vet[p]||0, maxV, '#ff6b6b')).join('')}
      <div class="ag-hm-row-label" style="color:#ffc107">OVERRIDES</div>
      ${PAIRS.map(p=>_cell(ovr[p]||0, maxO, '#ffc107')).join('')}
    </div>`;

  const memEl = document.getElementById('ag-memory');
  if (memEl) memEl.textContent = d.memory_summary || 'No prior decisions this session.';
}

/* ═══════════════════════════════════════════════════════════════════════
   7 ─ MARKET ENVIRONMENT MAP
══════════════════════════════════════════════════════════════════════ */
function _renderEnvMap(signals, pf) {
  const el = document.getElementById('ag-env-map');
  if (!el) return;
  const exp = pf?.pair_exposure || {};

  el.innerHTML = PAIRS.map((sym, i) => {
    const s = signals[sym] || {};
    const src = s.signal_source || s.signal_source;
    const envLabel = ENV_LABELS[src] || (s.signal && s.signal !== 'neutral' ? src : '— Watching');
    const bias = s.signal || 'neutral';
    const biasC = bias==='BUY'?'#50d963':bias==='SELL'?'#ff6b6b':'#6a7280';
    const conf  = s.confidence != null ? s.confidence : s.ai_confidence;
    const confPct = conf!=null ? `${typeof conf==='number'&&conf<=1?(conf*100).toFixed(0):conf.toFixed(0)}%` : '—';
    const e = exp[sym] || {};
    const hasPos = e.lots > 0;
    const choch = s.choch_detected || (src||'').includes('CHoCH');

    return `<div class="ag-env-card" style="animation-delay:${i*30}ms">
      <div class="ag-env-top">
        <span class="ag-env-sym">${sym}</span>
        <span class="ag-env-flag">${FLAG[sym]||''}</span>
        ${hasPos ? `<span style="padding:1px 6px;border-radius:2px;font-size:8px;
          font-weight:700;background:#4dd9e540;color:#4dd9e5;margin-left:auto">OPEN</span>` : ''}
      </div>
      <div class="ag-env-bias" style="color:${biasC}">${bias.toUpperCase()}</div>
      <div class="ag-env-label">${envLabel}</div>
      <div class="ag-env-meta">
        ${choch ? `<span class="ag-env-choch">CHoCH</span>` : ''}
        <span class="ag-env-conf">conf ${confPct}</span>
        ${e.sl_pips ? `<span class="ag-env-sl">SL ${e.sl_pips.toFixed(0)}p</span>` : ''}
      </div>
    </div>`;
  }).join('');
}

/* ═══════════════════════════════════════════════════════════════════════
   8 ─ LLM HEALTH PANEL
══════════════════════════════════════════════════════════════════════ */
function _renderLLMHealth(h, mx) {
  const el = document.getElementById('ag-llm-health');
  if (!el) return;

  const reach    = h?.reachable;
  const loaded   = h?.model_loaded;
  const latMs    = h?.latency_ms;
  const jsonOk   = h?.json_valid;
  const orchConf = mx?.orch_confidence;
  const latMax   = 4000;
  const latFrac  = latMs ? Math.min(latMs/latMax, 1) : 0;
  const latColor = !latMs ? '#6a7280' : latMs>2500?'#ff6b6b':latMs>800?'#ffc107':'#50d963';

  el.innerHTML = `
    <div class="ag-llm-rows">
      ${_llmRow('Reachable',   reach ? '✓ YES' : '✗ NO',     reach ? '#50d963':'#ff6b6b')}
      ${_llmRow('Model',       h?.model || '—',               '#a0aab8')}
      ${_llmRow('Model Loaded',loaded ? '✓ loaded':'✗ missing', loaded?'#50d963':'#ffc107')}
      ${_llmRow('Last JSON',   jsonOk===true?'✓ valid':jsonOk===false?'✗ invalid':'—',
                               jsonOk?'#50d963':jsonOk===false?'#ff6b6b':'#6a7280')}
      ${_llmRow('Orch Conf',   orchConf!=null?`${(orchConf*100).toFixed(0)}%`:'—', '#a0aab8')}
      ${_llmRow('Error',       h?.error ? h.error.slice(0,40)+'…' : 'none',
                               h?.error?'#ff6b6b':'#6a7280')}
    </div>
    <div class="ag-llm-lat-wrap">
      <div class="ag-llm-lat-label">
        <span style="color:var(--txt3);font-size:9px">LATENCY</span>
        <span style="font-family:var(--mono);font-size:10px;color:${latColor}">${latMs!=null?latMs+'ms':'—'}</span>
      </div>
      <div class="ag-llm-lat-track">
        <div class="ag-llm-lat-fill" style="width:${latFrac*100}%;background:${latColor}"></div>
      </div>
    </div>`;
}

function _llmRow(k, v, c) {
  return `<div class="ag-llm-row">
    <span class="ag-llm-key">${k}</span>
    <span class="ag-llm-val" style="color:${c};max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${v}</span>
  </div>`;
}

/* ═══════════════════════════════════════════════════════════════════════
   9 ─ GLOBAL DIRECTIVES
══════════════════════════════════════════════════════════════════════ */
function _renderDirectives(d) {
  const el = document.getElementById('ag-dir-active');
  if (!el || !d) return;
  const dir = d.directives || {};
  const freezeList = dir.freeze || [];
  const closeList  = dir.close  || [];
  const all = [
    ...freezeList.map(s => `<span class="ag-dir-tag ag-dir-freeze">🧊 FREEZE: ${s}</span>`),
    ...closeList.map(s  => `<span class="ag-dir-tag ag-dir-close">🔴 FORCE·CLOSE: ${s}</span>`),
  ];
  el.innerHTML = all.length
    ? all.join('')
    : `<span class="ag-dir-none">No active directives</span>`;

  if (dir.notes) {
    const notesEl = document.getElementById('ag-dir-note-display');
    if (notesEl) notesEl.textContent = `"${dir.notes}"`;
  }
}

function _wireForm() {
  // Wired via global _submitDir() called from onclick
}

window._submitDir = function() {
  const freeze = (document.getElementById('ag-in-freeze')?.value||'')
    .split(/[\s,]+/).filter(Boolean).map(s=>s.toUpperCase());
  const close_ = (document.getElementById('ag-in-close')?.value||'')
    .split(/[\s,]+/).filter(Boolean).map(s=>s.toUpperCase());
  const maxExp = parseFloat(document.getElementById('ag-in-exp')?.value||'0')||0;
  const notes  = document.getElementById('ag-in-notes')?.value||'';
  const msg    = document.getElementById('ag-dir-msg');
  if (msg) { msg.style.color='#ffc107'; msg.textContent='Applying…'; }

  fetch('/api/agent/directives', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ freeze, close: close_, max_exposure_pct: maxExp, notes }),
  }).then(r=>r.json()).then(r => {
    if (msg) {
      msg.style.color = r.ok ? '#50d963' : '#ff6b6b';
      msg.textContent = r.ok ? `✓ ${r.summary}` : `✗ ${r.error}`;
      setTimeout(() => { if (msg) msg.textContent=''; }, 5000);
    }
    if (r.ok) _fetchAll();
  }).catch(e => {
    if (msg) { msg.style.color='#ff6b6b'; msg.textContent=`✗ ${e.message}`; }
  });
};

/* ═══════════════════════════════════════════════════════════════════════
   10 ─ PERFORMANCE KPIs
══════════════════════════════════════════════════════════════════════ */
function _renderKPIs(kpi, pf) {
  const el = document.getElementById('ag-kpis');
  if (!el) return;
  const pl = pf?.daily_pl || 0;
  const dd = pf && pf.balance > 0
    ? Math.min(0, ((pf.equity||0) - pf.balance) / pf.balance * 100)
    : 0;

  const metrics = kpi ? [
    { k:'Win Rate',       v: `${kpi.winRate.toFixed(1)}%`,
      c: kpi.winRate>=60?'#50d963':kpi.winRate>=40?'#ffc107':'#ff6b6b' },
    { k:'Profit Factor',  v: isFinite(kpi.profitFactor)?kpi.profitFactor.toFixed(2):'∞',
      c: kpi.profitFactor>=1.5?'#50d963':kpi.profitFactor>=1?'#ffc107':'#ff6b6b' },
    { k:'Sharpe',         v: kpi.sharpe.toFixed(2),
      c: kpi.sharpe>=1?'#50d963':kpi.sharpe>=0?'#ffc107':'#ff6b6b' },
    { k:'Sortino',        v: kpi.sortino.toFixed(2),
      c: kpi.sortino>=1?'#50d963':kpi.sortino>=0?'#ffc107':'#ff6b6b' },
    { k:'Max Drawdown',   v: `-${kpi.maxDrawdownPct.toFixed(1)}%`,
      c: kpi.maxDrawdownPct>10?'#ff6b6b':kpi.maxDrawdownPct>5?'#ffc107':'#50d963' },
    { k:'Recovery',       v: kpi.recoveryFactor.toFixed(2),
      c: kpi.recoveryFactor>=1?'#50d963':'#ffc107' },
    { k:'Total Trades',   v: String(kpi.totalTrades), c: '#a0aab8' },
    { k:'Net P/L',        v: `${kpi.netPnl>=0?'+':''}${kpi.netPnl.toFixed(2)}`,
      c: kpi.netPnl>=0?'#50d963':'#ff6b6b' },
  ] : [
    { k:'Daily P/L',   v:`${pl>=0?'+':''}${pl.toFixed(2)}`,c:pl>=0?'#50d963':'#ff6b6b'},
    { k:'Equity',      v:`${(pf?.equity||0).toFixed(0)}`,  c:'#e8eef7' },
    { k:'Drawdown',    v:`${dd.toFixed(1)}%`,c:dd<-5?'#ff6b6b':dd<-2?'#ffc107':'#50d963'},
    { k:'Risk Budget', v:`${(pf?.risk_budget_pct||0).toFixed(0)}%`,
      c:(pf?.risk_budget_pct||0)>80?'#ff6b6b':'#a0aab8' },
  ];

  el.innerHTML = metrics.map(m =>
    `<div class="ag-kpi">
      <span class="ag-kpi-k">${m.k}</span>
      <span class="ag-kpi-v" style="color:${m.c}">${m.v}</span>
    </div>`
  ).join('');
}

/* ═══════════════════════════════════════════════════════════════════════
   11 ─ TRADE LIFECYCLE VIEW
══════════════════════════════════════════════════════════════════════ */
function _renderLifecycle(events) {
  const el = document.getElementById('ag-lifecycle');
  if (!el) return;
  if (!events.length) {
    el.innerHTML = `<div class="ag-lc-empty">Lifecycle events appear here when trades are active</div>`;
    return;
  }

  // Group events by symbol to show the lifecycle chain
  const bySymbol = {};
  events.forEach(ev => {
    if (!bySymbol[ev.sym]) bySymbol[ev.sym] = [];
    bySymbol[ev.sym].push(ev);
  });

  const phaseOrder = ['strategy','orchestrator','execution','exit'];
  const phaseLabels = { strategy:'Strategy', orchestrator:'Orchestrator', execution:'Execution', exit:'Exit' };
  const phaseColors = { strategy:'#4dd9e5', orchestrator:'#ffc107', execution:'#50d963', exit:'#a0aab8' };

  el.innerHTML = Object.entries(bySymbol).slice(0,4).map(([sym, evs]) =>
    `<div class="ag-lc-row">
      <span class="ag-lc-sym">${sym}</span>
      <div class="ag-lc-chain">
        ${evs.slice(-5).reverse().map(ev => {
          const ph = ev.phase || 'orchestrator';
          const c  = phaseColors[ph] || '#6a7280';
          return `<div class="ag-lc-node" style="border-color:${c}22">
            <span class="ag-lc-node-phase" style="color:${c}">${phaseLabels[ph]||ph}</span>
            <span class="ag-lc-node-verdict" style="color:${
              ev.verdict==='VETO'?'#ff6b6b':ev.verdict==='OVERRIDE'?'#ffc107':'#50d963'
            }">${ev.verdict}</span>
          </div>`;
        }).join('<span class="ag-lc-arrow">→</span>')}
      </div>
    </div>`
  ).join('');
}

/* ═══════════════════════════════════════════════════════════════════════
   SHELL HTML — full layout
══════════════════════════════════════════════════════════════════════ */
function _shell() {
  return `
<style>
/* ─── Reset / scope ─────────────────────────── */
#tab-agents{flex-direction:column;gap:12px;padding:14px;overflow-y:auto;
  background:var(--bg0)}
#tab-agents.active{display:flex}

/* ─── Header ────────────────────────────────── */
#ag-header{display:flex;align-items:center;justify-content:space-between;
  padding:8px 12px;background:var(--bg1);border:1px solid var(--line);border-radius:8px}
.ag-header-title{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--txt);display:flex;align-items:center;gap:8px}
.ag-header-title-icon{color:var(--amber);font-size:14px}

/* ─── Grid layouts ──────────────────────────── */
.ag-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ag-grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.ag-grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:1100px){.ag-grid-4{grid-template-columns:repeat(2,1fr)}}
@media(max-width:860px){.ag-grid-2,.ag-grid-3,.ag-grid-4{grid-template-columns:1fr}}

/* ─── Card ──────────────────────────────────── */
.ag-card{background:var(--bg1);border:1px solid var(--line);border-radius:9px;
  padding:12px;display:flex;flex-direction:column;gap:10px;
  animation:ag-fadein .35s ease both}
@keyframes ag-fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.ag-card-hdr{display:flex;align-items:center;justify-content:space-between}
.ag-card-title{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--txt3);display:flex;align-items:center;gap:6px}
.ag-card-icon{font-size:13px}
.ag-card-sub{font-size:9px;color:var(--txt3)}

/* ─── Agent Matrix ──────────────────────────── */
#ag-matrix-body{display:flex;flex-direction:column;gap:1px}
.ag-mrow{display:grid;grid-template-columns:auto auto auto 1fr auto auto;
  gap:10px;align-items:center;padding:7px 10px;border-radius:5px;
  background:var(--bg0);border:1px solid transparent;
  transition:border-color .2s;font-size:11px;animation:ag-fadein .3s ease both}
.ag-mrow:hover{border-color:var(--line)}
.ag-mrow-orch{border-color:var(--line);background:var(--bg2)}
.ag-mrow-name{font-weight:700;font-family:var(--mono);color:var(--txt);font-size:11px;
  letter-spacing:.06em;min-width:60px}
.ag-mrow-role{font-size:9px;color:var(--txt3);min-width:90px;font-family:var(--mono)}

/* ─── Risk Dial ─────────────────────────────── */
.ag-dial-wrap{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap}
#ag-dial-canvas{flex-shrink:0}
#ag-dial-metrics{display:grid;grid-template-columns:1fr 1fr;gap:5px 16px;flex:1}
.ag-dial-metric{display:flex;flex-direction:column;gap:2px}
.ag-dial-key{font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.06em}
.ag-dial-val{font-size:13px;font-family:var(--mono);font-weight:700}

/* ─── Bot diagnostics ───────────────────────── */
.ag-bot-card{background:var(--bg0);border:1px solid var(--line);border-radius:7px;
  padding:10px;display:flex;flex-direction:column;gap:8px;
  animation:ag-fadein .3s ease both}
.ag-bot-hdr{display:flex;align-items:center;gap:6px}
.ag-bot-title{font-weight:700;font-family:var(--mono);color:var(--txt);font-size:11px}
.ag-bot-sym{font-size:9px;color:var(--txt3);font-family:var(--mono)}
.ag-diag-rows{display:flex;flex-direction:column;gap:2px}
.ag-diag-row{display:flex;justify-content:space-between;align-items:center;
  padding:2px 0;border-bottom:1px solid rgba(255,255,255,.03);font-size:10px}
.ag-diag-row:last-child{border-bottom:none}
.ag-diag-key{color:var(--txt3)}
.ag-diag-val{font-family:var(--mono);font-weight:500;color:var(--txt2)}
.ag-conf-wrap{display:flex;align-items:center;gap:8px;margin-top:4px}
.ag-conf-track{flex:1;height:3px;background:var(--line);border-radius:2px;overflow:hidden}
.ag-conf-fill{height:100%;border-radius:2px;transition:width .5s ease}
.ag-conf-label{font-size:9px;font-family:var(--mono);min-width:28px;text-align:right}

/* ─── Timeline ──────────────────────────────── */
.ag-tl-wrap{overflow-y:auto;max-height:240px;display:flex;flex-direction:column;gap:0}
.ag-tl-row{display:grid;grid-template-columns:54px 44px 68px 1fr auto;
  gap:8px;align-items:center;padding:5px 10px;border-bottom:1px solid rgba(255,255,255,.04);
  font-size:10px;animation:ag-fadein .2s ease both}
.ag-tl-row:last-child{border-bottom:none}
.ag-tl-ts{font-family:var(--mono);color:var(--txt3);font-size:9px}
.ag-tl-phase{font-size:8px;font-weight:700;letter-spacing:.05em}
.ag-tl-sym{font-weight:700;font-family:var(--mono);color:var(--txt2)}
.ag-tl-detail{font-size:9px;color:var(--txt3);font-family:var(--mono)}
.ag-tl-empty{color:var(--txt3);font-size:10px;padding:16px;text-align:center}

/* ─── Heatmap ───────────────────────────────── */
.ag-hm-grid{display:grid;grid-template-columns:80px repeat(4,1fr);
  gap:0;border:1px solid var(--line);border-radius:7px;overflow:hidden;font-size:10px}
.ag-hm-corner,.ag-hm-head{background:var(--bg2);padding:6px 8px;
  font-size:8px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:var(--txt3);text-align:center;border-bottom:1px solid var(--line)}
.ag-hm-row-label{padding:8px;background:var(--bg2);font-size:8px;font-weight:700;
  letter-spacing:.07em;display:flex;align-items:center;border-bottom:1px solid rgba(255,255,255,.04)}
.ag-hm-cell{padding:8px 4px;text-align:center;font-family:var(--mono);font-weight:700;
  border-bottom:1px solid rgba(255,255,255,.04);border-right:1px solid rgba(255,255,255,.04);
  transition:background .4s}
.ag-hm-cell:last-child{border-right:none}
.ag-memory-text{font-size:10px;font-family:var(--mono);color:var(--txt3);line-height:1.7;
  white-space:pre-wrap;background:var(--bg0);border-radius:5px;padding:8px;
  max-height:90px;overflow-y:auto;margin-top:6px}

/* ─── Market env map ────────────────────────── */
.ag-env-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
.ag-env-card{background:var(--bg0);border:1px solid var(--line);border-radius:7px;
  padding:10px;display:flex;flex-direction:column;gap:5px;
  animation:ag-fadein .3s ease both;transition:border-color .2s}
.ag-env-card:hover{border-color:var(--line2)}
.ag-env-top{display:flex;align-items:center;gap:6px}
.ag-env-sym{font-weight:700;font-family:var(--mono);font-size:11px;color:var(--txt)}
.ag-env-flag{font-size:9px;color:var(--txt3)}
.ag-env-bias{font-size:18px;font-weight:700;font-family:var(--mono);letter-spacing:.04em}
.ag-env-label{font-size:9px;color:var(--txt3);font-family:var(--mono)}
.ag-env-meta{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:2px}
.ag-env-choch{padding:2px 6px;border-radius:3px;font-size:8px;font-weight:700;
  background:rgba(218,127,247,.15);color:#da7ff7;letter-spacing:.06em}
.ag-env-conf{font-size:9px;color:var(--txt3);font-family:var(--mono)}
.ag-env-sl{font-size:9px;color:var(--txt3);font-family:var(--mono)}

/* ─── LLM Health ────────────────────────────── */
.ag-llm-rows{display:flex;flex-direction:column;gap:0}
.ag-llm-row{display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:10px}
.ag-llm-row:last-child{border-bottom:none}
.ag-llm-key{color:var(--txt3);font-size:9px}
.ag-llm-val{font-family:var(--mono);font-size:10px}
.ag-llm-lat-wrap{margin-top:8px;display:flex;flex-direction:column;gap:4px}
.ag-llm-lat-label{display:flex;justify-content:space-between}
.ag-llm-lat-track{height:4px;background:var(--line);border-radius:2px;overflow:hidden}
.ag-llm-lat-fill{height:100%;border-radius:2px;transition:width .5s ease}

/* ─── Directives ────────────────────────────── */
.ag-dir-tags{display:flex;flex-wrap:wrap;gap:5px;min-height:20px;margin-bottom:6px}
.ag-dir-tag{padding:3px 10px;border-radius:3px;font-size:9px;font-weight:700;
  font-family:var(--mono);letter-spacing:.04em}
.ag-dir-freeze{background:rgba(255,193,7,.12);color:#ffc107;border:1px solid rgba(255,193,7,.25)}
.ag-dir-close{background:rgba(255,107,107,.12);color:#ff6b6b;border:1px solid rgba(255,107,107,.25)}
.ag-dir-none{font-size:10px;color:var(--txt3)}
.ag-dir-note{font-size:9px;color:var(--txt3);font-style:italic;margin-bottom:8px}
.ag-dir-form-inner{display:flex;flex-direction:column;gap:6px;
  padding-top:8px;border-top:1px solid var(--line)}
.ag-dir-label{font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em;
  margin-bottom:2px}
.ag-dir-input{width:100%;background:var(--bg0);border:1px solid var(--line);border-radius:4px;
  color:var(--txt);font-size:10px;font-family:var(--mono);padding:5px 8px;outline:none}
.ag-dir-input:focus{border-color:var(--cyan)}
.ag-dir-apply{align-self:flex-start;padding:5px 14px;background:var(--cyan-bg);
  border:1px solid var(--cyan);border-radius:4px;color:var(--cyan);font-size:9px;
  font-weight:700;letter-spacing:.07em;cursor:pointer;transition:background .15s}
.ag-dir-apply:hover{background:rgba(77,217,229,.22)}
.ag-dir-msg{font-size:9px;min-height:14px;font-family:var(--mono)}

/* ─── KPIs ──────────────────────────────────── */
.ag-kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
@media(max-width:900px){.ag-kpi-grid{grid-template-columns:repeat(2,1fr)}}
.ag-kpi{background:var(--bg0);border:1px solid var(--line);border-radius:6px;
  padding:10px 12px;display:flex;flex-direction:column;gap:3px}
.ag-kpi-k{font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em}
.ag-kpi-v{font-size:16px;font-family:var(--mono);font-weight:700}

/* ─── Lifecycle ─────────────────────────────── */
.ag-lc-wrap{display:flex;flex-direction:column;gap:8px;overflow-y:auto;max-height:120px}
.ag-lc-row{display:flex;align-items:center;gap:10px;padding:5px 0;
  border-bottom:1px solid rgba(255,255,255,.04)}
.ag-lc-sym{font-weight:700;font-family:var(--mono);font-size:10px;color:var(--txt);
  min-width:52px}
.ag-lc-chain{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.ag-lc-node{padding:3px 8px;border-radius:4px;border:1px solid transparent;
  display:flex;flex-direction:column;align-items:center;gap:1px}
.ag-lc-node-phase{font-size:8px;letter-spacing:.05em}
.ag-lc-node-verdict{font-size:10px;font-weight:700;font-family:var(--mono)}
.ag-lc-arrow{color:var(--txt3);font-size:11px}
.ag-lc-empty{color:var(--txt3);font-size:10px;text-align:center;padding:12px}
</style>

<!-- ═══ HEADER STATUS BAR ═══ -->
<div id="ag-header">
  <div class="ag-header-title">
    <span class="ag-header-title-icon">⧡</span>
    FORTIS · MULTI-AGENT PORTFOLIO MANAGER
  </div>
  <div id="ag-hdr-status"></div>
</div>

<!-- ═══ ROW 1: Agent Matrix + Risk Dial ═══ -->
<div class="ag-grid-2">
  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">🔲</span> Agent Matrix</span>
      <span class="ag-card-sub">1 orchestrator · 4 bots</span>
    </div>
    <div style="display:grid;grid-template-columns:auto auto auto 1fr auto auto;
      gap:10px;padding:4px 10px;border-bottom:1px solid var(--line);margin-bottom:2px">
      ${['AGENT','ROLE','TYPE','LAST ACTION','CONF','CODE'].map(h=>
        `<span style="font-size:8px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--txt3)">${h}</span>`
      ).join('')}
    </div>
    <div id="ag-matrix-body"></div>
  </div>

  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">🎯</span> Portfolio Risk Dial</span>
    </div>
    <div class="ag-dial-wrap">
      <canvas id="ag-dial-canvas" width="220" height="130"></canvas>
      <div id="ag-dial-metrics"></div>
    </div>
  </div>
</div>

<!-- ═══ ROW 2: Pair-Bot Diagnostics ═══ -->
<div class="ag-card">
  <div class="ag-card-hdr">
    <span class="ag-card-title"><span class="ag-card-icon">🤖</span> Pair-Bot Diagnostics</span>
    <span class="ag-card-sub">Deterministic · no LLM</span>
  </div>
  <div class="ag-grid-4" id="ag-bot-diag"></div>
</div>

<!-- ═══ ROW 3: Timeline + Heatmap ═══ -->
<div class="ag-grid-2">
  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">⏱</span> Orchestrator Decision Timeline</span>
    </div>
    <div class="ag-tl-wrap" id="ag-timeline"></div>
  </div>

  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">🔥</span> Disagreement Heatmap</span>
      <span class="ag-card-sub">session totals</span>
    </div>
    <div id="ag-heatmap"></div>
    <div class="ag-memory-text" id="ag-memory">No prior decisions this session.</div>
  </div>
</div>

<!-- ═══ ROW 4: Env Map + LLM Health ═══ -->
<div class="ag-grid-2">
  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">🗺</span> Market Environment Map</span>
    </div>
    <div class="ag-env-grid" id="ag-env-map"></div>
  </div>

  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">🧠</span> LLM Health Panel</span>
      <span class="ag-card-sub">Agent 0 · Ollama</span>
    </div>
    <div id="ag-llm-health"></div>
  </div>
</div>

<!-- ═══ ROW 5: Directives + KPIs ═══ -->
<div class="ag-grid-2">
  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">⚑</span> Global Directives</span>
    </div>
    <div class="ag-dir-tags" id="ag-dir-active"></div>
    <div class="ag-dir-note" id="ag-dir-note-display"></div>
    <div class="ag-dir-form-inner">
      <div>
        <div class="ag-dir-label">Freeze symbols (hold only — no tighten/close)</div>
        <input class="ag-dir-input" id="ag-in-freeze" placeholder="e.g. GBPJPY EURJPY">
      </div>
      <div>
        <div class="ag-dir-label">Force close symbols (emergency exit)</div>
        <input class="ag-dir-input" id="ag-in-close" placeholder="e.g. GBPUSD">
      </div>
      <div>
        <div class="ag-dir-label">Max exposure % (0 = disabled)</div>
        <input class="ag-dir-input" id="ag-in-exp" type="number" min="0" max="300" placeholder="50">
      </div>
      <div>
        <div class="ag-dir-label">Operator note (sent verbatim in LLM prompt)</div>
        <input class="ag-dir-input" id="ag-in-notes" placeholder="e.g. NFP day — be conservative">
      </div>
      <button class="ag-dir-apply" onclick="_submitDir()">APPLY DIRECTIVES</button>
      <div class="ag-dir-msg" id="ag-dir-msg"></div>
    </div>
  </div>

  <div class="ag-card">
    <div class="ag-card-hdr">
      <span class="ag-card-title"><span class="ag-card-icon">📊</span> Performance KPIs</span>
      <span class="ag-card-sub">30-day history</span>
    </div>
    <div class="ag-kpi-grid" id="ag-kpis"></div>
  </div>
</div>

<!-- ═══ ROW 6: Trade Lifecycle ═══ -->
<div class="ag-card">
  <div class="ag-card-hdr">
    <span class="ag-card-title"><span class="ag-card-icon">🔄</span> Trade Lifecycle View</span>
    <span class="ag-card-sub">Entry → Bot Verdicts → Orchestrator → Execution → Exit</span>
  </div>
  <div class="ag-lc-wrap" id="ag-lifecycle"></div>
</div>
`;
}
