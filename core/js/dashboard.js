// ─────────────────────────────────────────────────────────────
// Fortis AI Pro — Dashboard JavaScript
// ─────────────────────────────────────────────────────────────

// ── State ────────────────────────────────────────────────────
let __userHasChosenTab = false;
let __autoTabSwitchInProgress = false;
let __autoAiLogShown = false;
let __botWasRunning = false;
let lastThoughtTs = null;
const THOUGHTS_BUFFER_MAX = 200;
let thoughtsBuffer = [];
let _zlActivePair = 'ALL';
let equityData = [];

const marketScanCache = {};
let lastMarketScanFetchMs = 0;
const MARKET_SCAN_MIN_INTERVAL_MS = 15000;
// Keyed by symbol — holds the most recent open position for each pair (null if none)
const openPositionsBySymbol = {};

window.addEventListener('load', () => {
    autoCheckMt5();
    initSymbolToggle();
    setupTabClickTracking();
    fetchMarketScanBreakdown();
    updateRR();
});

function setupTabClickTracking() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!__autoTabSwitchInProgress) __userHasChosenTab = true;
        }, true);
    });
}

function showTab(tab, btn) {
    document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    const panel = document.getElementById('tab-' + tab);
    if (panel) { panel.classList.add('active'); } else { console.warn('showTab: no panel for "' + tab + '"'); }
    let targetBtn = btn || document.querySelector('.tab-btn[data-tab="' + tab + '"]');
    if (targetBtn) targetBtn.classList.add('active');
    try {
        if (tab === 'signals') fetchMarketScanBreakdown();
        if (tab === 'thoughts') { lastThoughtTs = null; thoughtsBuffer = []; fetchThoughtsNow(); }
        if (tab === 'positions') fetchPositions();
        if (tab === 'history') fetchHistory();
        if (tab === 'performance') fetchPerformance();
        if (tab === 'snapshot') fetchBotSnapshot();
        if (tab === 'backtest') _btLoadMemory();
    } catch (err) { console.error('showTab data load failed for "' + tab + '":', err); }
}

function fetchThoughtsNow() {
    const url = lastThoughtTs ? '/bot/ai_thoughts?limit=60&since=' + encodeURIComponent(lastThoughtTs) : '/bot/ai_thoughts?limit=60';
    return fetch(url).then(r => r.json()).then(data => {
        if (!data || !data.ok) return;
        const thoughts = data.thoughts || [];
        if (thoughts.length > 0) lastThoughtTs = thoughts[thoughts.length - 1].ts;
        displayThoughts(thoughts);
    }).catch(err => console.error('fetchThoughtsNow error:', err));
}

function autoShowAiLogTab() {
    if (__userHasChosenTab || __autoAiLogShown) return;
    const aiLogTab = document.querySelector('.tab-btn[data-tab="thoughts"]') || document.querySelector('.tab-btn:nth-child(2)');
    if (!aiLogTab || aiLogTab.classList.contains('active')) { __autoAiLogShown = true; return; }
    __autoTabSwitchInProgress = true;
    try { showTab('thoughts', aiLogTab); } finally { __autoTabSwitchInProgress = false; __autoAiLogShown = true; }
}

function initSymbolToggle() {
    document.querySelectorAll('.watch-card').forEach(card => {
        card.classList.add('active'); card.classList.remove('inactive');
        card.addEventListener('click', () => toggleSymbol(card.getAttribute('data-pair'), card));
    });
}

function toggleSymbol(pair, cardElement) {
    const nowActive = !cardElement.classList.contains('active');
    cardElement.classList.toggle('active', nowActive);
    cardElement.classList.toggle('inactive', !nowActive);
    fetch('/bot/config/symbols', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol: pair, enabled: nowActive }) })
        .then(r => r.json()).then(data => {
            if (!data.success) { cardElement.classList.toggle('active', !nowActive); cardElement.classList.toggle('inactive', nowActive); console.error('Symbol toggle failed:', data.message); }
        }).catch(err => { cardElement.classList.toggle('active', !nowActive); cardElement.classList.toggle('inactive', nowActive); console.error('Symbol toggle error:', err); });
}

function autoCheckMt5() { pollMt5Status(); }

function pollMt5Status() {
    fetch('/api/mt5/status').then(r => r.json()).then(data => { updateMt5Display(data); setConnectButtonState(!!data.connected); }).catch(() => {});
}

function setConnectButtonState(connected) {
    const btn = document.getElementById('connect-btn');
    if (!btn) return;
    if (connected) { btn.style.display = 'none'; btn.textContent = 'MT5 \u2713'; const overlay = document.getElementById('mt5-modal-overlay'); if (overlay) overlay.style.display = 'none'; }
    else { btn.style.display = 'inline-block'; btn.textContent = 'Connect MT5'; }
}

async function connectMt5() {
    const headerBtn = document.getElementById('connect-btn');
    const overlay = document.getElementById('mt5-modal-overlay');
    const statusEl = document.getElementById('mt5-modal-status');
    const modalBtn = document.getElementById('mt5-modal-btn');
    if (headerBtn && headerBtn.style.display === 'none') return;
    if (overlay) overlay.style.display = 'flex';
    if (statusEl) { statusEl.style.display = 'block'; statusEl.style.background = 'rgba(33,150,243,0.12)'; statusEl.style.border = '1px solid rgba(33,150,243,0.25)'; statusEl.style.color = 'var(--txt)'; statusEl.textContent = 'Connecting to MT5\u2026 (make sure MT5 is open + logged in)'; }
    const path = (document.getElementById('mt5-path') || {}).value || '';
    try {
        if (modalBtn) { modalBtn.disabled = true; modalBtn.textContent = 'Connecting\u2026'; }
        if (headerBtn) { headerBtn.disabled = true; headerBtn.textContent = 'Connecting\u2026'; }
        const r = await fetch('/api/mt5/connect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: path.trim() || null }) });
        const data = await r.json().catch(() => ({}));
        if (!r.ok || !data.connected) {
            let msg = (data && data.error) ? data.error : 'Failed to connect to MT5';
            if (r.status === 404) msg = 'Backend /api/mt5/connect not found. Start core/server.py.';
            if (statusEl) { statusEl.style.background = 'rgba(244,67,54,0.12)'; statusEl.style.border = '1px solid rgba(244,67,54,0.25)'; statusEl.style.color = 'var(--red)'; statusEl.textContent = msg; }
            setConnectButtonState(false); return;
        }
        updateMt5Display(data); setConnectButtonState(true);
    } catch (e) {
        let msg = (e && e.message) ? e.message : String(e);
        if (/Failed to fetch|NetworkError|REFUSED|refused/i.test(msg)) msg = 'Backend not reachable. Start core/server.py and open http://127.0.0.1:5000';
        if (statusEl) { statusEl.style.background = 'rgba(244,67,54,0.12)'; statusEl.style.border = '1px solid rgba(244,67,54,0.25)'; statusEl.style.color = 'var(--red)'; statusEl.textContent = 'Connect error: ' + msg; }
        setConnectButtonState(false);
    } finally {
        if (modalBtn) { modalBtn.disabled = false; modalBtn.textContent = 'Auto Connect'; }
        if (headerBtn) { headerBtn.disabled = false; if (headerBtn.style.display !== 'none') headerBtn.textContent = 'Connect MT5'; }
    }
}

function openMt5Modal() { document.getElementById('mt5-modal-overlay').style.display = 'flex'; }
function closeMt5Modal() { document.getElementById('mt5-modal-overlay').style.display = 'none'; }

function updateMt5Display(data) {
    const dot = document.getElementById('mt5-dot');
    const statusTxt = document.getElementById('mt5-status-txt');
    const detail = document.getElementById('mt5-detail');
    if (!dot || !statusTxt || !detail) return;
    if (!data.connected) { dot.className = 'mt5-dot err'; statusTxt.textContent = data.error || 'Disconnected'; statusTxt.style.color = 'var(--red)'; detail.style.display = 'none'; return; }
    dot.className = 'mt5-dot ok'; statusTxt.textContent = 'Connected'; statusTxt.style.color = 'var(--green)'; detail.style.display = 'flex';
    _setText('m-server', data.server || '\u2014');
    _setText('m-login', (data.login || '\u2014') + (data.account_name ? ' \u00b7 ' + data.account_name : ''));
    const trEl = document.getElementById('m-trade');
    if (trEl) { trEl.textContent = data.trade_allowed ? 'Allowed' : 'Disabled'; trEl.style.color = data.trade_allowed ? 'var(--green)' : 'var(--red)'; }
    const equityEl = document.getElementById('m-equity');
    if (equityEl && data.equity !== undefined) equityEl.textContent = '$' + data.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const balanceEl = document.getElementById('m-balance');
    if (balanceEl && data.balance !== undefined) balanceEl.textContent = '$' + data.balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function readBotConfigPayload() {
    return {
        symbols: (document.getElementById('cfg-symbols') || {}).value,
        volume: parseFloat((document.getElementById('cfg-volume') || { value: 0.5 }).value),
        poll_interval: parseInt((document.getElementById('cfg-poll') || { value: 300 }).value, 10),
        dry_run: !!((document.getElementById('cfg-dry') || {}).checked),
        ai_review: !!((document.getElementById('cfg-ai') || { checked: true }).checked),
        auto_trade: !!((document.getElementById('cfg-auto') || { checked: true }).checked),
        sl_mult: parseFloat((document.getElementById('cfg-sl-mult') || { value: 2.5 }).value),
        tp_mult: parseFloat((document.getElementById('cfg-tp-mult') || { value: 4.5 }).value),
        atr_mult: parseFloat((document.getElementById('cfg-atr-mult') || { value: 1.5 }).value),
        pc_rr: parseFloat((document.getElementById('cfg-pc-rr') || { value: 1.0 }).value),
        be_buffer: parseFloat((document.getElementById('cfg-be-buf') || { value: 1.0 }).value),
    };
}

async function startBot() {
    const btn = document.getElementById('start-btn'); btn.disabled = true; btn.textContent = 'Starting\u2026';
    try {
        const r = await fetch('/bot/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(readBotConfigPayload()) });
        const data = await r.json();
        if (r.status === 409) { _setBotRunning(true); btn.disabled = false; return; }
        if (!r.ok || data.error) { alert('Error: ' + (data.error || 'Failed to start bot')); btn.disabled = false; btn.textContent = 'Start Bot'; return; }
        if (data.running) _setBotRunning(true);
    } catch (e) { alert('Failed to start bot: ' + e.message); }
    btn.disabled = false;
}

async function stopBot() {
    const btn = document.getElementById('stop-btn'); btn.disabled = true; btn.textContent = 'Stopping\u2026';
    try {
        const r = await fetch('/bot/stop', { method: 'POST' }); const data = await r.json();
        if (!r.ok && r.status !== 409) { alert('Error: ' + (data.error || 'Failed to stop bot')); btn.disabled = false; btn.textContent = 'Stop'; return; }
        _setBotRunning(false);
    } catch (e) { alert('Failed to stop bot: ' + e.message); }
    btn.disabled = false;
}

function _setBotRunning(running) {
    _setText('status-text', running ? 'live' : 'idle');
    const dot = document.getElementById('dot'); if (dot) dot.className = running ? 'dot live' : 'dot';
    const startBtn = document.getElementById('start-btn'); const stopBtn = document.getElementById('stop-btn');
    if (startBtn) startBtn.style.display = running ? 'none' : 'inline-block';
    if (stopBtn) stopBtn.style.display = running ? 'inline-block' : 'none';
    if (startBtn) startBtn.disabled = false;
}

function placeTestTrade() {
    fetch('/api/mt5/test-trade', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
        .then(r => r.json()).then(data => { if (data.ok) console.log('\u2713 Test trade placed:', data.ticket, '@', data.price); else console.warn('Test trade failed:', data.error); })
        .catch(err => console.error('Test trade error:', err));
}

function toggleSym(el) { el.classList.toggle('active'); _syncSymbolsInput(); }
function _syncSymbolsInput() { const active = Array.from(document.querySelectorAll('.sym-chip.active')).map(e => e.textContent).join(','); const el = document.getElementById('cfg-symbols'); if (el) el.value = active; }
function stepInput(id, delta, min) { const el = document.getElementById(id); if (!el) return; el.value = Math.max(min, parseFloat(el.value) + delta).toFixed(2); }

function setPoll(val, el) {
    document.querySelectorAll('.poll-preset').forEach(e => e.classList.remove('active'));
    if (el) el.classList.add('active');
    const hidEl = document.getElementById('cfg-poll'); if (hidEl) hidEl.value = val;
    const disp = document.getElementById('poll-display'); if (disp) disp.textContent = val >= 60 ? Math.round(val / 60) + 'm' : val + 's';
}

function updateRR() {
    const atr = parseFloat((document.getElementById('cfg-atr-mult') || { value: 1.5 }).value);
    const sl = parseFloat((document.getElementById('cfg-sl-mult') || { value: 2.5 }).value);
    const tp = parseFloat((document.getElementById('cfg-tp-mult') || { value: 4.5 }).value);
    const ratio = sl > 0 ? (tp / sl) : 0;
    _setText('rr-label', '1 : ' + ratio.toFixed(1));
    const tpBar = document.getElementById('rr-tp-bar'); if (tpBar) tpBar.style.width = Math.min(ratio * 33, 100) + '%';
}

async function applyConfig() {
    const applyBtn = document.getElementById('apply-btn');
    if (applyBtn) { applyBtn.disabled = true; applyBtn.textContent = 'Applying\u2026'; }
    try {
        const r = await fetch('/bot/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(readBotConfigPayload()) });
        const data = await r.json();
        if (data.ok) { console.log('Config applied:', data.applied); if (applyBtn) { applyBtn.textContent = 'Applied \u2713'; setTimeout(() => { applyBtn.textContent = 'Apply'; applyBtn.disabled = false; }, 1500); } }
        else { alert('Config error: ' + (data.error || 'unknown')); if (applyBtn) { applyBtn.disabled = false; applyBtn.textContent = 'Apply'; } }
    } catch (e) { alert('Config apply failed: ' + e.message); if (applyBtn) { applyBtn.disabled = false; applyBtn.textContent = 'Apply'; } }
}

async function clearThoughts() {
    try { await fetch('/bot/thoughts/clear', { method: 'POST' }); } catch (_) {}
    thoughtsBuffer = []; lastThoughtTs = null;
    displayThoughts([]);
}

function switchZLPair(pair) {
    _zlActivePair = pair;
    document.querySelectorAll('.zl-pair-tab').forEach(b => b.classList.remove('active'));
    const tab = document.getElementById('zl-tab-' + pair);
    if (tab) tab.classList.add('active');
    displayThoughts([]);
}

// \u2500\u2500 Zero Log action colour/label lookup \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
const _ZL_ACTION = {
    close:          { col: '#d63031', arrow: '\u2193', label: 'CLOSE' },
    override_close: { col: '#d63031', arrow: '\u21d3', label: 'OVERRIDE CLOSE' },
    override:       { col: '#d63031', arrow: '\u21d3', label: 'OVERRIDE' },
    move_sl:        { col: '#00b4cc', arrow: '\u21d1', label: 'MOVE SL' },
    hold:           { col: '#4a5568', arrow: '\u2013', label: 'HOLD' },
    approve:        { col: '#2fa538', arrow: '\u2713', label: 'APPROVE' },
    veto:           { col: '#e67e22', arrow: '\u2717', label: 'VETO' },
    monitor:        { col: '#4a5568', arrow: '\u25cf', label: 'MONITORING' },
};
function _zlAction(action) {
    return _ZL_ACTION[(action || '').toLowerCase()] || { col: '#6a7280', arrow: '\u25c6', label: (action || 'INFO').toUpperCase() };
}

function _zlSource(source) {
    const s = String(source || '').toLowerCase();
    if (s === 'execution' || s.includes('order') || s.includes('market'))
        return { label: 'MARKET', icon: '\ud83d\udcc8', col: '#50d963', bg: 'rgba(80,217,99,0.15)' };
    if (s === 'ai_entry' || s === 'ai_risk' || s.includes('agent') || s.includes('ollama') || s.includes('orchestrat'))
        return { label: 'AGENT 0', icon: '\ud83e\udde0', col: '#c792ea', bg: 'rgba(199,146,234,0.15)' };
    if (s.includes('signal') || s.includes('strategy') || s.includes('ai_pro') || s.includes('bot'))
        return { label: 'STRATEGY', icon: '\ud83d\udcca', col: '#5ac8fa', bg: 'rgba(90,200,250,0.15)' };
    return { label: 'INFO', icon: '\u2139', col: '#6a7280', bg: 'rgba(106,114,128,0.15)' };
}

function _zlRelTime(ts, now) {
    const diff = (now - new Date(ts).getTime()) / 1000;
    if (diff < 5)    return 'just now';
    if (diff < 60)   return Math.round(diff) + 's ago';
    if (diff < 3600) return Math.round(diff / 60) + 'm ago';
    return new Date(ts).toLocaleTimeString();
}

function _buildActionCard(t, now, showSymbol) {
    const act     = _zlAction(t.action);
    const src     = _zlSource(t.source);
    const confPct = t.confidence != null ? Math.round(t.confidence * 100) : null;
    const confCol = confPct != null
        ? (confPct >= 70 ? 'var(--green)' : confPct >= 50 ? 'var(--amber)' : 'var(--red)') : '';

    const symBadge = showSymbol && t.symbol && t.symbol !== '*'
        ? `<span style="font-size:9px;font-weight:700;color:var(--txt);letter-spacing:.05em;margin-right:4px">${_esc(t.symbol)}</span>` : '';

    const stageBadge = t.stage
        ? `<span style="font-size:8px;color:var(--txt3);background:var(--bg0);padding:1px 6px;border-radius:2px;letter-spacing:.04em">${_esc(t.stage)}</span>` : '';

    const confBar = confPct != null ? `
        <div style="display:flex;align-items:center;gap:8px;margin:8px 0 2px">
            <div style="flex:1;height:3px;background:var(--bg0);border-radius:2px;overflow:hidden">
                <div style="height:100%;width:${confPct}%;background:${confCol};border-radius:2px"></div>
            </div>
            <span style="font-size:9px;color:${confCol};font-family:var(--mono);white-space:nowrap">${confPct}%</span>
        </div>` : '';

    return `<div style="background:var(--bg1);border-left:3px solid ${act.col};padding:13px 15px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px">
            <div style="display:flex;align-items:center;gap:6px">
                <span style="font-size:12px;font-weight:700;color:${act.col};letter-spacing:.03em">${act.arrow}&nbsp;${act.label}</span>
                ${symBadge}${stageBadge}
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <span style="font-size:9px;font-weight:600;background:${src.bg};color:${src.col};padding:2px 6px;border-radius:3px">${src.icon}&nbsp;${src.label}</span>
                <span style="font-size:8px;color:var(--txt3)" title="${new Date(t.ts).toLocaleString()}">${_zlRelTime(t.ts, now)}</span>
            </div>
        </div>
        ${confBar}
        <div style="font-size:11px;color:var(--txt);line-height:1.45;margin-top:6px">${_esc(t.summary || '')}</div>
        ${t.detail ? `<div style="font-size:9px;color:var(--txt3);margin-top:4px;line-height:1.4;border-top:1px solid var(--line);padding-top:4px">${_esc(t.detail)}</div>` : ''}
    </div>`;
}

function displayThoughts(newThoughts) {
    const container = document.getElementById('thoughts');
    const countEl   = document.getElementById('thought-count');
    if (!container) return;

    if (Array.isArray(newThoughts) && newThoughts.length > 0) {
        const seen = new Set(thoughtsBuffer.map(t => (t.ts || '') + '|' + (t.summary || '')));
        for (const t of newThoughts) {
            const key = (t.ts || '') + '|' + (t.summary || '');
            if (!seen.has(key)) { thoughtsBuffer.push(t); seen.add(key); }
        }
        if (thoughtsBuffer.length > THOUGHTS_BUFFER_MAX)
            thoughtsBuffer = thoughtsBuffer.slice(-THOUGHTS_BUFFER_MAX);
    }

    // Filter by active pair tab
    const filtered = _zlActivePair === 'ALL'
        ? thoughtsBuffer
        : thoughtsBuffer.filter(t => (t.symbol || '').toUpperCase() === _zlActivePair);

    // Update counts on each sub-tab button
    const _pairLabel = { EURUSD:'EUR/USD', GBPUSD:'GBP/USD', GBPJPY:'GBP/JPY', EURJPY:'EUR/JPY' };
    ['EURUSD','GBPUSD','GBPJPY','EURJPY'].forEach(p => {
        const btn = document.getElementById('zl-tab-' + p);
        if (!btn) return;
        const n = thoughtsBuffer.filter(t => (t.symbol || '').toUpperCase() === p).length;
        btn.textContent = _pairLabel[p] + (n ? ` (${n})` : '');
    });
    if (countEl) countEl.textContent = filtered.length + (filtered.length !== thoughtsBuffer.length ? ` / ${thoughtsBuffer.length} total` : '') + ' entries';

    if (filtered.length === 0) {
        const msg = thoughtsBuffer.length === 0
            ? 'No actions yet \u2014 start the bot to see live decisions.'
            : `No actions for ${_zlActivePair} yet.`;
        container.innerHTML = `<div style="padding:50px;text-align:center;color:var(--txt3)">
            <div style="font-size:22px;opacity:.3;margin-bottom:10px">\u25cb</div>
            <div style="font-size:11px">${msg}</div>
        </div>`;
        return;
    }

    const now = Date.now();
    const atTop = container.scrollTop < 40;
    const showSym = _zlActivePair === 'ALL';
    container.innerHTML = [...filtered].reverse().map(t => _buildActionCard(t, now, showSym)).join('');
    if (atTop) container.scrollTop = 0;
}

function fetchMarketScanBreakdown() {
    const now = Date.now(); if (now - lastMarketScanFetchMs < MARKET_SCAN_MIN_INTERVAL_MS) return; lastMarketScanFetchMs = now;
    const symbols = ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY']; const container = document.getElementById('signal-cards'); if (!container) return;
    symbols.forEach(symbol => {
        fetch('/bot/signal/' + symbol).then(r => r.json()).then(data => { marketScanCache[symbol] = data; renderMarketScanBreakdown(); updatePairWatch(symbol, data); })
            .catch(err => { marketScanCache[symbol] = { error: 'Fetch failed' }; console.error('Signal fetch error ' + symbol + ':', err); });
    });
}

function updatePairWatch(symbol, data) {
    if (!data || data.error) return;
    const statusEl = document.getElementById('status-' + symbol); if (!statusEl) return;
    const bias = data.bias || '\u2014';
    const dotEl = statusEl.querySelector('.watch-dot'); const stateEl = statusEl.querySelector('.watch-state');
    if (dotEl) dotEl.className = 'watch-dot ' + (bias === 'LONG' ? 'buy' : bias === 'SHORT' ? 'sell' : 'idle');
    if (stateEl) stateEl.textContent = bias === 'LONG' ? 'bullish' : bias === 'SHORT' ? 'bearish' : 'neutral';
}

// Parse ENV1\u20134 statuses from the environment string.
// Returns array of 4 values: 'on' | 'off' | 'lowconf' | 'unknown'
function _parseEnvStatuses(envStr) {
    if (!envStr || /no active environment/i.test(envStr)) return ['off','off','off','off'];
    const isLowConf = /low confidence/i.test(envStr);
    return [1,2,3,4].map(n => {
        const m = new RegExp('ENV\\s*' + n + '\\s*(on|off)', 'i').exec(envStr);
        if (m) return m[1].toLowerCase() === 'on' ? (isLowConf ? 'lowconf' : 'on') : 'off';
        // Not explicitly listed \u2014 if string contains at least one ENV reference, treat as unknown
        return envStr.includes('ENV') ? 'unknown' : (isLowConf ? 'lowconf' : 'off');
    });
}

function _buildEnvPills(envStr) {
    const statuses = _parseEnvStatuses(envStr);
    const isLowConf = envStr && /low confidence/i.test(envStr);
    const labels = ['1','2','3','4'];
    let html = '<span style="font-size:8px;color:var(--txt3);margin-right:3px;text-transform:uppercase;letter-spacing:.08em">ENV</span>';
    statuses.forEach((s, i) => {
        let col, bg, border;
        if (s === 'on')      { col = '#0a0e14'; bg = 'var(--green)';  border = 'transparent'; }
        else if (s === 'lowconf') { col = '#0a0e14'; bg = 'var(--amber)'; border = 'transparent'; }
        else                 { col = 'var(--txt3)'; bg = 'var(--bg2)'; border = 'var(--line)'; }
        html += `<span style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:18px;font-size:8px;font-weight:700;border-radius:3px;color:${col};background:${bg};border:1px solid ${border}">${labels[i]}</span>`;
    });
    if (isLowConf) {
        const m = /low confidence\s*\((\d+)%\s*[<>]\s*(\d+)%\)/i.exec(envStr || '');
        html += `<span style="font-size:8px;color:var(--amber);margin-left:3px">${m ? m[1]+'% < '+m[2]+'%' : 'low conf'}</span>`;
    }
    return html;
}

function _buildConfBar(conf) {
    const col = conf >= 75 ? '#50d963' : conf >= 50 ? '#ffc107' : '#ff6b6b';
    const label = conf >= 75 ? 'STRONG' : conf >= 55 ? 'MODERATE' : 'WEAK';
    const labelCol = conf >= 75 ? 'var(--green)' : conf >= 55 ? 'var(--amber)' : 'var(--red)';
    return `<div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span style="font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.08em">Confidence</span>
            <span style="font-size:9px;font-weight:700;color:${labelCol}">${conf}% &nbsp;${label}</span>
        </div>
        <div style="height:4px;background:var(--bg3);border-radius:2px;overflow:hidden">
            <div style="height:100%;width:${Math.min(conf,100)}%;background:${col};border-radius:2px"></div>
        </div>
    </div>`;
}

function _buildActivePositionBlock(pos, pipSize, digits) {
    const isBuy   = pos.type === 'BUY';
    const pnl     = Number(pos.profit || 0);
    const pnlCol  = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlSign = pnl >= 0 ? '+' : '';
    const tradeCol = isBuy ? 'var(--green)' : 'var(--red)';
    const tradeBg  = isBuy ? 'rgba(80,217,99,0.08)' : 'rgba(255,107,107,0.08)';
    const tradeBorder = isBuy ? 'rgba(80,217,99,0.3)' : 'rgba(255,107,107,0.3)';

    const open    = Number(pos.open_price    || pos.entry || 0);
    const current = Number(pos.current_price || 0);
    let movePips = 0;
    if (open && current && pipSize) {
        movePips = Math.round((isBuy ? current - open : open - current) / pipSize);
    }
    const moveCol = movePips >= 0 ? 'var(--green)' : 'var(--red)';

    const openedAge = pos.open_time
        ? (() => {
              // open_time is ISO string from server.py or Unix int from bot internal
              const d = typeof pos.open_time === 'number'
                  ? new Date(pos.open_time * 1000)
                  : new Date(pos.open_time);
              const s = Math.round((Date.now() - d.getTime()) / 1000);
              return s < 3600 ? Math.round(s / 60) + 'm' : Math.round(s / 3600) + 'h';
          })()
        : null;

    return `<div style="background:${tradeBg};border:1px solid ${tradeBorder};border-radius:6px;padding:10px;margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div style="display:flex;align-items:center;gap:6px">
                <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${tradeCol};animation:pulse-watch 1.5s infinite"></span>
                <span style="font-size:9px;font-weight:700;color:${tradeCol};text-transform:uppercase;letter-spacing:.08em">In Trade</span>
                <span style="font-size:9px;font-weight:700;color:${tradeCol}">${pos.type}</span>
                <span style="font-size:8px;color:var(--txt3)">${pos.volume} lot</span>
            </div>
            <div style="text-align:right">
                <div style="font-size:13px;font-weight:700;color:${pnlCol};font-family:var(--mono)">${pnlSign}$${Math.abs(pnl).toFixed(2)}</div>
                ${openedAge ? `<div style="font-size:8px;color:var(--txt3)">${openedAge} open</div>` : ''}
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:6px">
            <div style="background:var(--bg0);border-radius:4px;padding:5px 7px">
                <div style="font-size:8px;color:var(--txt3);margin-bottom:2px">Opened</div>
                <div style="font-family:var(--mono);font-size:10px;color:var(--txt)">${open ? open.toFixed(digits) : '\u2014'}</div>
            </div>
            <div style="background:var(--bg0);border-radius:4px;padding:5px 7px">
                <div style="font-size:8px;color:var(--txt3);margin-bottom:2px">Current</div>
                <div style="font-family:var(--mono);font-size:10px;color:var(--txt)">${current ? current.toFixed(digits) : '\u2014'}</div>
            </div>
            <div style="background:var(--bg0);border-radius:4px;padding:5px 7px">
                <div style="font-size:8px;color:var(--txt3);margin-bottom:2px">Move</div>
                <div style="font-family:var(--mono);font-size:10px;font-weight:700;color:${moveCol}">${movePips >= 0 ? '+' : ''}${movePips}p</div>
            </div>
        </div>
        ${(pos.sl || pos.tp) ? `<div style="display:flex;gap:10px;font-size:9px;color:var(--txt3)">
            ${pos.sl ? `<span>SL <span style="font-family:var(--mono);color:var(--red)">${Number(pos.sl).toFixed(digits)}</span></span>` : ''}
            ${pos.tp ? `<span>TP <span style="font-family:var(--mono);color:var(--green)">${Number(pos.tp).toFixed(digits)}</span></span>` : ''}
            <span style="margin-left:auto;color:var(--txt3)">#${pos.ticket}</span>
        </div>` : ''}
    </div>`;
}

function buildSignalCard(symbol, sig) {
    if (!sig) return `<div style="background:var(--bg1);padding:16px;color:var(--txt3);font-size:10px">Loading ${_esc(symbol)}\u2026</div>`;
    if (sig.error) return `<div style="background:var(--bg1);border-left:3px solid var(--red);padding:16px"><div style="font-size:13px;font-weight:700;color:var(--txt);margin-bottom:6px">${_esc(symbol)}</div><div style="font-size:10px;color:var(--red)">${_esc(sig.error)}</div></div>`;

    const pos    = openPositionsBySymbol[symbol] || null;
    const bias   = sig.bias || 'neutral';
    const conf   = sig.confidence || 0;
    const isLong  = bias === 'LONG';
    const isShort = bias === 'SHORT';
    const isActive = isLong || isShort;

    // Border priority: active trade > signal bias
    let borderColor, biasColor, arrow, biasLabel;
    if (pos) {
        borderColor = pos.type === 'BUY' ? '#2fa538' : '#d63031';
        biasColor   = pos.type === 'BUY' ? 'var(--green)' : 'var(--red)';
    } else {
        borderColor = isLong ? '#2fa538' : isShort ? '#d63031' : 'var(--line)';
        biasColor   = isLong ? 'var(--green)' : isShort ? 'var(--red)' : 'var(--txt3)';
    }
    arrow     = isLong ? '\u2191' : isShort ? '\u2193' : '\u2014';
    biasLabel = isLong ? 'LONG'  : isShort ? 'SHORT'  : 'NO SIGNAL';

    const pipSize = symbol.includes('JPY') ? 0.01 : 0.0001;
    const digits  = symbol.includes('JPY') ? 3 : 5;
    const entry = sig.entry_price, sl = sig.sl_price, tp = sig.tp_price;
    let slPips = 0, tpPips = 0, rr = 0;
    if (entry && sl) slPips = Math.round(Math.abs(entry - sl) / pipSize);
    if (entry && tp) tpPips = Math.round(Math.abs(tp - entry) / pipSize);
    if (slPips > 0 && tpPips > 0) rr = tpPips / slPips;

    let timeAgo = 'scanning\u2026';
    if (sig.ts) {
        const age = Math.round((Date.now() - new Date(sig.ts).getTime()) / 1000);
        timeAgo = age < 60 ? age + 's ago' : Math.round(age / 60) + 'm ago';
    }

    const choch  = sig.choch_status && sig.choch_status !== '\u2014' ? sig.choch_status : null;
    const levels = sig.level_interaction && sig.level_interaction !== '\u2014' ? sig.level_interaction : null;
    const rrCol  = rr >= 2 ? 'var(--green)' : rr >= 1.5 ? 'var(--amber)' : 'var(--red)';

    const signalBlock = isActive ? `
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:7px">
            <div style="background:var(--bg0);border-radius:5px;padding:7px 8px">
                <div style="font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">Entry</div>
                <div style="font-family:var(--mono);font-size:11px;color:var(--txt);font-weight:600">${entry ? entry.toFixed(digits) : '\u2014'}</div>
            </div>
            <div style="background:rgba(255,107,107,0.07);border-top:2px solid #d63031;border-radius:5px;padding:7px 8px">
                <div style="font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">Stop Loss</div>
                <div style="font-family:var(--mono);font-size:11px;color:var(--red);font-weight:600">${sl ? sl.toFixed(digits) : '\u2014'}</div>
                ${slPips > 0 ? `<div style="font-size:8px;color:var(--red);opacity:.6;margin-top:2px">\u2212${slPips}p</div>` : ''}
            </div>
            <div style="background:rgba(80,217,99,0.07);border-top:2px solid #2fa538;border-radius:5px;padding:7px 8px">
                <div style="font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">Take Profit</div>
                <div style="font-family:var(--mono);font-size:11px;color:var(--green);font-weight:600">${tp ? tp.toFixed(digits) : '\u2014'}</div>
                ${tpPips > 0 ? `<div style="font-size:8px;color:var(--green);opacity:.6;margin-top:2px">+${tpPips}p</div>` : ''}
            </div>
        </div>
        ${rr > 0 ? `<div style="display:flex;align-items:center;justify-content:space-between;background:var(--bg0);border-radius:5px;padding:6px 10px;margin-bottom:8px">
            <span style="font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em">Risk / Reward</span>
            <span style="font-family:var(--mono);font-weight:700;font-size:13px;color:${rrCol}">${rr.toFixed(1)}R</span>
            <span style="font-size:8px;color:var(--txt3)">${slPips}p \u2192 ${tpPips}p</span>
        </div>` : ''}` :
        (!pos ? `<div style="display:flex;align-items:center;justify-content:center;gap:8px;padding:18px 0;margin-bottom:8px;opacity:.5">
            <span style="font-size:20px;color:var(--txt3)">&#9711;</span>
            <span style="font-size:9px;color:var(--txt3);letter-spacing:.1em;text-transform:uppercase">Awaiting setup</span>
        </div>` : '');

    return `<div style="background:var(--bg1);border-left:3px solid ${borderColor};padding:14px 15px;display:flex;flex-direction:column">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
            <div style="display:flex;align-items:baseline;gap:7px">
                <span style="font-size:15px;font-weight:700;color:var(--txt);letter-spacing:.03em">${symbol}</span>
                <span style="font-size:8px;color:var(--txt3);letter-spacing:.1em">M15</span>
            </div>
            <div style="text-align:right">
                <div style="font-size:11px;font-weight:700;color:${biasColor};letter-spacing:.04em">${arrow} ${biasLabel}</div>
                <div style="font-size:8px;color:var(--txt3);margin-top:2px">${timeAgo}</div>
            </div>
        </div>
        ${_buildConfBar(conf)}
        ${pos ? _buildActivePositionBlock(pos, pipSize, digits) : ''}
        ${signalBlock}
        <div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-bottom:6px">
            ${_buildEnvPills(sig.environment)}
        </div>
        ${choch  ? `<div style="font-size:9px;color:var(--cyan);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px" title="${_esc(choch)}">\u21ba ${_esc(choch)}</div>` : ''}
        ${levels ? `<div style="font-size:9px;color:var(--txt3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:6px" title="${_esc(levels)}">\u25c6 ${_esc(levels)}</div>` : ''}
        ${_buildPocRow(sig, pipSize)}
    </div>`;
}

function _buildPocRow(sig, pipSize) {
    const poc = sig.poc;
    if (!poc) return '';

    const aligned   = sig.poc_aligned;   // true | false | null
    const distPips  = sig.poc_dist_pips; // signed float, positive = entry above poc
    const side      = sig.poc_side || '—';
    const tolMult   = sig.atr_tol_mult;

    // Alignment badge
    let alignBadge;
    if (aligned === true)       alignBadge = `<span style="background:rgba(80,217,99,0.15);color:var(--green);padding:2px 6px;border-radius:3px;font-size:9px;font-weight:700;font-family:var(--mono)">ALIGNED ✓</span>`;
    else if (aligned === false) alignBadge = `<span style="background:rgba(255,107,107,0.15);color:var(--red);padding:2px 6px;border-radius:3px;font-size:9px;font-weight:700;font-family:var(--mono)">GATED ✗</span>`;
    else                        alignBadge = `<span style="background:var(--bg2);color:var(--txt3);padding:2px 6px;border-radius:3px;font-size:9px;font-family:var(--mono)">WATCHING</span>`;

    const digits    = (pipSize === 0.01) ? 3 : 5;
    const distStr   = distPips != null ? (distPips >= 0 ? '+' : '') + distPips.toFixed(1) + 'p' : '—';
    const sideArrow = side === 'above' ? '▲' : side === 'below' ? '▼' : '●';
    const sideCol   = side === 'above' ? 'var(--green)' : side === 'below' ? 'var(--red)' : 'var(--txt3)';

    const zoneStr = tolMult != null
        ? `<span style="color:var(--txt3);font-size:9px">Zone <span style="font-family:var(--mono);color:var(--txt2)">${tolMult}×ATR</span> <span style="color:var(--txt3);font-size:8px">(backtest)</span></span>`
        : '';

    return `<div style="border-top:1px solid var(--line);margin-top:2px;padding-top:7px">
        <div style="font-size:8px;color:var(--txt3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">Volume Profile · POC</div>
        <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap">
            <div style="display:flex;align-items:center;gap:5px">
                <span style="color:${sideCol};font-size:10px">${sideArrow}</span>
                <span style="font-family:var(--mono);font-size:11px;color:var(--txt);font-weight:600">${poc.toFixed(digits)}</span>
                <span style="font-size:9px;color:var(--txt3)">${distStr}</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px">
                ${alignBadge}
            </div>
        </div>
        ${zoneStr ? `<div style="margin-top:4px">${zoneStr}</div>` : ''}
    </div>`;
}

function renderMarketScanBreakdown() {
    const container = document.getElementById('signal-cards'); if (!container) return;
    const symbols = ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY'];
    if (!symbols.every(s => marketScanCache[s])) {
        container.innerHTML = '<div class="empty-state"><div class="icon">\u27f3</div><div class="msg">Scanning markets\u2026</div></div>';
        return;
    }

    const longCount  = symbols.filter(s => (marketScanCache[s]||{}).bias === 'LONG').length;
    const shortCount = symbols.filter(s => (marketScanCache[s]||{}).bias === 'SHORT').length;
    const idleCount  = symbols.length - longCount - shortCount;
    const tradeCount = symbols.filter(s => openPositionsBySymbol[s]).length;

    const summaryBar = `<div style="display:flex;align-items:center;gap:14px;padding:9px 16px;background:var(--bg1);border-bottom:1px solid var(--line);font-size:9px;text-transform:uppercase;letter-spacing:.08em;flex-shrink:0">
        <span style="color:var(--txt3)">Signal Board &nbsp;&#xb7;&nbsp; M15</span>
        <span style="color:var(--green)">&#x2191; ${longCount} Long</span>
        <span style="color:var(--red)">&#x2193; ${shortCount} Short</span>
        <span style="color:var(--txt3)">&#9711; ${idleCount} Idle</span>
        ${tradeCount > 0 ? `<span style="color:var(--amber)">&#9679; ${tradeCount} In Trade</span>` : ''}
        <span style="margin-left:auto;color:var(--txt3)">Auto-refreshes every 5 min</span>
    </div>`;

    const grid = `<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);flex:1">${
        symbols.map(s => buildSignalCard(s, marketScanCache[s])).join('')
    }</div>`;

    container.innerHTML = summaryBar + grid;
}

function fetchPositions() { fetch('/bot/positions').then(r => r.json()).then(data => renderPositions(data.positions || [])).catch(() => renderPositions([])); }

function renderPositions(positions) {
    const container = document.getElementById('positions-content'); if (!container) return;
    if (!positions || positions.length === 0) { container.innerHTML = '<div class="pos-empty">No open positions</div>'; return; }
    let totalProfit = 0;
    let html = '<div style="padding:16px"><div style="display:grid;gap:12px">';
    positions.forEach(pos => {
        totalProfit += pos.profit || 0;
        const typeColor = pos.type === 'BUY' ? 'var(--green)' : 'var(--red)';
        const profitColor = (pos.profit || 0) >= 0 ? 'var(--green)' : 'var(--red)';
        const profitSign = (pos.profit || 0) >= 0 ? '+' : '';
        html += `<div style="background:var(--bg0);border:1px solid var(--line);border-radius:8px;padding:14px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:13px;font-weight:600;color:var(--txt)">${_esc(pos.symbol)}</span>
                    <span style="background:${typeColor};color:#000;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:600">${pos.type}</span>
                    <span style="color:var(--txt2);font-size:10px">Vol: ${pos.volume}</span>
                </div>
                <div style="text-align:right"><div style="font-size:14px;font-weight:600;color:${profitColor}">${profitSign}$${Math.abs(pos.profit||0).toFixed(2)}</div><div style="font-size:10px;color:var(--txt3)">#${pos.ticket}</div></div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:10px">
                <div><div style="color:var(--txt3);margin-bottom:2px">Open Price</div><div style="font-family:var(--mono);color:var(--txt)">${pos.open_price}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Current</div><div style="font-family:var(--mono);color:var(--txt)">${pos.current_price}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Swap</div><div style="font-family:var(--mono);color:var(--txt2)">${pos.swap||0}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Stop Loss</div><div style="font-family:var(--mono);color:var(--red)">${pos.sl||'\u2014'}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Take Profit</div><div style="font-family:var(--mono);color:var(--green)">${pos.tp||'\u2014'}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Opened</div><div style="color:var(--txt2)">${new Date(pos.open_time).toLocaleTimeString()}</div></div>
            </div></div>`;
    });
    const totalColor = totalProfit >= 0 ? 'var(--green)' : 'var(--red)';
    const totalSign = totalProfit >= 0 ? '+' : '';
    html += `</div><div style="margin-top:12px;padding:12px 16px;background:var(--bg1);border:1px solid var(--line);border-radius:6px;display:flex;justify-content:space-between;align-items:center"><span style="font-size:11px;color:var(--txt2)">${positions.length} open position${positions.length!==1?'s':''}</span><span style="font-size:13px;font-weight:600;color:${totalColor}">Floating P&amp;L: ${totalSign}$${Math.abs(totalProfit).toFixed(2)}</span></div></div>`;
    container.innerHTML = html;
}

function fetchHistory() { fetch('/bot/history').then(r => r.json()).then(data => renderHistory(data.trades || [])).catch(() => renderHistory([])); }

function renderHistory(trades) {
    const listEl = document.getElementById('history-list'); const summaryEl = document.getElementById('history-summary');
    if (!listEl) return;
    if (!trades || trades.length === 0) { if (summaryEl) summaryEl.innerHTML = ''; listEl.innerHTML = '<div class="pos-empty">No trades recorded yet</div>'; return; }

    const totalPL = trades.reduce((s, t) => s + (t.profit || 0), 0);
    const totalWins = trades.filter(t => (t.profit || 0) > 0).length;
    const totalWinRate = (totalWins / trades.length * 100).toFixed(1);
    const plColor = totalPL >= 0 ? 'var(--green)' : 'var(--red)';

    if (summaryEl) {
        summaryEl.innerHTML = `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:14px 16px 0">
            <div style="background:var(--bg1);border:1px solid var(--line);border-radius:6px;padding:10px 12px;text-align:center">
                <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Total Trades</div>
                <div style="font-size:20px;font-weight:700;color:var(--txt)">${trades.length}</div>
            </div>
            <div style="background:var(--bg1);border:1px solid var(--line);border-radius:6px;padding:10px 12px;text-align:center">
                <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Win Rate</div>
                <div style="font-size:20px;font-weight:700;color:var(--cyan)">${totalWinRate}%</div>
            </div>
            <div style="background:var(--bg1);border:1px solid var(--line);border-radius:6px;padding:10px 12px;text-align:center">
                <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Net P&amp;L</div>
                <div style="font-size:20px;font-weight:700;color:${plColor}">${totalPL>=0?'+':''}$${totalPL.toFixed(2)}</div>
            </div>
        </div>`;
    }

    const PAIRS = ['EURUSD','GBPUSD','GBPJPY','EURJPY'];
    const LABELS = {'EURUSD':'EUR/USD','GBPUSD':'GBP/USD','GBPJPY':'GBP/JPY','EURJPY':'EUR/JPY'};
    const byPair = {};
    PAIRS.forEach(p => { byPair[p] = []; });
    trades.forEach(t => { const sym = (t.symbol||'').toUpperCase().replace('/',''); if (byPair[sym]) byPair[sym].push(t); });

    function _pairCard(sym) {
        const pts = byPair[sym];
        if (!pts.length) {
            return `<div style="background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.2)">
                <div style="font-size:11px;font-weight:700;color:var(--txt);margin-bottom:10px">${LABELS[sym]}</div>
                <div style="font-size:10px;color:var(--txt3)">No trades</div>
            </div>`;
        }
        const pl = pts.reduce((s, t) => s + (t.profit || 0), 0);
        const wins = pts.filter(t => (t.profit || 0) > 0).length;
        const wr = (wins / pts.length * 100);
        const buys = pts.filter(t => t.type === 'BUY').length;
        const sells = pts.length - buys;
        const plCol = pl >= 0 ? 'var(--green)' : 'var(--red)';
        const wrCol = wr >= 55 ? 'var(--green)' : wr >= 40 ? '#e67e22' : 'var(--red)';
        const recent = pts.slice().sort((a,b) => new Date(b.time) - new Date(a.time)).slice(0, 5);

        let rows = '';
        recent.forEach(t => {
            const pc = (t.profit||0) >= 0 ? 'var(--green)' : 'var(--red)';
            const ps = (t.profit||0) >= 0 ? '+' : '';
            rows += `<div style="display:grid;grid-template-columns:44px 1fr 72px;gap:6px;align-items:center;padding:4px 0;border-top:1px solid var(--line)">
                <span style="background:${t.type==='BUY'?'var(--green)':'var(--red)'};color:#000;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:700;text-align:center">${t.type}</span>
                <span style="font-size:10px;color:var(--txt3)">${new Date(t.time).toLocaleDateString(undefined,{month:'short',day:'numeric'})}</span>
                <span style="font-size:10px;font-family:var(--mono);color:${pc};font-weight:600;text-align:right">${ps}$${Math.abs(t.profit||0).toFixed(2)}</span>
            </div>`;
        });

        return `<div style="background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.2);display:flex;flex-direction:column;gap:10px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:12px;font-weight:700;color:var(--txt)">${LABELS[sym]}</span>
                <span style="font-size:10px;color:var(--txt3)">${pts.length} trade${pts.length!==1?'s':''}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div>
                    <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px">Win Rate</div>
                    <div style="background:var(--bg0);border-radius:3px;height:5px;overflow:hidden;margin-bottom:3px">
                        <div style="height:100%;width:${wr.toFixed(0)}%;background:${wrCol};transition:width .4s"></div>
                    </div>
                    <div style="font-size:13px;font-weight:700;color:${wrCol}">${wr.toFixed(0)}%</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px">Net P&amp;L</div>
                    <div style="font-size:13px;font-weight:700;color:${plCol}">${pl>=0?'+':''}$${pl.toFixed(2)}</div>
                    <div style="font-size:9px;color:var(--txt3);margin-top:2px">${buys}B · ${sells}S</div>
                </div>
            </div>
            <div>${rows}</div>
        </div>`;
    }

    listEl.innerHTML = `<div style="padding:12px 16px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
        ${PAIRS.map(_pairCard).join('')}
    </div>`;
}

function fetchPerformance() {
    fetch('/bot/performance').then(r => r.json()).then(data => {
        renderKPIs(data.kpis || {});
        equityData = (data.equity_curve && data.equity_curve.length) ? data.equity_curve : [];
        renderEquityCurve(equityData, data.error || null);
        renderPairPerformance(data.pair_stats || []);
    }).catch(() => { renderKPIs({}); renderEquityCurve([], 'Failed to reach /bot/performance'); renderPairPerformance([]); });
}

function renderPairPerformance(pairStats) {
    const el = document.getElementById('perf-pair-grid');
    if (!el) return;
    const LABELS = {EURUSD:'EUR/USD', GBPUSD:'GBP/USD', GBPJPY:'GBP/JPY', EURJPY:'EUR/JPY'};
    const PAIRS  = ['EURUSD','GBPUSD','GBPJPY','EURJPY'];

    const bySymbol = {};
    (pairStats || []).forEach(p => { bySymbol[p.symbol] = p; });

    el.innerHTML = PAIRS.map(sym => {
        const p = bySymbol[sym];
        if (!p || !p.trades) {
            return `<div style="background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.2)">
                <div style="font-size:12px;font-weight:700;color:var(--txt);margin-bottom:8px">${LABELS[sym]}</div>
                <div style="font-size:10px;color:var(--txt3)">No trades in last 90d</div>
            </div>`;
        }
        const wr    = p.win_rate || 0;
        const pl    = p.total_pnl || 0;
        const wrCol = wr >= 55 ? 'var(--green)' : wr >= 40 ? '#e67e22' : 'var(--red)';
        const plCol = pl >= 0 ? 'var(--green)' : 'var(--red)';
        const pf    = (p.wins || 0) > 0 && (p.losses || 0) > 0 ? '' : p.wins ? ' · all wins' : ' · all losses';

        return `<div style="background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.2);display:flex;flex-direction:column;gap:10px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:12px;font-weight:700;color:var(--txt)">${LABELS[sym]}</span>
                <span style="font-size:10px;color:var(--txt3)">${p.trades} trade${p.trades!==1?'s':''}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div>
                    <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Win Rate</div>
                    <div style="background:var(--bg3);border-radius:3px;height:5px;overflow:hidden;margin-bottom:4px">
                        <div style="height:100%;width:${wr.toFixed(0)}%;background:${wrCol};transition:width .5s"></div>
                    </div>
                    <div style="font-size:14px;font-weight:700;color:${wrCol}">${wr.toFixed(1)}%</div>
                    <div style="font-size:9px;color:var(--txt3);margin-top:2px">${p.wins}W · ${p.losses}L${pf}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Net P&amp;L</div>
                    <div style="font-size:14px;font-weight:700;color:${plCol}">${pl>=0?'+':''}$${Math.abs(pl).toFixed(2)}</div>
                    <div style="font-size:9px;color:var(--txt3);margin-top:6px">${p.buys||0}B · ${p.sells||0}S</div>
                </div>
            </div>
        </div>`;
    }).join('');
}

function renderKPIs(kpis) {
    _setKPI('kpi-winrate', (kpis.win_rate || 0).toFixed(1) + '%');
    const pfVal = Number(kpis.profit_factor || 0);
    _setKPI('kpi-profitfactor', pfVal >= 999 ? '\u221e' : pfVal.toFixed(2));
    _setKPI('kpi-trades', kpis.total_trades || 0);
    _setKPI('kpi-return', (kpis.equity_return || 0).toFixed(2) + '%');
    _setKPI('kpi-drawdown', (kpis.max_drawdown || 0).toFixed(2) + '%');
    _setKPI('kpi-sharpe', (kpis.sharpe || 0).toFixed(2));
    _setKPI('kpi-avgratio', (kpis.avg_win_loss_ratio || 0).toFixed(2));
    _setKPI('kpi-currentdd', (kpis.current_drawdown || 0).toFixed(2) + '%');
    _setKPI('kpi-recovery', (kpis.recovery_factor || 0).toFixed(2));
    _setKPI('kpi-sortino', (kpis.sortino || 0).toFixed(2));
    // Sub-labels
    const W=kpis.wins_count||0,L=kpis.losses_count||0;
    _setKPI('kpi-winrate-sub',W+'W \u00b7 '+L+'L');
    const gp=Number(kpis.gross_profit||0),gl=Number(kpis.gross_loss||0);
    _setKPI('kpi-profitfactor-sub','+$'+gp.toFixed(2)+' \u00b7 \u2212$'+gl.toFixed(2));
    const openCnt=kpis.open_positions||0,openPnl=Number(kpis.open_pnl||0);
    const openStr=openCnt>0?openCnt+' open ('+(openPnl>=0?'+':'')+'$'+openPnl.toFixed(2)+')':'0 open';
    _setKPI('kpi-trades-sub',(kpis.total_trades||0)+' closed \u00b7 '+openStr);
    const netPnl=Number(kpis.total_pnl||0);
    _setKPI('kpi-return-sub',(netPnl>=0?'+':'')+'$'+netPnl.toFixed(2)+' net');
    const mddAbs=Number(kpis.max_drawdown_abs||0);
    _setKPI('kpi-drawdown-sub','\u2212$'+mddAbs.toFixed(2)+' peak\u2013trough');
    const sv=Number(kpis.sharpe||0);
    _setKPI('kpi-sharpe-sub',sv>=2?'excellent':sv>=1?'good':sv>=0.5?'moderate':sv>0?'weak':'negative');
    const aw=Number(kpis.avg_win_abs||0),al=Number(kpis.avg_loss_abs||0);
    _setKPI('kpi-avgratio-sub','$'+aw.toFixed(2)+' win / $'+al.toFixed(2)+' loss');
    const cddAbs=Number(kpis.current_drawdown_abs||0);
    _setKPI('kpi-currentdd-sub',cddAbs>0?'\u2212$'+cddAbs.toFixed(2)+' from peak':'at peak');
    _setKPI('kpi-recovery-sub','$'+netPnl.toFixed(2)+' net / $'+mddAbs.toFixed(2)+' max DD');
    const stv=Number(kpis.sortino||0);
    _setKPI('kpi-sortino-sub',stv>=2?'excellent':stv>=1?'good':stv>=0.5?'moderate':stv>0?'weak':'negative');
    const db=kpis.direction_breakdown||{};
    ['BUY','SELL'].forEach(s=>{const b=db[s]||{},dc=(b.wins||0)+(b.losses||0);const t=dc>0?(b.wins||0)+'W \u00b7 '+(b.losses||0)+'L \u00b7 '+(b.win_rate||0).toFixed(1)+'% \u00b7 '+(Number(b.total_pnl||0)>=0?'+':'')+'$'+Number(b.total_pnl||0).toFixed(2):'no trades';const el=document.getElementById('kpi-direction-'+s.toLowerCase());if(el)el.textContent=t;});
}

function renderEquityCurve(data, errorMsg) {
    const canvas = document.getElementById('equityCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width || 600; const cssH = rect.height || 300;
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(dpr, dpr);
    const W = cssW; const H = cssH;
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    if (errorMsg || !data || data.length < 1) {
        ctx.fillStyle = errorMsg ? '#ff6b6b' : '#6a7280'; ctx.font = '12px "IBM Plex Mono",monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const msg = errorMsg ? '\u26a0 ' + errorMsg : 'No closed trades in the last 90 days \u2014 equity curve will appear once trades start closing.';
        const lines = _wrapText(msg, 70); const lineH = 16; const startY = H / 2 - ((lines.length - 1) * lineH) / 2;
        lines.forEach((ln, i) => ctx.fillText(ln, W / 2, startY + i * lineH)); return;
    }
    if (data.length < 2) { ctx.fillStyle = '#6a7280'; ctx.font = '12px "IBM Plex Mono",monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('Only 1 closed trade so far (P&L $' + Number(data[0]).toFixed(2) + ').', W / 2, H / 2); return; }
    const series = (Number(data[0]) === 0) ? data.slice() : [0, ...data];
    const pad = { top: 20, right: 20, bottom: 28, left: 56 };
    const plotW = W - pad.left - pad.right; const plotH = H - pad.top - pad.bottom;
    const min = Math.min(...series); const max = Math.max(...series); const range = (max - min) || 1;
    ctx.strokeStyle = 'rgba(42,58,82,0.6)'; ctx.lineWidth = 0.5; ctx.font = '9px "IBM Plex Mono",monospace';
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH * i / 4);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        const val = max - (range * i / 4); ctx.fillStyle = '#6a7280'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText((val >= 0 ? '+$' : '-$') + Math.abs(val).toFixed(0), pad.left - 6, y);
    }
    const lastVal = series[series.length - 1]; const lineCol = lastVal >= 0 ? '#50d963' : '#ff6b6b';
    const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    grad.addColorStop(0, lastVal >= 0 ? 'rgba(80,217,99,0.35)' : 'rgba(255,107,107,0.35)'); grad.addColorStop(1, 'rgba(10,14,20,0)');
    const xStep = plotW / (series.length - 1);
    const xAt = i => pad.left + i * xStep; const yAt = v => pad.top + plotH - ((v - min) / range) * plotH;
    ctx.beginPath(); ctx.moveTo(xAt(0), yAt(series[0]));
    for (let i = 1; i < series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
    ctx.lineTo(xAt(series.length - 1), H - pad.bottom); ctx.lineTo(xAt(0), H - pad.bottom); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath(); ctx.moveTo(xAt(0), yAt(series[0]));
    for (let i = 1; i < series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
    ctx.strokeStyle = lineCol; ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.stroke();
    if (min < 0 && max > 0) { const y0 = yAt(0); ctx.beginPath(); ctx.moveTo(pad.left, y0); ctx.lineTo(W - pad.right, y0); ctx.strokeStyle = 'rgba(160,170,184,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]); }
    ctx.fillStyle = '#6a7280'; ctx.font = '9px "IBM Plex Mono",monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('start', pad.left, H - pad.bottom + 8); ctx.fillText('mid', pad.left + plotW / 2, H - pad.bottom + 8); ctx.fillText('now', W - pad.right, H - pad.bottom + 8);
    const lvLabel = (lastVal >= 0 ? '+$' : '-$') + Math.abs(lastVal).toFixed(2);
    ctx.fillStyle = lineCol; ctx.font = 'bold 11px "IBM Plex Mono",monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText(lvLabel, W - pad.right, pad.top - 14);
}

function _wrapText(text, maxChars) {
    const words = String(text).split(/\s+/); const lines = []; let cur = '';
    for (const w of words) { if (!cur) { cur = w; continue; } if ((cur + ' ' + w).length > maxChars) { lines.push(cur); cur = w; } else cur += ' ' + w; }
    if (cur) lines.push(cur); return lines;
}

let _equityResizeTimer = null;
window.addEventListener('resize', () => {
    if (_equityResizeTimer) clearTimeout(_equityResizeTimer);
    _equityResizeTimer = setTimeout(() => { const perfTab = document.getElementById('tab-performance'); if (perfTab && perfTab.classList.contains('active')) renderEquityCurve(equityData || []); }, 150);
});

function fetchBotSnapshot() {
    fetch('/bot/status').then(r => r.json()).then(data => {
        const bot = data.bot || {};
        _setText('snap-bot-status', bot.running ? 'Running \u2713' : 'Idle');
        const snapStatus = document.getElementById('snap-bot-status'); if (snapStatus) snapStatus.style.color = bot.running ? 'var(--green)' : 'var(--txt2)';
        _setText('snap-bias', bot.market_bias || '\u2014');
        _setText('snap-confidence', bot.confidence !== undefined ? (bot.confidence * 100).toFixed(0) + '%' : '\u2014');
        _setText('snap-last-signal', bot.last_signal || 'None');
        _setText('snap-session-summary', bot.session_summary || 'Waiting for bot to connect\u2026');
        const tradeDetails = document.getElementById('snap-trade-details');
        if (tradeDetails) {
            const trades = bot.open_trades || [];
            tradeDetails.innerHTML = trades.length > 0 ? trades.map(t => `<div style="padding:10px;background:var(--bg1);border-radius:6px;border:1px solid var(--line)"><div style="font-weight:600;color:var(--cyan);margin-bottom:4px">${_esc(t.symbol)} ${_esc(t.direction||'')}</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:10px"><div>Entry: <span style="color:var(--txt)">${t.entry_price||'\u2014'}</span></div><div>SL: <span style="color:var(--red)">${t.stop_loss||'\u2014'}</span></div><div>TP: <span style="color:var(--green)">${t.take_profit||'\u2014'}</span></div><div>P&amp;L: <span style="color:${(t.profit||0)>0?'var(--green)':'var(--red)'}">${(t.profit||0)>0?'+':''}${typeof t.profit==='number'?t.profit.toFixed(2):(t.profit||0)}</span></div></div></div>`).join('') : '<div style="padding:12px;background:var(--bg1);border-radius:6px;color:var(--txt2);text-align:center">No active trades</div>';
        }
    }).catch(() => {});
}

function updatePortfolioWatch(positions) {
    const pairs = ['GBPJPY', 'EURJPY', 'GBPUSD', 'EURUSD'];
    const pairStats = {}; pairs.forEach(p => { pairStats[p] = { trades: 0, pnl: 0, hasTrade: false }; });
    (positions || []).forEach(pos => { if (pairStats[pos.symbol]) { pairStats[pos.symbol].trades++; pairStats[pos.symbol].pnl += pos.profit || 0; pairStats[pos.symbol].hasTrade = true; } });
    pairs.forEach(pair => {
        const stats = pairStats[pair];
        const statusEl = document.getElementById('status-' + pair);
        if (statusEl) { const dotEl = statusEl.querySelector('.watch-dot'); const stateEl = statusEl.querySelector('.watch-state'); if (dotEl) dotEl.className = 'watch-dot ' + (stats.hasTrade ? 'trading' : 'idle'); if (stateEl) stateEl.textContent = stats.hasTrade ? 'trading' : 'idle'; }
        const tradesEl = document.getElementById('trades-' + pair); if (tradesEl) tradesEl.textContent = stats.trades;
        const pnlEl = document.getElementById('pnl-' + pair);
        if (pnlEl) { pnlEl.textContent = stats.pnl >= 0 ? '+$' + stats.pnl.toFixed(0) : '-$' + Math.abs(stats.pnl).toFixed(0); pnlEl.style.color = stats.pnl >= 0 ? 'var(--green)' : 'var(--red)'; }
    });
}

setInterval(() => {
    fetch('/api/mt5/status').then(r => r.json()).then(data => {
        updateMt5Display(data); setConnectButtonState(!!data.connected);
        fetch('/bot/status').then(b => b.json()).then(botData => {
            const bot = botData.bot || {}; const runningNow = !!(bot.running && data.connected);
            if (runningNow && !__botWasRunning) autoShowAiLogTab();
            if (!runningNow && !__userHasChosenTab) __autoAiLogShown = false;
            __botWasRunning = runningNow; _setBotRunning(runningNow);
            const signalsTab = document.getElementById('tab-signals');
            if (signalsTab && signalsTab.classList.contains('active') && data.connected) fetchMarketScanBreakdown();
        }).catch(() => {});
    }).catch(() => {});
}, 2000);

setInterval(() => {
    const url = lastThoughtTs ? '/bot/ai_thoughts?limit=60&since=' + encodeURIComponent(lastThoughtTs) : '/bot/ai_thoughts?limit=60';
    fetch(url).then(r => r.json()).then(data => { if (!data.ok) return; const thoughts = data.thoughts || []; if (thoughts.length > 0) lastThoughtTs = thoughts[thoughts.length - 1].ts; displayThoughts(thoughts); }).catch(() => {});
}, 2000);

setInterval(() => {
    fetch('/bot/positions').then(r => r.json()).then(data => {
        const positions = data.positions || [];
        // Rebuild per-symbol lookup so signal cards can show active trade info
        const symbols = ['EURUSD','GBPUSD','EURJPY','GBPJPY'];
        symbols.forEach(s => { openPositionsBySymbol[s] = null; });
        positions.forEach(p => { if (p.symbol) openPositionsBySymbol[p.symbol] = p; });
        updatePortfolioWatch(positions);
        // Re-render signal cards so the active-trade overlay stays current
        const sigTab = document.getElementById('tab-signals');
        if (sigTab && sigTab.classList.contains('active')) renderMarketScanBreakdown();
    }).catch(() => {});
    const posTab = document.getElementById('tab-positions'); if (posTab && posTab.classList.contains('active')) fetchPositions();
}, 5000);

setInterval(() => { const snapTab = document.getElementById('tab-snapshot'); if (snapTab && snapTab.classList.contains('active')) fetchBotSnapshot(); }, 3000);
setInterval(() => { const histTab = document.getElementById('tab-history'); if (histTab && histTab.classList.contains('active')) fetchHistory(); }, 10000);
setInterval(() => { const perfTab = document.getElementById('tab-performance'); if (perfTab && perfTab.classList.contains('active')) fetchPerformance(); }, 10000);

// ── Per-Pair Strategy tuning panel ────────────────────────────
// Fetches /api/agent/tuning every 8 s and renders per-pair strategy
// tiles in the sidebar. Populated by backtest_insights.json which the
// backtester writes after every run.
function _fetchAndRenderTuning() {
    fetch('/api/agent/tuning')
        .then(r => r.json())
        .then(data => {
            const tiles  = document.getElementById('tuning-tiles');
            const badge  = document.getElementById('tuning-source-badge');
            const meta   = document.getElementById('tuning-meta');
            if (!tiles) return;

            const pairs = data.pairs || {};
            const anyBacktest = Object.values(pairs).some(p => p.source === 'backtest');

            if (badge) {
                badge.textContent = anyBacktest ? 'backtest' : 'defaults';
                badge.style.background = anyBacktest ? 'rgba(80,217,99,0.15)' : 'rgba(106,114,128,0.15)';
                badge.style.color      = anyBacktest ? 'var(--green)' : 'var(--txt3)';
            }

            const LABELS = {
                sl_atr_mult:        'SL atr ×',
                tp_atr_mult:        'TP atr ×',
                be_buffer_pips:     'BE buffer p',
                min_atr_to_tighten: 'min atr to tighten',
                partial_close_rr:   'partial close R',
                trail_atr_mult:     'trail atr ×',
                atr_tolerance_mult: 'atr tolerance ×',
            };

            tiles.innerHTML = Object.entries(pairs).map(([sym, info]) => {
                const params  = info.params || {};
                const isBt    = info.source === 'backtest';
                const wr      = info.win_rate != null ? (info.win_rate * 100).toFixed(1) + '% WR' : '';
                const n       = info.trade_count != null ? info.trade_count + ' trades' : '';
                const srcCol  = isBt ? 'var(--green)' : 'var(--txt3)';
                const srcTxt  = isBt ? 'tuned' : 'default';

                const rows = Object.entries(LABELS).map(([k, label]) => {
                    const val = params[k];
                    if (val == null) return '';
                    return `<div style="display:flex;justify-content:space-between;font-size:9px;padding:1px 0">
                        <span style="color:var(--txt3)">${label}</span>
                        <span style="color:var(--txt);font-family:var(--mono)">${Number(val).toFixed(2)}</span>
                    </div>`;
                }).join('');

                return `<div style="background:var(--bg1);border-radius:5px;padding:8px;border-left:2px solid ${srcCol}">
                    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
                        <span style="font-weight:600;color:var(--txt);font-size:10px">${_esc(sym)}</span>
                        <span style="font-size:8px;color:${srcCol};letter-spacing:.04em">${srcTxt}</span>
                    </div>
                    ${rows}
                    ${(wr||n) ? `<div style="font-size:8px;color:var(--txt3);margin-top:4px">${[n,wr].filter(Boolean).join(' · ')}</div>` : ''}
                </div>`;
            }).join('');

            if (meta && data.last_mtime) {
                meta.textContent = 'Updated ' + new Date(data.last_mtime + 'Z').toLocaleTimeString();
            }
        })
        .catch(() => {});
}

_fetchAndRenderTuning();
setInterval(_fetchAndRenderTuning, 8000);


// Fetches /api/ollama/health every 5 s and updates the Agents
// panel in the sidebar (ollama-dot, agent roster, learning row).
function _pollOllamaAndRoster() {
    fetch('/api/ollama/health')
        .then(r => r.json())
        .then(data => {
            const dot     = document.getElementById('ollama-dot');
            const txt     = document.getElementById('ollama-status-txt');
            const detail  = document.getElementById('ollama-detail');
            const reach   = !!data.reachable;
            const loaded  = !!data.model_loaded;
            const ok      = reach && loaded;

            if (dot) dot.className = 'mt5-dot ' + (ok ? 'ok' : reach ? 'warn' : 'err');
            if (txt) {
                txt.textContent = ok ? 'LLM ready' : reach ? 'model not loaded' : 'offline';
                txt.style.color = ok ? 'var(--green)' : reach ? 'var(--amber)' : 'var(--red)';
            }
            if (detail) {
                detail.style.display = reach ? 'flex' : 'none';
                _setText('ol-url',   data.url   || '—');
                _setText('ol-model', data.model || '—');
                _setText('ol-loaded', loaded ? 'yes' : 'no');
                const learnRow = document.getElementById('ol-learning-row');
                const learnEl  = document.getElementById('ol-learning');
                if (data.learning && learnEl) {
                    if (learnRow) learnRow.style.display = 'flex';
                    learnEl.textContent = data.learning.pair_count
                        ? data.learning.pair_count + ' pairs'
                        : 'no data yet';
                    learnEl.style.color = data.learning.pair_count ? 'var(--green)' : 'var(--txt3)';
                }
            }

            // Update agent roster dots + meta
            _renderAgentRoster(ok);
        })
        .catch(() => {
            const dot = document.getElementById('ollama-dot');
            const txt = document.getElementById('ollama-status-txt');
            if (dot) dot.className = 'mt5-dot err';
            if (txt) { txt.textContent = 'offline'; txt.style.color = 'var(--red)'; }
            _renderAgentRoster(false);
        });
}

function _renderAgentRoster(ollamaReady) {
    const botRunning = document.getElementById('status-text')?.textContent === 'live';

    // ── Agent 0: Orchestrator (LLM) ───────────────────────────────
    // GREEN  = Ollama up + model loaded + bot running  → actively reviewing verdicts
    // AMBER  = Ollama up + model loaded, bot not yet running  → standing by
    // RED    = Ollama offline or model not loaded  → cannot orchestrate
    const orchRow = document.querySelector('[data-agent-id="orchestrator"]');
    if (orchRow) {
        const dot  = orchRow.querySelector('.agent-dot');
        const meta = orchRow.querySelector('.agent-meta');
        const dotClass = ollamaReady && botRunning ? 'ready'
                       : ollamaReady              ? 'warn'
                       :                            'err';
        const label    = ollamaReady && botRunning ? 'REVIEWING'
                       : ollamaReady              ? 'STANDBY'
                       :                            'OFFLINE';
        const colour   = ollamaReady && botRunning ? 'var(--green)'
                       : ollamaReady              ? 'var(--amber,#f5a623)'
                       :                            'var(--red,#ff6b6b)';
        if (dot)  dot.className  = 'agent-dot ' + dotClass;
        if (meta) { meta.textContent = label; meta.style.color = colour; }
    }

    // ── Bots 1–4: Pair bots (deterministic) ────────────────────────
    // GREEN  = Bot running  → actively scanning pair for CHoCH + level entries
    // GREY   = Bot not started  → idle, no scanning
    // These bots don’t need Ollama — they’re fully deterministic.
    const ids = ['agent-eurusd','agent-gbpusd','agent-gbpjpy','agent-eurjpy'];
    ids.forEach(id => {
        const row = document.querySelector('[data-agent-id="' + id + '"]');
        if (!row) return;
        const dot  = row.querySelector('.agent-dot');
        const meta = row.querySelector('.agent-meta');
        if (dot)  dot.className  = 'agent-dot ' + (botRunning ? 'ready' : '');
        if (meta) {
            meta.textContent = botRunning ? 'SCANNING' : 'IDLE';
            meta.style.color = botRunning ? 'var(--green)' : 'var(--txt3)';
        }
    });
}

// Start Ollama polling immediately and repeat every 5 s.
_pollOllamaAndRoster();
setInterval(_pollOllamaAndRoster, 5000);

function _setText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }
function _setKPI(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }
function _esc(str) { if (!str) return ''; return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }

// ── Backtest tab ──────────────────────────────────────────────────────────────

function _btSetStatus(text, isError) { const el = document.getElementById('bt-status'); if (!el) return; el.textContent = text || ''; el.style.color = isError ? 'var(--red, #ff6b6b)' : 'var(--txt3)'; }

function _btClearResults() {
    ['bt-total-trades','bt-win-rate','bt-total-pnl','bt-avg-pnl','bt-max-dd'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '\u2014'; });
    const body = document.getElementById('bt-pairs-body'); if (body) body.innerHTML = '<div style="color:var(--txt3);padding:8px;grid-column:1/-1">Running\u2026</div>';
    const chart = document.getElementById('bt-pnl-chart'); if (chart) chart.innerHTML = '<div style="color:var(--txt3);padding:6px 0">Running\u2026</div>';
}

function _btMoney(v) { const n = Number(v||0); return (n>=0?'+':'-')+'$'+Math.abs(n).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
function _btMoneyShort(v) { const n=Number(v||0);const abs=Math.abs(n);const s=n>=0?'+':'-'; return abs>=1000?s+'$'+(abs/1000).toFixed(1)+'k':s+'$'+abs.toFixed(0); }

function _btRenderPnlChart(per, totalPnl) {
    const chart = document.getElementById('bt-pnl-chart'); if (!chart) return;
    const rows = (per||[]).filter(r=>!r.error);
    if (!rows.length) { chart.innerHTML = '<div style="color:var(--txt3);padding:6px 0">No per-pair P&amp;L to visualise.</div>'; return; }
    const maxAbs = Math.max(1, ...rows.map(r=>Math.abs(Number(r.total_pnl||0))));
    const totalAbs = rows.reduce((a,r)=>a+Math.abs(Number(r.total_pnl||0)),0)||1;
    chart.innerHTML = rows.map(r => {
        const pnl=Number(r.total_pnl||0);const pct=(Math.abs(pnl)/maxAbs)*100;const share=(Math.abs(pnl)/totalAbs)*100;
        const colour=pnl>=0?'var(--green,#2ecc71)':'var(--red,#ff6b6b)';const bg=pnl>=0?'rgba(46,204,113,0.15)':'rgba(255,107,107,0.15)';
        return `<div style="display:grid;grid-template-columns:80px 1fr 110px 70px;align-items:center;gap:10px"><div style="color:var(--txt2)">${_esc(r.symbol)}</div><div style="position:relative;height:16px;background:var(--bg1);border-radius:4px;overflow:hidden"><div style="position:absolute;left:0;top:0;bottom:0;width:${pct.toFixed(1)}%;background:${bg};border-right:2px solid ${colour};transition:width .25s"></div></div><div style="text-align:right;color:${colour}">${_esc(r.pnl_label||_btMoney(pnl))}</div><div style="text-align:right;color:var(--txt3)">${share.toFixed(1)}%</div></div>`;
    }).join('');
}

function _btRenderPairCards(per) {
    const body = document.getElementById('bt-pairs-body'); if (!body) return;
    const rows = per || [];
    if (!rows.length) { body.innerHTML = '<div style="color:var(--txt3);padding:8px">No per-pair results returned.</div>'; return; }

    body.innerHTML = rows.map(r => {
        if (r.error) return `<div style="background:var(--bg1);padding:12px;border-radius:6px;border-left:3px solid var(--red,#ff6b6b)"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px"><div style="font-weight:600;color:var(--txt)">${_esc(r.symbol)}</div><div style="font-size:10px;color:var(--red,#ff6b6b)">FAILED</div></div><div style="font-size:10px;color:var(--txt3)">${_esc(r.error)}</div></div>`;

        const pnl      = Number(r.total_pnl||0);
        const pnlColor = pnl>=0?'var(--green)':'var(--red,#ff6b6b)';
        const wrPct    = Math.max(0,Math.min(100,Number(r.win_rate||0)*100));
        const wrLabel  = r.win_rate_label||(wrPct.toFixed(1)+'%');
        const pnlLabel = r.pnl_label||_btMoney(pnl);
        const pf       = Number(r.profit_factor||0);

        // ── Profit Factor display ─────────────────────────────────────────────
        // 999 = backend sentinel: no losing trades → profit factor is infinite.
        // ∞* (amber) = no losses YET on a thin sample (<20 trades) — suspicious,
        //              not impressive. Just means strategy hasn't been tested enough.
        // ∞  (green) = genuinely zero losses across n≥20 trades — meaningful edge.
        const n        = Number(r.trades||0);
        const pfLabel  = pf>=999 ? (n<20?'\u221e*':'\u221e') : (pf>0?pf.toFixed(2)+'x':'\u2014');
        const pfColor  = pf>=999 ? (n<20?'var(--amber,#f5a623)':'var(--green)')
                       : pf>=1.5  ? 'var(--green)'
                       : pf>=1.0  ? 'var(--amber,#f5a623)'
                       : pf>0     ? 'var(--red,#ff6b6b)'
                       : 'var(--txt)';

        const rr       = Number(r.avg_rr||0);
        const rrLabel  = rr>0?rr.toFixed(2):'\u2014';
        const trades   = n;
        const wins     = Number(r.wins||0);
        const losses   = Number(r.losses||0);
        const mdd      = Number(r.max_drawdown||0);
        const mddLabel = mdd>0?('-$'+mdd.toFixed(0)):'\u2014';
        const barColor = wrPct>=55?'var(--green)':(wrPct>=45?'var(--amber,#f5a623)':'var(--red,#ff6b6b)');

        // ── Sample-size warning ───────────────────────────────────────────────
        // <5 trades: metrics are completely meaningless noise
        // <20 trades: matches ML training threshold — not enough for reliable signal
        let sampleBadge = '';
        if (trades>0 && trades<5) {
            sampleBadge = `<span style="background:rgba(255,107,107,0.15);color:var(--red,#ff6b6b);padding:2px 6px;border-radius:3px;font-size:9px;font-weight:600;letter-spacing:.04em;margin-left:6px" title="Too few trades \u2014 win-rate and profit-factor are not meaningful. Run more backtests to accumulate data.">VERY THIN n=${trades}</span>`;
        } else if (trades<20) {
            sampleBadge = `<span style="background:rgba(245,166,35,0.15);color:var(--amber,#f5a623);padding:2px 6px;border-radius:3px;font-size:9px;font-weight:600;letter-spacing:.04em;margin-left:6px" title="Sample below 20 trades \u2014 metrics are indicative only. ML training also needs n\u226520.">THIN n=${trades}</span>`;
        }

        return `<div style="background:var(--bg1);padding:12px;border-radius:6px;border-left:3px solid ${pnlColor}">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
              <div style="display:flex;align-items:baseline;flex-wrap:wrap"><div style="font-weight:600;color:var(--txt);font-size:12px">${_esc(r.symbol)}</div>${sampleBadge}</div>
              <div style="color:${pnlColor};font-weight:600">${_esc(pnlLabel)}</div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--txt3);margin-bottom:4px"><span>Win Rate</span><span style="color:var(--txt2)">${_esc(wrLabel)} \u00b7 ${wins}W / ${losses}L</span></div>
            <div style="height:6px;background:var(--bg0);border-radius:3px;overflow:hidden;margin-bottom:10px"><div style="height:100%;width:${wrPct.toFixed(1)}%;background:${barColor};transition:width .25s"></div></div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:10px;color:var(--txt3)">
              <div><div>Trades</div><div style="color:var(--txt);font-size:11px">${trades.toLocaleString()}</div></div>
              <div title="Profit Factor: gross win pips \u00f7 gross loss pips.\n\u221e* = no losses yet but thin sample \u2014 not reliable.\n\u221e = genuinely zero losses on n\u226520 trades.\n&lt;1.0 = losing strategy. 1.0\u20131.5 = marginal. \u22651.5 = healthy.">
                <div>Profit Factor</div><div style="color:${pfColor};font-size:11px;font-weight:600">${_esc(pfLabel)}</div>
              </div>
              <div title="Avg R:R across ALL decided trades (wins + losses).\nA losing pair can still show ~1.0 R:R if losses hit full SL.">
                <div>Avg R:R</div><div style="color:var(--txt);font-size:11px">${_esc(rrLabel)}</div>
              </div>
              <div title="Peak-to-trough drawdown on this pair's equity curve">
                <div>Max DD</div><div style="color:${mdd>0?'var(--red,#ff6b6b)':'var(--txt)'};font-size:11px">${_esc(mddLabel)}</div>
              </div>
            </div>
          </div>`;
    }).join('');
}

function _btRenderResults(data) {
    const agg=data.aggregate||{};const set=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v;};
    set('bt-total-trades',(agg.total_trades??0).toLocaleString());set('bt-win-rate',agg.win_rate_label||'0.0%');set('bt-total-pnl',agg.total_pnl_label||'$0.00');set('bt-avg-pnl',agg.avg_pnl_label||'$0.00');
    const mddVal=Number(agg.max_drawdown||0);const mddLabel=agg.max_drawdown_label||(mddVal>0?('-$'+mddVal.toFixed(2)):'$0.00');
    set('bt-max-dd',mddLabel);const ddEl=document.getElementById('bt-max-dd');if(ddEl)ddEl.style.color=mddVal>0?'var(--red,#ff6b6b)':'';
    const mddPct=Number(agg.max_drawdown_pct||0);set('bt-max-dd-deco',mddPct>0?(mddPct.toFixed(1)+'% of peak'):'peak to trough');
    const wins=Number(agg.wins||0),losses=Number(agg.losses||0);set('bt-total-trades-deco',wins||losses?(wins+'W \u00b7 '+losses+'L'):'backtest');
    const pairCount=(data.per_pair||[]).length;set('bt-win-rate-deco',pairCount?('across '+pairCount+' pair'+(pairCount>1?'s':'')):'aggregate');
    const lot=Number(data.lot_size||0).toFixed(2);
    const lotEl=document.getElementById('bt-lot-label');if(lotEl)lotEl.textContent='at '+lot+' lot';
    const lotEl2=document.getElementById('bt-lot-label-2');if(lotEl2)lotEl2.textContent='('+lot+' lot)';
    const sub=document.getElementById('bt-subtitle');if(sub)sub.textContent=(data.days||7)+'-day raw-strategy validation ('+lot+' lot, single SL/TP exit)'+(data.period?' \u2014 '+data.period:'');
    const banner=document.getElementById('bt-summary-banner');if(banner){banner.style.display='flex';set('bt-last-run',new Date().toLocaleString());set('bt-period',data.period||'\u2014');set('bt-duration',(data.duration_s!=null)?(data.duration_s+'s'):'\u2014');set('bt-lot-summary',lot);}
    _btRenderPnlChart(data.per_pair||[],Number(agg.total_pnl||0));_btRenderPairCards(data.per_pair||[]);
}

function _btNormalisePair(r) {
    if(!r)return r;const pnl=Number(r.total_pnl||0);
    return {symbol:r.symbol,error:r.error,total_pnl:pnl,pnl_label:_btMoney(pnl),win_rate:Number(r.win_rate||0),win_rate_label:((Number(r.win_rate||0))*100).toFixed(1)+'%',trades:Number(r.total_trades||0),wins:Number(r.wins||0),losses:Number(r.losses||0),profit_factor:Number(r.profit_factor||0),avg_rr:Number(r.avg_rr_achieved||0),avg_win_pips:Number(r.avg_win_pips||0),avg_loss_pips:Number(r.avg_loss_pips||0),feature_importance:r.feature_importance||null,assumptions:r.assumptions||null,max_drawdown:Number(r.max_drawdown||0),max_drawdown_pct:Number(r.max_drawdown_pct||0),equity_curve:r.equity_curve||[],by_env:r.by_env||{},by_side:r.by_side||{},by_hour:r.by_hour||{},by_dow:r.by_dow||{},confidence_buckets:r.confidence_buckets||{},pnl_distribution:r.pnl_distribution||{},component_correlations:r.component_correlations||{},by_quality:r.by_quality||{},by_exit_reason:r.by_exit_reason||{}};
}

function _btRenderAssumptions(perPair) {
    let host=document.getElementById('bt-assumptions');
    if(!host){const body=document.getElementById('bt-pairs-body');if(!body||!body.parentNode)return;host=document.createElement('div');host.id='bt-assumptions';host.style.cssText='margin-top:12px;padding:10px 12px;border:1px dashed var(--bd,#2a2f3a);border-radius:6px;background:var(--bg0);font-size:10px;color:var(--txt3);line-height:1.5';body.parentNode.parentNode.appendChild(host);}
    const first=(perPair||[]).find(p=>p&&p.assumptions);if(!first){host.innerHTML='';return;}
    const a=first.assumptions||{};const spread=(a.spread_pips==null)?'default per-pair (1.0 majors / 2.0 JPY)':(a.spread_pips+' pips (override)');
    const tmRaw=/raw/i.test(String(a.trade_management||''));const tmColour=tmRaw?'var(--accent,#5ac8fa)':'var(--txt2)';
    host.innerHTML=`<div style="font-weight:600;color:var(--txt2);margin-bottom:4px">\ud83d\udcd0 Modelling assumptions</div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 16px"><div><b>Trade management:</b> <span style="color:${tmColour}">${_esc(a.trade_management||'\u2014')}</span></div><div><b>Lot size:</b> ${_esc(String(a.lot_size))}</div><div><b>Spread on entry:</b> ${_esc(spread)}</div><div><b>Intrabar TP/SL:</b> ${_esc(a.intrabar_policy||'\u2014')}</div><div><b>Entry fill:</b> ${_esc(a.entry_fill||'\u2014')}</div><div><b>Pip\u2192USD model:</b> ${_esc(a.pip_usd_model||'\u2014')}</div><div><b>Commission:</b> ${_esc(a.commission||'\u2014')}</div><div><b>Swap:</b> ${_esc(a.swap||'\u2014')}</div><div><b>Lookahead:</b> ${_esc(a.lookahead||'\u2014')}</div></div>`;
}

function _btColourForWR(wr){if(wr>=0.55)return'var(--green,#2ecc71)';if(wr>=0.45)return'var(--amber,#f5a623)';return'var(--red,#ff6b6b)';}

function _btAggregateBuckets(perPair,field){
    const out={};const order=[];
    (perPair||[]).forEach(p=>{const b=(p&&p[field])||{};Object.keys(b).forEach(k=>{const s=b[k]||{};if(!out[k]){out[k]={count:0,wins:0,losses:0,total_pnl:0,pips_sum:0};order.push(k);}const n=Number(s.count||0);out[k].count+=n;out[k].wins+=Number(s.wins||0);out[k].losses+=Number(s.losses||0);out[k].total_pnl+=Number(s.total_pnl||0);out[k].pips_sum+=Number(s.avg_pips||0)*n;});});
    order.forEach(k=>{const s=out[k];s.win_rate=s.count>0?s.wins/s.count:0;s.avg_pips=s.count>0?s.pips_sum/s.count:0;});
    return{order,data:out};
}

function _btBucketBarsHTML(agg,opts){
    opts=opts||{};const order=opts.orderOverride||agg.order;
    if(!order||!order.length)return'<div style="color:var(--txt3);font-size:10px;padding:6px 0">No data in this window.</div>';
    const maxCount=Math.max(1,...order.map(k=>Number((agg.data[k]||{}).count||0)));
    const labelFmt=opts.labelFmt||(k=>k);
    return order.map(k=>{
        const s=agg.data[k]||{};const n=Number(s.count||0);if(!n)return'';
        const wr=Number(s.win_rate||0);const wrLabel=(wr*100).toFixed(0)+'%';const share=(n/maxCount)*100;
        const colour=_btColourForWR(wr);const pnl=Number(s.total_pnl||0);
        const pnlLabel=pnl>=0?('+$'+pnl.toFixed(0)):('-$'+Math.abs(pnl).toFixed(0));const pnlColor=pnl>=0?'var(--green,#2ecc71)':'var(--red,#ff6b6b)';
        return`<div style="display:grid;grid-template-columns:110px 1fr 48px 60px 64px;gap:8px;align-items:center;font-size:10px;margin-bottom:3px" title="${_esc(String(k))} \u00b7 ${n} trade${n===1?'':'s'} \u00b7 ${s.wins||0}W/${s.losses||0}L \u00b7 avg ${(Number(s.avg_pips||0)).toFixed(1)}p"><div style="color:var(--txt2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${_esc(labelFmt(k))}</div><div style="height:12px;background:var(--bg0);border-radius:3px;overflow:hidden;position:relative"><div style="position:absolute;left:0;top:0;bottom:0;width:${share.toFixed(1)}%;background:${colour};opacity:.85"></div><div style="position:absolute;left:0;right:0;top:0;bottom:0;background:linear-gradient(to right, rgba(0,0,0,0) ${(wr*100).toFixed(1)}%, rgba(0,0,0,.35) ${(wr*100).toFixed(1)}%)"></div></div><div style="text-align:right;color:${colour};font-weight:600">${wrLabel}</div><div style="text-align:right;color:var(--txt3)">${n}t</div><div style="text-align:right;color:${pnlColor}">${pnlLabel}</div></div>`;
    }).join('');
}

function _btHourBucketsHTML(perPair){const agg=_btAggregateBuckets(perPair,'by_hour');const hours=Array.from({length:24},(_,i)=>String(i));hours.forEach(h=>{if(!agg.data[h])agg.data[h]={count:0,wins:0,losses:0,total_pnl:0,win_rate:0,avg_pips:0};});return _btBucketBarsHTML({order:hours,data:agg.data},{labelFmt:h=>String(h).padStart(2,'0')+':00'});}
function _btDowBucketsHTML(perPair){const agg=_btAggregateBuckets(perPair,'by_dow');const order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].filter(d=>agg.data[d]);return _btBucketBarsHTML({order,data:agg.data});}
function _btConfidenceBucketsHTML(perPair){const agg=_btAggregateBuckets(perPair,'confidence_buckets');const order=['<50','50-60','60-70','70-80','80-90','90-100'].filter(k=>agg.data[k]);return _btBucketBarsHTML({order,data:agg.data},{labelFmt:k=>k+'%'});}
function _btEnvBucketsHTML(perPair){const agg=_btAggregateBuckets(perPair,'by_env');const order=agg.order.slice().sort((a,b)=>Number((agg.data[b]||{}).count||0)-Number((agg.data[a]||{}).count||0));return _btBucketBarsHTML({order,data:agg.data});}
function _btSideBucketsHTML(perPair){const agg=_btAggregateBuckets(perPair,'by_side');const order=['BUY','SELL'].filter(k=>agg.data[k]);return _btBucketBarsHTML({order,data:agg.data});}
function _btExitReasonBucketsHTML(perPair){const agg=_btAggregateBuckets(perPair,'by_exit_reason');const KNOWN=['tp','sl','timeout','partial+tp','partial+be','partial+timeout','be+tp','be+sl'];const order=KNOWN.filter(k=>agg.data[k]).concat(agg.order.filter(k=>!KNOWN.includes(k)));return _btBucketBarsHTML({order,data:agg.data},{labelFmt:k=>({'tp':'Take Profit','sl':'Stop Loss','timeout':'Timeout','partial+tp':'Partial + TP','partial+be':'Partial + BE','partial+timeout':'Partial + Timeout','be+tp':'Runner TP','be+sl':'Runner BE'}[k]||k)});}
function _btQualityBucketsHTML(perPair){const agg=_btAggregateBuckets(perPair,'by_quality');const order=['strong','good','fair','weak'].filter(k=>agg.data[k]);return _btBucketBarsHTML({order,data:agg.data},{labelFmt:k=>k.charAt(0).toUpperCase()+k.slice(1)});}

function _btPnlDistributionHTML(perPair){
    const pairs=(perPair||[]).filter(p=>p&&p.pnl_distribution&&(p.pnl_distribution.bins||[]).length);
    if(!pairs.length)return'<div style="color:var(--txt3);font-size:10px;padding:6px 0">No P&amp;L distribution to plot.</div>';
    return pairs.map(p=>{
        const pd=p.pnl_distribution||{};const bins=pd.bins||[];const maxC=Math.max(1,...bins.map(b=>Number(b.count||0)));
        const bars=bins.map(b=>{const h=(Number(b.count||0)/maxC)*100;const col=b.sign==='win'?'var(--green,#2ecc71)':(b.sign==='loss'?'var(--red,#ff6b6b)':'var(--txt3)');const midStr=b.mid>=0?('+'+Number(b.mid).toFixed(0)):Number(b.mid).toFixed(0);return`<div style="flex:1;display:flex;flex-direction:column;align-items:center;min-width:0" title="${midStr}p \u00b7 ${b.count} trade${b.count===1?'':'s'} (${Number(b.x0).toFixed(1)} \u2192 ${Number(b.x1).toFixed(1)})"><div style="width:100%;height:70px;display:flex;align-items:flex-end;justify-content:center"><div style="width:70%;height:${h.toFixed(1)}%;background:${col};border-radius:2px 2px 0 0;min-height:${b.count>0?'2px':'0'}"></div></div><div style="font-size:9px;color:var(--txt3);margin-top:2px">${midStr}</div></div>`;}).join('');
        return`<div style="background:var(--bg1);padding:10px;border-radius:6px;margin-bottom:8px"><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;font-size:10px;color:var(--txt3)"><span style="color:var(--txt);font-weight:600">${_esc(p.symbol)}</span><span>n=${pd.n||0} \u00b7 median ${Number(pd.median||0).toFixed(1)}p</span></div><div style="display:flex;gap:2px;align-items:flex-end;padding:0 4px">${bars}</div></div>`;
    }).join('');
}

function _btCorrelationsHTML(perPair){
    const acc={};
    (perPair||[]).forEach(p=>{const cc=(p&&p.component_correlations)||{};const w=Number(p.trades||0);Object.keys(cc).forEach(k=>{const v=Number(cc[k]||0);if(!isFinite(v))return;if(!acc[k])acc[k]={sum:0,weight:0};acc[k].sum+=v*w;acc[k].weight+=w;});});
    const keys=Object.keys(acc);if(!keys.length)return'<div style="color:var(--txt3);font-size:10px;padding:6px 0">No component correlations available.</div>';
    const items=keys.map(k=>({component:k,coef:acc[k].weight>0?acc[k].sum/acc[k].weight:0})).sort((a,b)=>Math.abs(b.coef)-Math.abs(a.coef));
    return items.map(it=>{const centred=50+(it.coef*50);const sign=it.coef>=0?'+':'-';const coefLabel=sign+Math.abs(it.coef).toFixed(2);const colour=it.coef>0.15?'var(--green,#2ecc71)':it.coef<-0.15?'var(--red,#ff6b6b)':'var(--amber,#f5a623)';const barLeft=it.coef>=0?'50%':centred.toFixed(1)+'%';const barWidth=Math.abs(it.coef*50).toFixed(1)+'%';return`<div style="display:grid;grid-template-columns:160px 1fr 56px;gap:8px;align-items:center;font-size:10px;margin-bottom:3px"><div style="color:var(--txt2)">${_esc(it.component)}</div><div style="height:12px;background:var(--bg0);border-radius:3px;position:relative"><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--bd,#2a2f3a)"></div><div style="position:absolute;left:${barLeft};top:0;bottom:0;width:${barWidth};background:${colour};opacity:.85;border-radius:2px"></div></div><div style="text-align:right;color:${colour};font-weight:600">${coefLabel}</div></div>`;}).join('');
}

function _btRenderCorrelations(perPair){
    let host=document.getElementById('bt-correlations');
    if(!host){const body=document.getElementById('bt-pairs-body');if(!body||!body.parentNode)return;host=document.createElement('div');host.id='bt-correlations';host.style.cssText='margin-top:16px;border:1px solid var(--bd,#2a2f3a);padding:14px;border-radius:8px;background:var(--bg0)';const mlHost=document.getElementById('bt-ml-insights');if(mlHost&&mlHost.parentNode)mlHost.parentNode.insertBefore(host,mlHost);else body.parentNode.parentNode.appendChild(host);}
    const withData=(perPair||[]).filter(p=>p&&!p.error);
    if(!withData.length){host.innerHTML='<div style="color:var(--txt3);font-size:11px">\ud83d\udcca Win/Loss Correlations \u2014 no trades in this window.</div>';return;}
    const totalTrades=withData.reduce((a,p)=>a+Number(p.trades||0),0);
    const hasAnyBreakdown=withData.some(p=>(p.by_env&&Object.keys(p.by_env).length)||(p.by_hour&&Object.keys(p.by_hour).length)||(p.by_side&&Object.keys(p.by_side).length));
    if(totalTrades>0&&!hasAnyBreakdown){host.innerHTML=`<div style="font-weight:600;color:var(--txt);margin-bottom:8px">\ud83d\udcca Win / Loss Correlations</div><div style="padding:12px;background:var(--bg1);border-left:3px solid var(--amber,#f5a623);border-radius:4px;font-size:11px;color:var(--txt2);line-height:1.6"><b>Data pipeline mismatch.</b> ${totalTrades.toLocaleString()} trade${totalTrades===1?'':'s'} returned but no bucket breakdowns. Restart Flask server so updated code loads, then run the backtest again.</div>`;return;}
    const panel=(title,subtitle,bodyHTML)=>`<div style="background:var(--bg1);padding:12px;border-radius:6px"><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px"><div style="font-weight:600;color:var(--txt);font-size:11px">${_esc(title)}</div><div style="font-size:9px;color:var(--txt3)">${_esc(subtitle||'')}</div></div>${bodyHTML}</div>`;
    const legend=`<div style="display:flex;gap:12px;font-size:9px;color:var(--txt3);margin-bottom:10px;flex-wrap:wrap"><span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--green,#2ecc71);border-radius:2px;display:inline-block"></span>WR \u2265 55%</span><span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--amber,#f5a623);border-radius:2px;display:inline-block"></span>45\u201355%</span><span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--red,#ff6b6b);border-radius:2px;display:inline-block"></span>&lt; 45%</span><span style="margin-left:auto">bar width = trade volume \u00b7 number = win rate \u00b7 right col = total P&amp;L</span></div>`;
    host.innerHTML=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px"><div style="font-weight:600;color:var(--txt)">\ud83d\udcca Win / Loss Correlations</div><div style="font-size:10px;color:var(--txt3)">what tends to win \u2014 and lose \u2014 in this window</div></div>${legend}<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px">${panel('By Environment','CHoCH-BUY@PDL, Continuation-SELL@PDH, \u2026',_btEnvBucketsHTML(withData))}${panel('By Side','BUY vs SELL',_btSideBucketsHTML(withData))}${panel('By Signal Quality','weak / fair / good / strong',_btQualityBucketsHTML(withData))}${panel('By Exit Reason','how trades closed',_btExitReasonBucketsHTML(withData))}${panel('By Confidence','model-assigned confidence bucket',_btConfidenceBucketsHTML(withData))}${panel('By Day of Week','entry day (UTC)',_btDowBucketsHTML(withData))}${panel('By Hour of Day','entry hour, UTC \u2014 00\u201323',_btHourBucketsHTML(withData))}${panel('Component Correlations','structure / level / momentum / spread / env \u2014 +1 = wins, \u22121 = losses',_btCorrelationsHTML(withData))}</div><div style="margin-top:12px"><div style="font-weight:600;color:var(--txt);font-size:11px;margin-bottom:6px">P&amp;L Distribution (pips, per pair)</div>${_btPnlDistributionHTML(withData)}</div>`;
}

function _btRenderMLInsights(perPair){
    let host=document.getElementById('bt-ml-insights');
    if(!host){const body=document.getElementById('bt-pairs-body');if(!body||!body.parentNode)return;host=document.createElement('div');host.id='bt-ml-insights';host.style.cssText='margin-top:16px;border:1px solid var(--bd,#2a2f3a);padding:14px;border-radius:8px;background:var(--bg0)';body.parentNode.parentNode.appendChild(host);}
    const withFI=(perPair||[]).filter(p=>p&&p.feature_importance);
    if(!withFI.length){host.innerHTML='<div style="color:var(--txt3);font-size:11px">\ud83e\udde0 ML Insights (sklearn) \u2014 no trained model (pip install scikit-learn, or too few decided trades).</div>';return;}
    const card=(p)=>{
        const fi=p.feature_importance||{};
        if(fi.error)return`<div style="background:var(--bg1);padding:10px;border-radius:6px;border-left:3px solid var(--amber,#f5a623)"><div style="font-weight:600;color:var(--txt);font-size:12px;margin-bottom:4px">${_esc(p.symbol)}</div><div style="font-size:10px;color:var(--txt3)">${_esc(fi.error)}</div></div>`;
        const imps=Array.isArray(fi.importances)?fi.importances.slice(0,8):[];const maxImp=Math.max(0.0001,...imps.map(r=>Number(r.importance||0)));
        const rows=imps.map(r=>{const v=Number(r.importance||0);const pct=(v/maxImp)*100;return`<div style="display:grid;grid-template-columns:130px 1fr 50px;gap:8px;align-items:center;font-size:10px;margin-bottom:4px"><div style="color:var(--txt2)">${_esc(r.feature)}</div><div style="height:10px;background:var(--bg0);border-radius:3px;overflow:hidden"><div style="height:100%;width:${pct.toFixed(1)}%;background:var(--accent,#5ac8fa)"></div></div><div style="text-align:right;color:var(--txt3)">${v.toFixed(3)}</div></div>`;}).join('');
        const bal=fi.balanced_accuracy!=null?fi.balanced_accuracy:null;const lift=fi.lift_vs_random!=null?fi.lift_vs_random:null;
        const liftLabel=lift!=null?((lift>=0?'+':'')+(lift*100).toFixed(1)+'%'):'\u2014';
        const liftColour=lift!=null&&lift>0.02?'var(--green)':lift!=null&&lift<-0.02?'var(--red,#ff6b6b)':'var(--txt3)';
        const baseline=fi.baseline_win_rate!=null?(fi.baseline_win_rate*100).toFixed(1)+'%':'\u2014';
        const balLabel=bal!=null?(bal*100).toFixed(1)+'%':'\u2014';
        const prec=fi.win_precision!=null?(fi.win_precision*100).toFixed(1)+'%':'\u2014';
        const rec=fi.win_recall!=null?(fi.win_recall*100).toFixed(1)+'%':'\u2014';
        return`<div style="background:var(--bg1);padding:12px;border-radius:6px"><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px"><div style="font-weight:600;color:var(--txt);font-size:12px">${_esc(p.symbol)}</div><div style="font-size:10px;color:var(--txt3)">n=${fi.n_trades}</div></div><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:10px;color:var(--txt3);margin-bottom:10px"><div title="Share of trades that were WINS"><div>Baseline WR</div><div style="color:var(--txt);font-size:11px">${baseline}</div></div><div title="Avg of WIN-recall and LOSS-recall from OOB predictions. 50% = coin flip."><div>Balanced Acc</div><div style="color:var(--txt);font-size:11px">${balLabel}</div></div><div title="When model predicts WIN, how often correct?"><div>Win Precision</div><div style="color:var(--txt);font-size:11px">${prec}</div></div><div title="Of all actual WINs, how many did model catch?"><div>Win Recall</div><div style="color:var(--txt);font-size:11px">${rec}</div></div></div><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;font-size:10px;color:var(--txt3)"><span>Lift vs. random guess</span><span style="color:${liftColour};font-weight:600">${liftLabel}</span></div>${rows||'<div style="color:var(--txt3);font-size:10px">No importances returned.</div>'}</div>`;
    };
    host.innerHTML=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px"><div style="font-weight:600;color:var(--txt)">\ud83e\udde0 ML Insights</div><div style="font-size:10px;color:var(--txt3)">(sklearn \u00b7 RandomForest feature importance per pair)</div></div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px">${withFI.map(card).join('')}</div>`;
}

function runBacktest(){
    const btn=document.getElementById('bt-run');const daysEl=document.getElementById('bt-days');const lotEl=document.getElementById('bt-lot');
    const days=Math.max(1,Math.min(60,Number((daysEl&&daysEl.value)||7)));const lot=Math.max(0.01,Number((lotEl&&lotEl.value)||0.5));
    const pairs=Array.from(document.querySelectorAll('.bt-pair')).filter(c=>c.checked).map(c=>c.value);
    if(!pairs.length){_btSetStatus('Select at least one pair to backtest.',true);return;}
    if(btn){btn.disabled=true;btn.textContent='Running\u2026';}
    _btClearResults();_btSetStatus('Backtesting '+pairs.length+' pair'+(pairs.length>1?'s':'')+' over '+days+' day'+(days>1?'s':'')+'\u2026');
    const perPair=[];const pairIndex={};const qs=new URLSearchParams({pairs:pairs.join(','),days:String(days),lot_size:String(lot)});
    if(window.__btSSE){try{window.__btSSE.close();}catch(_){}}
    const es=new EventSource('/api/backtest/stream?'+qs.toString());window.__btSSE=es;
    const cleanup=()=>{try{es.close();}catch(_){}if(btn){btn.disabled=false;btn.textContent='Run Backtest';}if(window.__btSSE===es)window.__btSSE=null;};
    es.addEventListener('hello',()=>{_btSetStatus('Connected \u2014 starting '+pairs.length+' pair'+(pairs.length>1?'s':'')+'\u2026');});
    es.addEventListener('start',(e)=>{try{JSON.parse(e.data);}catch(_){}});
    es.addEventListener('pair_start',(e)=>{try{const p=JSON.parse(e.data);_btSetStatus('['+p.symbol+'] starting\u2026');}catch(_){}});
    es.addEventListener('progress',(e)=>{
        try{const p=JSON.parse(e.data);const sym=p.symbol||'';
            if(p.type==='stage')_btSetStatus('['+sym+'] '+p.stage.replace(/_/g,' ')+'\u2026');
            else if(p.type==='bar'){const pct=p.total?Math.round((p.bar/p.total)*100):0;_btSetStatus('['+sym+'] signal gen '+p.bar+'/'+p.total+' ('+pct+'%) \u00b7 '+(p.signals_so_far||0)+' signals');}
            else if(p.type==='sim_progress'){const pct=p.total?Math.round((p.done/p.total)*100):0;_btSetStatus('['+sym+'] simulating trades '+p.done+'/'+p.total+' ('+pct+'%)');}
            else if(p.type==='signals_done')_btSetStatus('['+sym+'] '+p.signals+' signals generated');
            else if(p.type==='sim_done')_btSetStatus('['+sym+'] '+p.trades+' trades simulated');
        }catch(_){}
    });
    es.addEventListener('pair_done',(e)=>{
        try{const p=JSON.parse(e.data);const norm=_btNormalisePair(p);
            if(pairIndex[norm.symbol]!=null)perPair[pairIndex[norm.symbol]]=norm;else{pairIndex[norm.symbol]=perPair.length;perPair.push(norm);}
            _btRenderPairCards(perPair);_btRenderPnlChart(perPair,perPair.reduce((a,r)=>a+Number(r.total_pnl||0),0));_btRenderCorrelations(perPair);_btRenderAssumptions(perPair);
        }catch(_){}
    });
    es.addEventListener('done',(e)=>{
        try{const data=JSON.parse(e.data);data.per_pair=(data.per_pair||[]).map(_btNormalisePair);_btRenderResults(data);_btRenderCorrelations(data.per_pair);_btRenderMLInsights(data.per_pair);_btRenderAssumptions(data.per_pair);_btLoadMemory();const dur=data.duration_s!=null?' in '+data.duration_s+'s':'';_btSetStatus('Done'+dur+'.');}
        catch(err){_btSetStatus('Stream ended but payload was unreadable.',true);}finally{cleanup();}
    });
    es.addEventListener('error',(e)=>{
        let msg='stream error';try{if(e&&e.data)msg=(JSON.parse(e.data).error||msg);}catch(_){}
        if(es.readyState===EventSource.CLOSED){
            fetch('/api/backtest/stream?'+qs.toString(),{method:'GET'}).then(async(r)=>{
                if(r.status===404)_btSetStatus('Backtest failed: server does not know /api/backtest/stream \u2014 restart Flask so new code loads.',true);
                else if(r.status===409)_btSetStatus('Backtest failed: another backtest is still running.',true);
                else if(r.status>=500){let body='';try{body=(await r.text()).slice(0,300);}catch(_){}; _btSetStatus('Backtest failed: server error '+r.status+'. '+body,true);}
                else if(!r.ok)_btSetStatus('Backtest failed: HTTP '+r.status+'.',true);
                else _btSetStatus('Backtest failed: '+msg+' (stream closed before any events).',true);
            }).catch((err)=>{_btSetStatus('Backtest failed: cannot reach server ('+String(err).slice(0,120)+').',true);}).finally(()=>cleanup());
        } else if(es.readyState===EventSource.CONNECTING){/* transient */} else{_btSetStatus('Backtest failed: '+msg,true);cleanup();}
    });
}
