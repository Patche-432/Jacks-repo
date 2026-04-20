// ─────────────────────────────────────────────────────────────
// Fortis AI Pro — Dashboard JavaScript
// ─────────────────────────────────────────────────────────────

// ── State ────────────────────────────────────────────────────
let __userHasChosenTab = false;
let __autoTabSwitchInProgress = false;
let __autoAiLogShown = false;
let __botWasRunning = false;
let lastThoughtTs = null;
// Accumulated thoughts buffer — the /bot/ai_thoughts endpoint returns
// INCREMENTAL results (only entries newer than lastThoughtTs). We must keep
// a client-side buffer so the UI doesn't flash "No AI thoughts yet" every
// time a poll returns an empty delta.
const THOUGHTS_BUFFER_MAX = 200;
let thoughtsBuffer = [];
let equityData = [];

const marketScanCache = {};
let lastMarketScanFetchMs = 0;
const MARKET_SCAN_MIN_INTERVAL_MS = 15000;

// ── Init ─────────────────────────────────────────────────────
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

// ── Tab switching ─────────────────────────────────────────────
// `btn` is passed explicitly from inline onclick (e.g. showTab('signals', this)).
// This avoids relying on the deprecated global `event` object and works
// correctly when the tab is switched programmatically too.
function showTab(tab, btn) {
    document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    const panel = document.getElementById('tab-' + tab);
    if (panel) {
        panel.classList.add('active');
    } else {
        console.warn('showTab: no panel found for tab "' + tab + '"');
    }

    // Find the button either from the event (preferred) or by data-tab attr
    let targetBtn = btn || null;
    if (!targetBtn) {
        targetBtn = document.querySelector('.tab-btn[data-tab="' + tab + '"]');
    }
    if (targetBtn) targetBtn.classList.add('active');

    // Fire-and-forget data loads for tabs that pull server data on open
    try {
        if (tab === 'signals') fetchMarketScanBreakdown();
        if (tab === 'thoughts') {
            // Force a full refresh so the user sees latest state immediately.
            // Reset the buffer so we rebuild from the server's current view
            // rather than stacking duplicates over what we already had.
            lastThoughtTs = null;
            thoughtsBuffer = [];
            fetchThoughtsNow();
        }
        if (tab === 'positions') fetchPositions();
        if (tab === 'history') fetchHistory();
        if (tab === 'performance') fetchPerformance();
        if (tab === 'snapshot') fetchBotSnapshot();
        // 'backtest' is static HTML — no fetch required
    } catch (err) {
        console.error('showTab data load failed for "' + tab + '":', err);
    }
}

// Explicit helper so the Zero Log tab can refresh on demand
function fetchThoughtsNow() {
    const url = lastThoughtTs
        ? '/bot/ai_thoughts?limit=60&since=' + encodeURIComponent(lastThoughtTs)
        : '/bot/ai_thoughts?limit=60';
    return fetch(url)
        .then(r => r.json())
        .then(data => {
            if (!data || !data.ok) return;
            const thoughts = data.thoughts || [];
            if (thoughts.length > 0) lastThoughtTs = thoughts[thoughts.length - 1].ts;
            displayThoughts(thoughts);
        })
        .catch(err => console.error('fetchThoughtsNow error:', err));
}

function autoShowAiLogTab() {
    if (__userHasChosenTab || __autoAiLogShown) return;
    const aiLogTab = document.querySelector('.tab-btn[data-tab="thoughts"]')
        || document.querySelector('.tab-btn:nth-child(2)');
    if (!aiLogTab || aiLogTab.classList.contains('active')) {
        __autoAiLogShown = true;
        return;
    }
    __autoTabSwitchInProgress = true;
    try {
        // Call showTab directly so it doesn't depend on the click event
        showTab('thoughts', aiLogTab);
    } finally {
        __autoTabSwitchInProgress = false;
        __autoAiLogShown = true;
    }
}

// ── Symbol toggle ─────────────────────────────────────────────
function initSymbolToggle() {
    document.querySelectorAll('.watch-card').forEach(card => {
        card.classList.add('active');
        card.classList.remove('inactive');
        card.addEventListener('click', () => {
            const pair = card.getAttribute('data-pair');
            toggleSymbol(pair, card);
        });
    });
}

function toggleSymbol(pair, cardElement) {
    const nowActive = !cardElement.classList.contains('active');
    cardElement.classList.toggle('active', nowActive);
    cardElement.classList.toggle('inactive', !nowActive);
    fetch('/bot/config/symbols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: pair, enabled: nowActive }),
    })
    .then(r => r.json())
    .then(data => {
        if (!data.success) {
            cardElement.classList.toggle('active', !nowActive);
            cardElement.classList.toggle('inactive', nowActive);
            console.error('Symbol toggle failed:', data.message);
        }
    })
    .catch(err => {
        cardElement.classList.toggle('active', !nowActive);
        cardElement.classList.toggle('inactive', nowActive);
        console.error('Symbol toggle error:', err);
    });
}

// ── MT5 connection ────────────────────────────────────────────
function autoCheckMt5() {
    pollMt5Status();
}

function pollMt5Status() {
    fetch('/api/mt5/status')
        .then(r => r.json())
        .then(data => {
            updateMt5Display(data);
            setConnectButtonState(!!data.connected);
        })
        .catch(() => {});
}

function setConnectButtonState(connected) {
    const btn = document.getElementById('connect-btn');
    if (!btn) return;
    if (connected) {
        btn.style.display = 'none';
        btn.textContent = 'MT5 ✓';
        const overlay = document.getElementById('mt5-modal-overlay');
        if (overlay) overlay.style.display = 'none';
    } else {
        btn.style.display = 'inline-block';
        btn.textContent = 'Connect MT5';
    }
}

async function connectMt5() {
    const headerBtn = document.getElementById('connect-btn');
    const overlay = document.getElementById('mt5-modal-overlay');
    const statusEl = document.getElementById('mt5-modal-status');
    const modalBtn = document.getElementById('mt5-modal-btn');

    if (headerBtn && headerBtn.style.display === 'none') return;

    if (overlay) overlay.style.display = 'flex';
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.style.background = 'rgba(33,150,243,0.12)';
        statusEl.style.border = '1px solid rgba(33,150,243,0.25)';
        statusEl.style.color = 'var(--txt)';
        statusEl.textContent = 'Connecting to MT5… (make sure MT5 is open + logged in)';
    }

    const path = (document.getElementById('mt5-path') || {}).value || '';

    try {
        if (modalBtn) { modalBtn.disabled = true; modalBtn.textContent = 'Connecting…'; }
        if (headerBtn) { headerBtn.disabled = true; headerBtn.textContent = 'Connecting…'; }

        const r = await fetch('/api/mt5/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path.trim() || null }),
        });
        const data = await r.json().catch(() => ({}));

        if (!r.ok || !data.connected) {
            let msg = (data && data.error) ? data.error : 'Failed to connect to MT5';
            if (r.status === 404) msg = 'Backend /api/mt5/connect not found. Start core/server.py.';
            if (statusEl) {
                statusEl.style.background = 'rgba(244,67,54,0.12)';
                statusEl.style.border = '1px solid rgba(244,67,54,0.25)';
                statusEl.style.color = 'var(--red)';
                statusEl.textContent = msg;
            }
            setConnectButtonState(false);
            return;
        }

        updateMt5Display(data);
        setConnectButtonState(true);
    } catch (e) {
        let msg = (e && e.message) ? e.message : String(e);
        if (/Failed to fetch|NetworkError|REFUSED|refused/i.test(msg))
            msg = 'Backend not reachable. Start core/server.py and open http://127.0.0.1:5000';
        if (statusEl) {
            statusEl.style.background = 'rgba(244,67,54,0.12)';
            statusEl.style.border = '1px solid rgba(244,67,54,0.25)';
            statusEl.style.color = 'var(--red)';
            statusEl.textContent = 'Connect error: ' + msg;
        }
        setConnectButtonState(false);
    } finally {
        if (modalBtn) { modalBtn.disabled = false; modalBtn.textContent = 'Auto Connect'; }
        if (headerBtn) {
            headerBtn.disabled = false;
            if (headerBtn.style.display !== 'none') headerBtn.textContent = 'Connect MT5';
        }
    }
}

function openMt5Modal() {
    document.getElementById('mt5-modal-overlay').style.display = 'flex';
}

function closeMt5Modal() {
    document.getElementById('mt5-modal-overlay').style.display = 'none';
}

function updateMt5Display(data) {
    const dot = document.getElementById('mt5-dot');
    const statusTxt = document.getElementById('mt5-status-txt');
    const detail = document.getElementById('mt5-detail');
    if (!dot || !statusTxt || !detail) return;

    if (!data.connected) {
        dot.className = 'mt5-dot err';
        statusTxt.textContent = data.error || 'Disconnected';
        statusTxt.style.color = 'var(--red)';
        detail.style.display = 'none';
        return;
    }

    dot.className = 'mt5-dot ok';
    statusTxt.textContent = 'Connected';
    statusTxt.style.color = 'var(--green)';
    detail.style.display = 'flex';
    _setText('m-server', data.server || '—');
    _setText('m-login', (data.login || '—') + (data.account_name ? ' · ' + data.account_name : ''));
    const trEl = document.getElementById('m-trade');
    if (trEl) {
        trEl.textContent = data.trade_allowed ? 'Allowed' : 'Disabled';
        trEl.style.color = data.trade_allowed ? 'var(--green)' : 'var(--red)';
    }
    const equityEl = document.getElementById('m-equity');
    if (equityEl && data.equity !== undefined)
        equityEl.textContent = '$' + data.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const balanceEl = document.getElementById('m-balance');
    if (balanceEl && data.balance !== undefined)
        balanceEl.textContent = '$' + data.balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── Bot control ───────────────────────────────────────────────
async function startBot() {
    const btn = document.getElementById('start-btn');
    btn.disabled = true;
    btn.textContent = 'Starting…';
    try {
        const r = await fetch('/bot/start', { method: 'POST' });
        const data = await r.json();
        if (r.status === 409) {
            _setBotRunning(true);
            btn.disabled = false;
            return;
        }
        if (!r.ok || data.error) {
            alert('Error: ' + (data.error || 'Failed to start bot'));
            btn.disabled = false;
            btn.textContent = 'Start Bot';
            return;
        }
        if (data.running) _setBotRunning(true);
    } catch (e) {
        alert('Failed to start bot: ' + e.message);
    }
    btn.disabled = false;
}

async function stopBot() {
    const btn = document.getElementById('stop-btn');
    btn.disabled = true;
    btn.textContent = 'Stopping…';
    try {
        const r = await fetch('/bot/stop', { method: 'POST' });
        const data = await r.json();
        if (!r.ok && r.status !== 409) {
            alert('Error: ' + (data.error || 'Failed to stop bot'));
            btn.disabled = false;
            btn.textContent = 'Stop';
            return;
        }
        _setBotRunning(false);
    } catch (e) {
        alert('Failed to stop bot: ' + e.message);
    }
    btn.disabled = false;
}

function _setBotRunning(running) {
    _setText('status-text', running ? 'live' : 'idle');
    const dot = document.getElementById('dot');
    if (dot) dot.className = running ? 'dot live' : 'dot';
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    if (startBtn) startBtn.style.display = running ? 'none' : 'inline-block';
    if (stopBtn) stopBtn.style.display = running ? 'inline-block' : 'none';
    if (startBtn) startBtn.disabled = false;
}

function placeTestTrade() {
    fetch('/api/mt5/test-trade', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
        .then(r => r.json())
        .then(data => {
            if (data.ok) console.log('✓ Test trade placed:', data.ticket, '@', data.price);
            else console.warn('Test trade failed:', data.error);
        })
        .catch(err => console.error('Test trade error:', err));
}

// ── Sidebar controls ──────────────────────────────────────────
function toggleSym(el) {
    el.classList.toggle('active');
    _syncSymbolsInput();
}

function _syncSymbolsInput() {
    const active = Array.from(document.querySelectorAll('.sym-chip.active')).map(e => e.textContent).join(',');
    const el = document.getElementById('cfg-symbols');
    if (el) el.value = active;
}

function stepInput(id, delta, min) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = Math.max(min, parseFloat(el.value) + delta).toFixed(2);
}

function setPoll(val, el) {
    document.querySelectorAll('.poll-preset').forEach(e => e.classList.remove('active'));
    if (el) el.classList.add('active');
    const hidEl = document.getElementById('cfg-poll');
    if (hidEl) hidEl.value = val;
    const disp = document.getElementById('poll-display');
    if (disp) disp.textContent = val >= 60 ? Math.round(val / 60) + 'm' : val + 's';
}

function updateRR() {
    const atr = parseFloat((document.getElementById('cfg-atr-mult') || { value: 1.5 }).value);
    const sl  = parseFloat((document.getElementById('cfg-sl-mult')  || { value: 2.5 }).value);
    const tp  = parseFloat((document.getElementById('cfg-tp-mult')  || { value: 4.5 }).value);
    const ratio = sl > 0 ? (tp / sl) : 0;
    _setText('rr-label', '1 : ' + ratio.toFixed(1));
    const tpBar = document.getElementById('rr-tp-bar');
    if (tpBar) tpBar.style.width = Math.min(ratio * 33, 100) + '%';
}

async function applyConfig() {
    const applyBtn = document.getElementById('apply-btn');
    if (applyBtn) { applyBtn.disabled = true; applyBtn.textContent = 'Applying…'; }
    try {
        const payload = {
            symbols:       (document.getElementById('cfg-symbols')  || {}).value,
            volume:        parseFloat((document.getElementById('cfg-volume')   || { value: 0.5  }).value),
            poll_interval: parseInt(  (document.getElementById('cfg-poll')     || { value: 300  }).value, 10),
            dry_run:       !!((document.getElementById('cfg-dry')    || {}).checked),
            ai_review:     !!((document.getElementById('cfg-ai')     || { checked: true }).checked),
            auto_trade:    !!((document.getElementById('cfg-auto')   || { checked: true }).checked),
            sl_mult:       parseFloat((document.getElementById('cfg-sl-mult')  || { value: 2.5  }).value),
            tp_mult:       parseFloat((document.getElementById('cfg-tp-mult')  || { value: 4.5  }).value),
            atr_mult:      parseFloat((document.getElementById('cfg-atr-mult') || { value: 1.5  }).value),
            pc_rr:         parseFloat((document.getElementById('cfg-pc-rr')    || { value: 1.0  }).value),
            be_buffer:     parseFloat((document.getElementById('cfg-be-buf')   || { value: 1.0  }).value),
        };
        const r = await fetch('/bot/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await r.json();
        if (data.ok) {
            console.log('Config applied:', data.applied);
            if (applyBtn) {
                applyBtn.textContent = 'Applied ✓';
                setTimeout(() => {
                    applyBtn.textContent = 'Apply';
                    applyBtn.disabled = false;
                }, 1500);
            }
        } else {
            alert('Config error: ' + (data.error || 'unknown'));
            if (applyBtn) { applyBtn.disabled = false; applyBtn.textContent = 'Apply'; }
        }
    } catch (e) {
        alert('Config apply failed: ' + e.message);
        if (applyBtn) { applyBtn.disabled = false; applyBtn.textContent = 'Apply'; }
    }
}

// ── Zero Log ──────────────────────────────────────────────────
async function clearThoughts() {
    try { await fetch('/bot/thoughts/clear', { method: 'POST' }); } catch (_) {}
    thoughtsBuffer = [];
    lastThoughtTs = null;
    const container = document.getElementById('thoughts');
    if (container) container.innerHTML = '<div style="padding:40px;text-align:center;color:var(--txt3)">Log cleared</div>';
    _setText('thought-count', '0 entries');
}

// Merge incoming thoughts into the persistent buffer (dedup by ts+summary)
// and render the full accumulated list. This prevents the UI from flashing
// the empty-state message every time a polling response is empty.
function displayThoughts(newThoughts) {
    const container = document.getElementById('thoughts');
    const countEl = document.getElementById('thought-count');
    if (!container) return;

    if (Array.isArray(newThoughts) && newThoughts.length > 0) {
        const seen = new Set(
            thoughtsBuffer.map(t => (t.ts || '') + '|' + (t.summary || ''))
        );
        for (const t of newThoughts) {
            const key = (t.ts || '') + '|' + (t.summary || '');
            if (!seen.has(key)) {
                thoughtsBuffer.push(t);
                seen.add(key);
            }
        }
        // Cap buffer size to avoid DOM bloat on long sessions
        if (thoughtsBuffer.length > THOUGHTS_BUFFER_MAX) {
            thoughtsBuffer = thoughtsBuffer.slice(-THOUGHTS_BUFFER_MAX);
        }
    }

    if (thoughtsBuffer.length === 0) {
        container.innerHTML = `
            <div style="padding:40px;text-align:center;color:var(--txt3)">
                <div style="margin-bottom:10px">No entries yet — start the bot to see the live workflow.</div>
                <div style="font-size:10px;letter-spacing:.05em;color:var(--txt3)">
                    📊 STRATEGY &nbsp;→&nbsp; 🧠 AGENT &nbsp;→&nbsp; 📈 MARKET
                </div>
            </div>`;
        if (countEl) countEl.textContent = '0 entries';
        return;
    }

    if (countEl) countEl.textContent = thoughtsBuffer.length + ' entries';
    container.innerHTML = thoughtsBuffer.map(t => {
        const confHtml = (t.confidence !== null && t.confidence !== undefined)
            ? `<span style="background:var(--cyan-bg);color:var(--cyan);padding:2px 6px;border-radius:3px;margin-left:6px">${(t.confidence * 100).toFixed(0)}%</span>`
            : '';
        const actionHtml = t.action
            ? `<span style="background:var(--amber-bg);color:var(--amber);padding:2px 6px;border-radius:3px;margin-left:6px">${_esc(t.action)}</span>`
            : '';
        const symbolHtml = t.symbol
            ? `<span style="background:rgba(33,150,243,0.15);color:#64B5F6;padding:2px 7px;border-radius:3px;font-weight:600;letter-spacing:.06em;margin-left:6px">${_esc(t.symbol)}</span>`
            : '';
        // Stage-coloured pill so the strategy → agent → market workflow is
        // visually distinguishable at a glance:
        //   strategy  → blue   (signal generation)
        //   agent     → purple (Ollama review/risk)
        //   market    → green/red (broker execution)
        const stage = _zeroLogStageBadge(t.source);
        return `
        <div style="padding:12px 16px;border-bottom:1px solid var(--line);font-size:11px;border-left:3px solid ${stage.col};transition:background 0.15s" onmouseover="this.style.background='rgba(255,255,255,0.02)'" onmouseout="this.style.background=''">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:8px;flex-wrap:wrap">
                <div style="display:flex;align-items:center;gap:6px">
                    <span style="background:${stage.bg};color:${stage.col};padding:2px 7px;border-radius:3px;font-weight:600;letter-spacing:.06em;font-size:10px">${stage.icon} ${stage.label}</span>
                    <strong style="color:var(--cyan)">${_esc(t.source || '—')}</strong>
                    <span style="background:var(--bg2);color:var(--txt2);padding:2px 6px;border-radius:3px">${_esc(t.stage || '')}</span>
                    ${symbolHtml}${confHtml}${actionHtml}
                </div>
                <span style="color:var(--txt3);white-space:nowrap">${new Date(t.ts).toLocaleTimeString()}</span>
            </div>
            <div style="color:var(--txt);margin-bottom:${t.detail ? 4 : 0}px;line-height:1.4">${_esc(t.summary || '')}</div>
            ${t.detail ? `<div style="color:var(--txt3);font-size:10px;margin-top:2px;line-height:1.4">📋 ${_esc(t.detail)}</div>` : ''}
        </div>`;
    }).join('');
}

// Map a thought's `source` field to a workflow-stage badge so the Zero Log
// shows the strategy → agent → market pipeline visually.
function _zeroLogStageBadge(source) {
    const s = String(source || '').toLowerCase();
    // Order matters: 'ai_entry' / 'ai_risk' (agent) must match before 'strategy' / 'ai_pro_signal' (strategy).
    if (s === 'execution' || s.includes('order') || s.includes('market')) {
        return { label: 'MARKET',   icon: '📈', col: '#50d963', bg: 'rgba(80,217,99,0.15)' };
    }
    if (s === 'ai_entry' || s === 'ai_risk' || s.includes('agent') || s.includes('ollama')) {
        return { label: 'AGENT',    icon: '🧠', col: '#c792ea', bg: 'rgba(199,146,234,0.15)' };
    }
    if (s.includes('signal') || s.includes('strategy') || s.includes('ai_pro')) {
        return { label: 'STRATEGY', icon: '📊', col: '#5ac8fa', bg: 'rgba(90,200,250,0.15)' };
    }
    return { label: 'INFO', icon: 'ℹ', col: '#6a7280', bg: 'rgba(106,114,128,0.15)' };
}

// ── Signals tab ───────────────────────────────────────────────
function fetchMarketScanBreakdown() {
    const now = Date.now();
    if (now - lastMarketScanFetchMs < MARKET_SCAN_MIN_INTERVAL_MS) return;
    lastMarketScanFetchMs = now;

    const symbols = ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY'];
    const container = document.getElementById('signal-cards');
    if (!container) return;

    symbols.forEach(symbol => {
        fetch('/bot/signal/' + symbol)
            .then(r => r.json())
            .then(data => {
                marketScanCache[symbol] = data;
                renderMarketScanBreakdown();
                updatePairWatch(symbol, data);
            })
            .catch(err => {
                marketScanCache[symbol] = { error: 'Fetch failed' };
                console.error('Signal fetch error ' + symbol + ':', err);
            });
    });
}

function updatePairWatch(symbol, data) {
    if (!data || data.error) return;
    const statusEl = document.getElementById('status-' + symbol);
    if (!statusEl) return;
    const bias = data.bias || '—';
    const dotEl = statusEl.querySelector('.watch-dot');
    const stateEl = statusEl.querySelector('.watch-state');
    if (dotEl) dotEl.className = 'watch-dot ' + (bias === 'LONG' ? 'buy' : bias === 'SHORT' ? 'sell' : 'idle');
    if (stateEl) stateEl.textContent = bias === 'LONG' ? 'bullish' : bias === 'SHORT' ? 'bearish' : 'neutral';
}

function renderMarketScanBreakdown() {
    const container = document.getElementById('signal-cards');
    if (!container) return;
    const symbols = ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY'];
    const allCached = symbols.every(s => marketScanCache[s]);
    if (!allCached) {
        container.innerHTML = '<div class="empty-state"><div class="icon">⟳</div><div class="msg">Scanning markets…</div></div>';
        return;
    }
    let html = '<div style="display:grid;gap:16px;padding:20px">';
    symbols.forEach(symbol => {
        const sig = marketScanCache[symbol];
        if (!sig) return;
        if (sig.error) {
            html += `<div style="background:var(--bg0);border:1px solid var(--line);border-radius:8px;padding:16px"><strong style="color:var(--txt)">${symbol}</strong> <span style="color:var(--red)">— ${_esc(sig.error)}</span></div>`;
            return;
        }
        const bias = sig.bias || '—';
        const conf = (sig.confidence || 0);
        const entry = sig.entry_price ? sig.entry_price.toFixed(5) : '—';
        const sl = sig.sl_price ? sig.sl_price.toFixed(5) : '—';
        const tp = sig.tp_price ? sig.tp_price.toFixed(5) : '—';
        const biasColor = bias === 'LONG' ? 'var(--green)' : bias === 'SHORT' ? 'var(--red)' : 'var(--txt2)';
        const confColor = conf >= 70 ? 'var(--green)' : conf >= 50 ? 'var(--amber)' : 'var(--red)';

        // Pip Risk: distance from entry to SL in pips
        // JPY pairs use 2 decimal places (1 pip = 0.01), others use 4 (1 pip = 0.0001)
        let pipRiskHtml = '—';
        if (sig.entry_price && sig.sl_price) {
            const pipSize = symbol.includes('JPY') ? 0.01 : 0.0001;
            const slPips = Math.abs(sig.entry_price - sig.sl_price) / pipSize;
            const tpPips = Math.abs((sig.tp_price || sig.entry_price) - sig.entry_price) / pipSize;
            const slPipsStr = slPips.toFixed(1);
            const tpPipsStr = tpPips.toFixed(1);
            pipRiskHtml = `
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="color:var(--txt3);font-size:10px">SL</span>
                    <span style="color:var(--red);font-family:var(--mono);font-weight:600">${slPipsStr} pips</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px">
                    <span style="color:var(--txt3);font-size:10px">TP</span>
                    <span style="color:var(--green);font-family:var(--mono);font-weight:600">${tpPipsStr} pips</span>
                </div>`;
        }

        // Trend Strength: derived from confidence score
        let strengthLabel, strengthColor, strengthBg;
        if (conf >= 75) {
            strengthLabel = 'STRONG'; strengthColor = 'var(--green)'; strengthBg = 'rgba(0,255,136,0.1)';
        } else if (conf >= 50) {
            strengthLabel = 'MODERATE'; strengthColor = 'var(--amber)'; strengthBg = 'rgba(255,170,0,0.1)';
        } else {
            strengthLabel = 'WEAK'; strengthColor = 'var(--red)'; strengthBg = 'rgba(255,80,80,0.1)';
        }

        html += `
        <div style="background:var(--bg0);border:1px solid var(--line);border-radius:8px;padding:16px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid var(--line)">
                <div><span style="font-size:13px;font-weight:600;color:var(--txt)">${symbol}</span><span style="font-size:10px;color:var(--txt3);margin-left:8px;letter-spacing:.08em">MARKET SCAN</span></div>
                <div style="display:flex;gap:8px;align-items:center">
                    <span style="font-size:11px;background:${biasColor};color:#000;padding:4px 8px;border-radius:4px;font-weight:600">${bias}</span>
                    <span style="font-size:11px;background:${confColor};color:#000;padding:4px 8px;border-radius:4px;font-weight:600">${conf}% conf</span>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:11px;margin-bottom:12px">
                <div><div style="color:var(--txt3);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px">Entry</div><div style="font-family:var(--mono);color:var(--txt);font-size:12px;font-weight:500">${entry}</div></div>
                <div><div style="color:var(--txt3);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px">Stop Loss</div><div style="font-family:var(--mono);color:var(--red);font-size:12px;font-weight:500">${sl}</div></div>
                <div><div style="color:var(--txt3);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px">Take Profit</div><div style="font-family:var(--mono);color:var(--green);font-size:12px;font-weight:500">${tp}</div></div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
                <div style="background:var(--bg1);border-left:3px solid var(--cyan);border-radius:4px;padding:10px">
                    <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Pip Risk / Reward</div>
                    ${pipRiskHtml}
                </div>
                <div style="background:${strengthBg};border:1px solid ${strengthColor};border-radius:4px;padding:10px;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:4px">
                    <div style="font-size:9px;color:var(--txt3);text-transform:uppercase;letter-spacing:.08em">Trend Strength</div>
                    <div style="font-size:15px;font-weight:700;color:${strengthColor};letter-spacing:.06em">${strengthLabel}</div>
                </div>
            </div>
            <div style="display:grid;gap:8px;font-size:10px">${renderSignalBreakdown(sig)}</div>
        </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderSignalBreakdown(sig) {
    let html = '';

    // Environment — now carries the real ENV 1/2/3/4 label from AI_Pro
    if (sig.environment && sig.environment !== 'No active environment') {
        // Colour the ENV badge by number
        const envNum = (sig.environment.match(/ENV\s*(\d)/i) || [])[1];
        const envColors = { '1':'var(--green)', '2':'var(--red)', '3':'var(--cyan)', '4':'var(--amber)' };
        const envColor  = envColors[envNum] || 'var(--cyan)';
        html += `<div style="background:var(--bg0);padding:8px;border-radius:4px;border-left:2px solid ${envColor}">
            <span style="color:var(--txt3)">Environment:</span>
            <span style="color:var(--txt);margin-left:8px;font-weight:500">${_esc(sig.environment)}</span>
        </div>`;
    } else if (sig.environment) {
        html += `<div style="background:var(--bg0);padding:8px;border-radius:4px;border-left:2px solid var(--line)">
            <span style="color:var(--txt3)">Environment:</span>
            <span style="color:var(--txt2);margin-left:8px">${_esc(sig.environment)}</span>
        </div>`;
    }

    // CHoCH status
    if (sig.choch_status && sig.choch_status !== '—') {
        const chochColor = sig.choch_status.includes('✓') ? 'var(--green)' : 'var(--amber)';
        html += `<div style="background:var(--bg0);padding:8px;border-radius:4px;border-left:2px solid ${chochColor}">
            <span style="color:var(--txt3)">CHoCH:</span>
            <span style="color:var(--txt);margin-left:8px;font-weight:500">${_esc(sig.choch_status)}</span>
        </div>`;
    }

    // PDH / PDL key levels
    if (sig.level_interaction && sig.level_interaction !== '—') {
        html += `<div style="background:var(--bg0);padding:8px;border-radius:4px;border-left:2px solid var(--purple)">
            <span style="color:var(--txt3)">Key Levels:</span>
            <span style="color:var(--txt);margin-left:8px;font-family:var(--mono);font-size:10px">${_esc(sig.level_interaction)}</span>
        </div>`;
    }

    // Scan age
    if (sig.ts) {
        const age = Math.round((Date.now() - new Date(sig.ts).getTime()) / 1000);
        const ageStr = age < 60 ? age + 's ago' : Math.round(age / 60) + 'm ago';
        html += `<div style="background:var(--bg0);padding:8px;border-radius:4px;border-left:2px solid var(--line)">
            <span style="color:var(--txt3)">Last Scan:</span>
            <span style="color:var(--txt2);margin-left:8px;font-family:var(--mono)">${ageStr}</span>
        </div>`;
    }

    return html;
}

// ── Positions tab ─────────────────────────────────────────────
function fetchPositions() {
    fetch('/bot/positions')
        .then(r => r.json())
        .then(data => renderPositions(data.positions || []))
        .catch(() => renderPositions([]));
}

function renderPositions(positions) {
    const container = document.getElementById('positions-content');
    if (!container) return;
    if (!positions || positions.length === 0) {
        container.innerHTML = '<div class="pos-empty">No open positions</div>';
        return;
    }
    let totalProfit = 0;
    let html = '<div style="padding:16px"><div style="display:grid;gap:12px">';
    positions.forEach(pos => {
        totalProfit += pos.profit || 0;
        const typeColor = pos.type === 'BUY' ? 'var(--green)' : 'var(--red)';
        const profitColor = (pos.profit || 0) >= 0 ? 'var(--green)' : 'var(--red)';
        const profitSign = (pos.profit || 0) >= 0 ? '+' : '';
        html += `
        <div style="background:var(--bg0);border:1px solid var(--line);border-radius:8px;padding:14px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:13px;font-weight:600;color:var(--txt)">${_esc(pos.symbol)}</span>
                    <span style="background:${typeColor};color:#000;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:600">${pos.type}</span>
                    <span style="color:var(--txt2);font-size:10px">Vol: ${pos.volume}</span>
                </div>
                <div style="text-align:right">
                    <div style="font-size:14px;font-weight:600;color:${profitColor}">${profitSign}$${Math.abs(pos.profit || 0).toFixed(2)}</div>
                    <div style="font-size:10px;color:var(--txt3)">#${pos.ticket}</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:10px">
                <div><div style="color:var(--txt3);margin-bottom:2px">Open Price</div><div style="font-family:var(--mono);color:var(--txt)">${pos.open_price}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Current</div><div style="font-family:var(--mono);color:var(--txt)">${pos.current_price}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Swap</div><div style="font-family:var(--mono);color:var(--txt2)">${pos.swap || 0}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Stop Loss</div><div style="font-family:var(--mono);color:var(--red)">${pos.sl || '—'}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Take Profit</div><div style="font-family:var(--mono);color:var(--green)">${pos.tp || '—'}</div></div>
                <div><div style="color:var(--txt3);margin-bottom:2px">Opened</div><div style="color:var(--txt2)">${new Date(pos.open_time).toLocaleTimeString()}</div></div>
            </div>
        </div>`;
    });
    const totalColor = totalProfit >= 0 ? 'var(--green)' : 'var(--red)';
    const totalSign = totalProfit >= 0 ? '+' : '';
    html += `</div>
        <div style="margin-top:12px;padding:12px 16px;background:var(--bg1);border:1px solid var(--line);border-radius:6px;display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:11px;color:var(--txt2)">${positions.length} open position${positions.length !== 1 ? 's' : ''}</span>
            <span style="font-size:13px;font-weight:600;color:${totalColor}">Floating P&L: ${totalSign}$${Math.abs(totalProfit).toFixed(2)}</span>
        </div>
    </div>`;
    container.innerHTML = html;
}

// ── History tab ───────────────────────────────────────────────
function fetchHistory() {
    fetch('/bot/history')
        .then(r => r.json())
        .then(data => renderHistory(data.trades || []))
        .catch(() => renderHistory([]));
}

function renderHistory(trades) {
    const listEl = document.getElementById('history-list');
    const summaryEl = document.getElementById('history-summary');
    if (!listEl) return;
    if (!trades || trades.length === 0) {
        if (summaryEl) summaryEl.innerHTML = '';
        listEl.innerHTML = '<div class="pos-empty">No trades recorded yet</div>';
        return;
    }
    const totalPL = trades.reduce((s, t) => s + (t.profit || 0), 0);
    const wins = trades.filter(t => (t.profit || 0) > 0);
    const winRate = trades.length > 0 ? (wins.length / trades.length * 100).toFixed(1) : '0.0';
    if (summaryEl) {
        const plColor = totalPL >= 0 ? 'var(--green)' : 'var(--red)';
        summaryEl.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:16px 16px 0">
            <div style="background:var(--bg1);border:1px solid var(--line);border-radius:6px;padding:12px;text-align:center">
                <div style="font-size:10px;color:var(--txt3);text-transform:uppercase;margin-bottom:4px">Trades (30d)</div>
                <div style="font-size:18px;font-weight:600;color:var(--txt)">${trades.length}</div>
            </div>
            <div style="background:var(--bg1);border:1px solid var(--line);border-radius:6px;padding:12px;text-align:center">
                <div style="font-size:10px;color:var(--txt3);text-transform:uppercase;margin-bottom:4px">Win Rate</div>
                <div style="font-size:18px;font-weight:600;color:var(--cyan)">${winRate}%</div>
            </div>
            <div style="background:var(--bg1);border:1px solid var(--line);border-radius:6px;padding:12px;text-align:center">
                <div style="font-size:10px;color:var(--txt3);text-transform:uppercase;margin-bottom:4px">Total P&L</div>
                <div style="font-size:18px;font-weight:600;color:${plColor}">${totalPL >= 0 ? '+' : ''}$${totalPL.toFixed(2)}</div>
            </div>
        </div>`;
    }
    let html = '<div style="padding:12px 16px;display:grid;gap:6px">';
    trades.forEach(t => {
        const profitColor = (t.profit || 0) >= 0 ? 'var(--green)' : 'var(--red)';
        const profitSign = (t.profit || 0) >= 0 ? '+' : '';
        html += `
        <div style="display:grid;grid-template-columns:80px 50px 60px 1fr 80px 80px;gap:8px;align-items:center;padding:8px 12px;background:var(--bg1);border-radius:6px;font-size:11px;border:1px solid var(--line)">
            <span style="font-weight:600;color:var(--txt)">${_esc(t.symbol)}</span>
            <span style="background:${(t.type === 'BUY') ? 'var(--green)' : 'var(--red)'};color:#000;padding:2px 6px;border-radius:3px;font-size:9px;font-weight:600;text-align:center">${t.type}</span>
            <span style="color:var(--txt2)">${t.volume} lot</span>
            <span style="font-family:var(--mono);color:var(--txt)">${t.price}</span>
            <span style="font-family:var(--mono);color:${profitColor};font-weight:600">${profitSign}$${Math.abs(t.profit || 0).toFixed(2)}</span>
            <span style="color:var(--txt3)">${new Date(t.time).toLocaleDateString()}</span>
        </div>`;
    });
    html += '</div>';
    listEl.innerHTML = html;
}

// ── Performance tab ───────────────────────────────────────────
function fetchPerformance() {
    fetch('/bot/performance')
        .then(r => r.json())
        .then(data => {
            renderKPIs(data.kpis || {});
            // Always call renderEquityCurve, even with [] — it draws an
            // informative empty-state message instead of leaving a blank box.
            equityData = (data.equity_curve && data.equity_curve.length) ? data.equity_curve : [];
            renderEquityCurve(equityData, data.error || null);
        })
        .catch(() => {
            renderKPIs({});
            renderEquityCurve([], 'Failed to reach /bot/performance');
        });
}

function renderKPIs(kpis) {
    _setKPI('kpi-winrate',      (kpis.win_rate       || 0).toFixed(1) + '%');
    // Treat the 999 sentinel from the backtester/server as infinity (no losing trades).
    const pfVal = Number(kpis.profit_factor || 0);
    _setKPI('kpi-profitfactor', pfVal >= 999 ? '∞' : pfVal.toFixed(2));
    _setKPI('kpi-trades',       kpis.total_trades     || 0);
    _setKPI('kpi-return',       (kpis.equity_return  || 0).toFixed(2) + '%');
    _setKPI('kpi-drawdown',     (kpis.max_drawdown   || 0).toFixed(2) + '%');
    _setKPI('kpi-sharpe',       (kpis.sharpe         || 0).toFixed(2));
    _setKPI('kpi-avgratio',     (kpis.avg_win_loss_ratio || 0).toFixed(2));
    _setKPI('kpi-currentdd',    (kpis.current_drawdown   || 0).toFixed(2) + '%');
    _setKPI('kpi-recovery',     (kpis.recovery_factor    || 0).toFixed(2));
    _setKPI('kpi-sortino',      (kpis.sortino        || 0).toFixed(2));
}

function renderEquityCurve(data, errorMsg) {
    const canvas = document.getElementById('equityCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    // Match CSS height (300px in dashboard.css). Fallback only if the canvas
    // is in a hidden tab and getBoundingClientRect returns 0.
    const cssW = rect.width  || 600;
    const cssH = rect.height || 300;
    canvas.width  = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    // Reset any prior transform before scaling so DPR changes (e.g. moving
    // window between monitors) don't compound the scale.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    const W = cssW;
    const H = cssH;

    // Background — always paint so the panel never looks broken.
    ctx.fillStyle = '#0a0e14';
    ctx.fillRect(0, 0, W, H);

    // Empty / error state — draw a centred message and bail out.
    if (errorMsg || !data || data.length < 1) {
        ctx.fillStyle = errorMsg ? '#ff6b6b' : '#6a7280';
        ctx.font = '12px "IBM Plex Mono",monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const msg = errorMsg
            ? `⚠ ${errorMsg}`
            : 'No closed trades in the last 90 days — equity curve will appear once trades start closing.';
        // Soft word-wrap at ~70 chars
        const lines = _wrapText(msg, 70);
        const lineH = 16;
        const startY = H / 2 - ((lines.length - 1) * lineH) / 2;
        lines.forEach((ln, i) => ctx.fillText(ln, W / 2, startY + i * lineH));
        return;
    }

    // Single-trade case — render a flat baseline + label so the user knows
    // the panel is alive but there isn't enough data for a curve yet.
    if (data.length < 2) {
        ctx.fillStyle = '#6a7280';
        ctx.font = '12px "IBM Plex Mono",monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`Only 1 closed trade so far (P&L $${Number(data[0]).toFixed(2)}).`, W / 2, H / 2);
        return;
    }

    // Anchor the series at 0 so the chart visually starts at break-even
    // rather than the first trade's P&L. Avoid duplicating a leading 0.
    const series = (Number(data[0]) === 0) ? data.slice() : [0, ...data];

    const pad = { top: 20, right: 20, bottom: 28, left: 56 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top  - pad.bottom;
    const min = Math.min(...series);
    const max = Math.max(...series);
    const range = (max - min) || 1;

    // Gridlines + Y-axis labels
    ctx.strokeStyle = 'rgba(42,58,82,0.6)';
    ctx.lineWidth = 0.5;
    ctx.font = '9px "IBM Plex Mono",monospace';
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH * i / 4);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        const val = max - (range * i / 4);
        ctx.fillStyle = '#6a7280';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText((val >= 0 ? '+$' : '-$') + Math.abs(val).toFixed(0), pad.left - 6, y);
    }

    const lastVal = series[series.length - 1];
    const lineCol = lastVal >= 0 ? '#50d963' : '#ff6b6b';
    const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    grad.addColorStop(0, lastVal >= 0 ? 'rgba(80,217,99,0.35)' : 'rgba(255,107,107,0.35)');
    grad.addColorStop(1, 'rgba(10,14,20,0)');

    const xStep = plotW / (series.length - 1);
    const xAt = i => pad.left + i * xStep;
    const yAt = v => pad.top + plotH - ((v - min) / range) * plotH;

    // Filled area under the curve
    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(series[0]));
    for (let i = 1; i < series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
    ctx.lineTo(xAt(series.length - 1), H - pad.bottom);
    ctx.lineTo(xAt(0), H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Curve line
    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(series[0]));
    for (let i = 1; i < series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
    ctx.strokeStyle = lineCol;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Zero baseline (only if it's actually inside the visible range)
    if (min < 0 && max > 0) {
        const y0 = yAt(0);
        ctx.beginPath(); ctx.moveTo(pad.left, y0); ctx.lineTo(W - pad.right, y0);
        ctx.strokeStyle = 'rgba(160,170,184,0.3)'; ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    }

    // X-axis labels
    ctx.fillStyle = '#6a7280';
    ctx.font = '9px "IBM Plex Mono",monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('start', pad.left, H - pad.bottom + 8);
    ctx.fillText('mid',   pad.left + plotW / 2, H - pad.bottom + 8);
    ctx.fillText('now',   W - pad.right, H - pad.bottom + 8);

    // Last-value pill in the top-right
    const lvLabel = (lastVal >= 0 ? '+$' : '-$') + Math.abs(lastVal).toFixed(2);
    ctx.fillStyle = lineCol;
    ctx.font = 'bold 11px "IBM Plex Mono",monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(lvLabel, W - pad.right, pad.top - 14);
}

// Simple word-wrap helper for short status / error messages drawn into the
// equity-curve canvas.
function _wrapText(text, maxChars) {
    const words = String(text).split(/\s+/);
    const lines = [];
    let cur = '';
    for (const w of words) {
        if (!cur) { cur = w; continue; }
        if ((cur + ' ' + w).length > maxChars) { lines.push(cur); cur = w; }
        else cur += ' ' + w;
    }
    if (cur) lines.push(cur);
    return lines;
}

// Re-render the equity curve when the window resizes, so it stays sharp.
// Debounced to avoid thrashing during drag-resize.
let _equityResizeTimer = null;
window.addEventListener('resize', () => {
    if (_equityResizeTimer) clearTimeout(_equityResizeTimer);
    _equityResizeTimer = setTimeout(() => {
        const perfTab = document.getElementById('tab-performance');
        if (perfTab && perfTab.classList.contains('active')) {
            renderEquityCurve(equityData || []);
        }
    }, 150);
});

// ── Market Watch (Snapshot) tab ───────────────────────────────
function fetchBotSnapshot() {
    fetch('/bot/status')
        .then(r => r.json())
        .then(data => {
            const bot = data.bot || {};
            _setText('snap-bot-status', bot.running ? 'Running ✓' : 'Idle');
            const snapStatus = document.getElementById('snap-bot-status');
            if (snapStatus) snapStatus.style.color = bot.running ? 'var(--green)' : 'var(--txt2)';
            _setText('snap-bias', bot.market_bias || '—');
            const conf = bot.confidence !== undefined ? (bot.confidence * 100).toFixed(0) + '%' : '—';
            _setText('snap-confidence', conf);
            _setText('snap-last-signal', bot.last_signal || 'None');
            _setText('snap-session-summary', bot.session_summary || 'Waiting for bot to connect…');
            const tradeDetails = document.getElementById('snap-trade-details');
            if (tradeDetails) {
                const trades = bot.open_trades || [];
                if (trades.length > 0) {
                    tradeDetails.innerHTML = trades.map(t => `
                    <div style="padding:10px;background:var(--bg1);border-radius:6px;border:1px solid var(--line)">
                        <div style="font-weight:600;color:var(--cyan);margin-bottom:4px">${_esc(t.symbol)} ${_esc(t.direction || '')}</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:10px">
                            <div>Entry: <span style="color:var(--txt)">${t.entry_price || '—'}</span></div>
                            <div>SL: <span style="color:var(--red)">${t.stop_loss || '—'}</span></div>
                            <div>TP: <span style="color:var(--green)">${t.take_profit || '—'}</span></div>
                            <div>P&L: <span style="color:${(t.profit || 0) > 0 ? 'var(--green)' : 'var(--red)'}">
                                ${(t.profit || 0) > 0 ? '+' : ''}${typeof t.profit === 'number' ? t.profit.toFixed(2) : (t.profit || 0)}</span></div>
                        </div>
                    </div>`).join('');
                } else {
                    tradeDetails.innerHTML = '<div style="padding:12px;background:var(--bg1);border-radius:6px;color:var(--txt2);text-align:center">No active trades</div>';
                }
            }
        })
        .catch(() => {});
}

// ── Portfolio watch updater from positions ────────────────────
function updatePortfolioWatch(positions) {
    const pairs = ['GBPJPY', 'EURJPY', 'GBPUSD', 'EURUSD'];
    const pairStats = {};
    pairs.forEach(p => { pairStats[p] = { trades: 0, pnl: 0, hasTrade: false }; });
    (positions || []).forEach(pos => {
        if (pairStats[pos.symbol]) {
            pairStats[pos.symbol].trades++;
            pairStats[pos.symbol].pnl += pos.profit || 0;
            pairStats[pos.symbol].hasTrade = true;
        }
    });
    pairs.forEach(pair => {
        const stats = pairStats[pair];
        const statusEl = document.getElementById('status-' + pair);
        if (statusEl) {
            const dotEl = statusEl.querySelector('.watch-dot');
            const stateEl = statusEl.querySelector('.watch-state');
            if (dotEl) dotEl.className = 'watch-dot ' + (stats.hasTrade ? 'trading' : 'idle');
            if (stateEl) stateEl.textContent = stats.hasTrade ? 'trading' : 'idle';
        }
        const tradesEl = document.getElementById('trades-' + pair);
        if (tradesEl) tradesEl.textContent = stats.trades;
        const pnlEl = document.getElementById('pnl-' + pair);
        if (pnlEl) {
            pnlEl.textContent = stats.pnl >= 0 ? '+$' + stats.pnl.toFixed(0) : '-$' + Math.abs(stats.pnl).toFixed(0);
            pnlEl.style.color = stats.pnl >= 0 ? 'var(--green)' : 'var(--red)';
        }
    });
}

// ── Polling loops ─────────────────────────────────────────────
setInterval(() => {
    fetch('/api/mt5/status')
        .then(r => r.json())
        .then(data => {
            updateMt5Display(data);
            setConnectButtonState(!!data.connected);
            fetch('/bot/status')
                .then(b => b.json())
                .then(botData => {
                    const bot = botData.bot || {};
                    const runningNow = !!(bot.running && data.connected);
                    if (runningNow && !__botWasRunning) autoShowAiLogTab();
                    if (!runningNow && !__userHasChosenTab) __autoAiLogShown = false;
                    __botWasRunning = runningNow;
                    _setBotRunning(runningNow);
                    const signalsTab = document.getElementById('tab-signals');
                    if (signalsTab && signalsTab.classList.contains('active') && data.connected)
                        fetchMarketScanBreakdown();
                })
                .catch(() => {});
        })
        .catch(() => {});
}, 2000);

setInterval(() => {
    const url = lastThoughtTs
        ? '/bot/ai_thoughts?limit=60&since=' + encodeURIComponent(lastThoughtTs)
        : '/bot/ai_thoughts?limit=60';
    fetch(url)
        .then(r => r.json())
        .then(data => {
            if (!data.ok) return;
            const thoughts = data.thoughts || [];
            if (thoughts.length > 0) lastThoughtTs = thoughts[thoughts.length - 1].ts;
            displayThoughts(thoughts);
        })
        .catch(() => {});
}, 2000);

setInterval(() => {
    fetch('/bot/positions')
        .then(r => r.json())
        .then(data => updatePortfolioWatch(data.positions || []))
        .catch(() => {});
    const posTab = document.getElementById('tab-positions');
    if (posTab && posTab.classList.contains('active')) fetchPositions();
}, 5000);

setInterval(() => {
    const snapTab = document.getElementById('tab-snapshot');
    if (snapTab && snapTab.classList.contains('active')) fetchBotSnapshot();
}, 3000);

// Refresh the History tab while it's the active panel (closed trades over time)
setInterval(() => {
    const histTab = document.getElementById('tab-history');
    if (histTab && histTab.classList.contains('active')) fetchHistory();
}, 10000);

// Refresh the Performance tab while it's the active panel (KPIs + equity curve)
setInterval(() => {
    const perfTab = document.getElementById('tab-performance');
    if (perfTab && perfTab.classList.contains('active')) fetchPerformance();
}, 10000);

// ── Utilities ─────────────────────────────────────────────────
function _setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function _setKPI(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function _esc(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Backtest tab ────────────────────────────────────────────────────────────
//
// Posts to /api/backtest/run with the inputs from the tab and renders the
// aggregate KPIs and per-pair breakdown from the response.

function _btSetStatus(text, isError) {
    const el = document.getElementById('bt-status');
    if (!el) return;
    el.textContent = text || '';
    el.style.color = isError ? 'var(--red, #ff6b6b)' : 'var(--txt3)';
}

function _btClearResults() {
    ['bt-total-trades', 'bt-win-rate', 'bt-total-pnl', 'bt-avg-pnl', 'bt-max-dd']
        .forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '—'; });
    const body = document.getElementById('bt-pairs-body');
    if (body) body.innerHTML = '<div style="color:var(--txt3);padding:8px;grid-column:1/-1">Running…</div>';
    const chart = document.getElementById('bt-pnl-chart');
    if (chart) chart.innerHTML = '<div style="color:var(--txt3);padding:6px 0">Running…</div>';
}

function _btMoney(v) {
    const n = Number(v || 0);
    const s = n >= 0 ? '+' : '-';
    return s + '$' + Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function _btMoneyShort(v) {
    const n = Number(v || 0);
    const abs = Math.abs(n);
    const s = n >= 0 ? '+' : '-';
    if (abs >= 1000) return s + '$' + (abs / 1000).toFixed(1) + 'k';
    return s + '$' + abs.toFixed(0);
}

function _btRenderPnlChart(per, totalPnl) {
    const chart = document.getElementById('bt-pnl-chart');
    if (!chart) return;
    const rows = (per || []).filter(r => !r.error);
    if (!rows.length) {
        chart.innerHTML = '<div style="color:var(--txt3);padding:6px 0">No per-pair P&L to visualise.</div>';
        return;
    }
    const maxAbs = Math.max(1, ...rows.map(r => Math.abs(Number(r.total_pnl || 0))));
    const totalAbs = rows.reduce((a, r) => a + Math.abs(Number(r.total_pnl || 0)), 0) || 1;

    chart.innerHTML = rows.map(r => {
        const pnl = Number(r.total_pnl || 0);
        const pct = (Math.abs(pnl) / maxAbs) * 100;
        const share = (Math.abs(pnl) / totalAbs) * 100;
        const colour = pnl >= 0 ? 'var(--green,#2ecc71)' : 'var(--red,#ff6b6b)';
        const bg = pnl >= 0 ? 'rgba(46,204,113,0.15)' : 'rgba(255,107,107,0.15)';
        return `
          <div style="display:grid;grid-template-columns:80px 1fr 110px 70px;align-items:center;gap:10px">
            <div style="color:var(--txt2)">${_esc(r.symbol)}</div>
            <div style="position:relative;height:16px;background:var(--bg1);border-radius:4px;overflow:hidden">
              <div style="position:absolute;left:0;top:0;bottom:0;width:${pct.toFixed(1)}%;background:${bg};border-right:2px solid ${colour};transition:width .25s"></div>
            </div>
            <div style="text-align:right;color:${colour}">${_esc(r.pnl_label || _btMoney(pnl))}</div>
            <div style="text-align:right;color:var(--txt3)">${share.toFixed(1)}%</div>
          </div>`;
    }).join('');
}

function _btRenderPairCards(per) {
    const body = document.getElementById('bt-pairs-body');
    if (!body) return;
    const rows = per || [];
    if (!rows.length) {
        body.innerHTML = '<div style="color:var(--txt3);padding:8px">No per-pair results returned.</div>';
        return;
    }

    body.innerHTML = rows.map(r => {
        if (r.error) {
            return `
              <div style="background:var(--bg1);padding:12px;border-radius:6px;border-left:3px solid var(--red,#ff6b6b)">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                  <div style="font-weight:600;color:var(--txt)">${_esc(r.symbol)}</div>
                  <div style="font-size:10px;color:var(--red,#ff6b6b)">FAILED</div>
                </div>
                <div style="font-size:10px;color:var(--txt3)">${_esc(r.error)}</div>
              </div>`;
        }

        const pnl      = Number(r.total_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red,#ff6b6b)';
        const wrPct    = Math.max(0, Math.min(100, Number(r.win_rate || 0) * 100));
        const wrLabel  = r.win_rate_label || (wrPct.toFixed(1) + '%');
        const pnlLabel = r.pnl_label || _btMoney(pnl);
        const pf       = Number(r.profit_factor || 0);
        const pfLabel  = pf > 0 ? pf.toFixed(2) + 'x' : '—';
        const rr       = Number(r.avg_rr || 0);
        const rrLabel  = rr > 0 ? rr.toFixed(2) : '—';
        const trades   = Number(r.trades || 0);
        const wins     = Number(r.wins || 0);
        const losses   = Number(r.losses || 0);
        const mdd      = Number(r.max_drawdown || 0);
        const mddLabel = mdd > 0 ? ('-$' + mdd.toFixed(0)) : '—';
        const barColor = wrPct >= 55 ? 'var(--green)' : (wrPct >= 45 ? 'var(--amber,#f5a623)' : 'var(--red,#ff6b6b)');

        return `
          <div style="background:var(--bg1);padding:12px;border-radius:6px;border-left:3px solid ${pnlColor}">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
              <div style="font-weight:600;color:var(--txt);font-size:12px">${_esc(r.symbol)}</div>
              <div style="color:${pnlColor};font-weight:600">${_esc(pnlLabel)}</div>
            </div>

            <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--txt3);margin-bottom:4px">
              <span>Win Rate</span>
              <span style="color:var(--txt2)">${_esc(wrLabel)} · ${wins}W / ${losses}L</span>
            </div>
            <div style="height:6px;background:var(--bg0);border-radius:3px;overflow:hidden;margin-bottom:10px">
              <div style="height:100%;width:${wrPct.toFixed(1)}%;background:${barColor};transition:width .25s"></div>
            </div>

            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:10px;color:var(--txt3)">
              <div>
                <div>Trades</div>
                <div style="color:var(--txt);font-size:11px">${trades.toLocaleString()}</div>
              </div>
              <div>
                <div>Profit Factor</div>
                <div style="color:var(--txt);font-size:11px">${_esc(pfLabel)}</div>
              </div>
              <div>
                <div>Avg R:R</div>
                <div style="color:var(--txt);font-size:11px">${_esc(rrLabel)}</div>
              </div>
              <div title="Peak-to-trough drawdown on this pair's equity curve">
                <div>Max DD</div>
                <div style="color:${mdd > 0 ? 'var(--red,#ff6b6b)' : 'var(--txt)'};font-size:11px">${_esc(mddLabel)}</div>
              </div>
            </div>
          </div>`;
    }).join('');
}

function _btRenderResults(data) {
    const agg = data.aggregate || {};
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    set('bt-total-trades', (agg.total_trades ?? 0).toLocaleString());
    set('bt-win-rate',     agg.win_rate_label  || '0.0%');
    set('bt-total-pnl',    agg.total_pnl_label || '$0.00');
    set('bt-avg-pnl',      agg.avg_pnl_label   || '$0.00');

    // Max drawdown — peak-to-trough on the merged cross-pair equity
    // curve. Shown as a loss-style "-$X" with a % -of-peak deco line.
    const mddVal   = Number(agg.max_drawdown || 0);
    const mddLabel = agg.max_drawdown_label || (mddVal > 0 ? ('-$' + mddVal.toFixed(2)) : '$0.00');
    set('bt-max-dd', mddLabel);
    const ddEl = document.getElementById('bt-max-dd');
    if (ddEl) ddEl.style.color = mddVal > 0 ? 'var(--red,#ff6b6b)' : '';
    const mddPct = Number(agg.max_drawdown_pct || 0);
    set('bt-max-dd-deco', mddPct > 0
        ? `${mddPct.toFixed(1)}% of peak`
        : 'peak to trough');

    // KPI deco lines: quick context under the numbers
    const wins = Number(agg.wins || 0), losses = Number(agg.losses || 0);
    set('bt-total-trades-deco', wins || losses ? `${wins}W · ${losses}L` : 'backtest');
    const pairCount = (data.per_pair || []).length;
    set('bt-win-rate-deco', pairCount ? `across ${pairCount} pair${pairCount > 1 ? 's' : ''}` : 'aggregate');

    const lot = Number(data.lot_size || 0).toFixed(2);
    const lotEl = document.getElementById('bt-lot-label');
    if (lotEl) lotEl.textContent = `at ${lot} lot`;
    const lotEl2 = document.getElementById('bt-lot-label-2');
    if (lotEl2) lotEl2.textContent = `(${lot} lot)`;

    const sub = document.getElementById('bt-subtitle');
    if (sub) {
        const period = data.period ? ` — ${data.period}` : '';
        // Show raw-strategy mode explicitly — these numbers measure signal
        // quality, not the management overlay the live bot applies.
        sub.textContent = `${data.days || 7}-day raw-strategy validation (${lot} lot, single SL/TP exit)${period}`;
    }

    // Summary banner
    const banner = document.getElementById('bt-summary-banner');
    if (banner) {
        banner.style.display = 'flex';
        set('bt-last-run',     new Date().toLocaleString());
        set('bt-period',       data.period || '—');
        set('bt-duration',     (data.duration_s != null) ? `${data.duration_s}s` : '—');
        set('bt-lot-summary',  lot);
    }

    _btRenderPnlChart(data.per_pair || [], Number(agg.total_pnl || 0));
    _btRenderPairCards(data.per_pair || []);
}

// Normalise a per-pair payload coming off the wire so the existing card
// renderer finds the field names it expects.
function _btNormalisePair(r) {
    if (!r) return r;
    const pnl = Number(r.total_pnl || 0);
    return {
        symbol:        r.symbol,
        error:         r.error,
        total_pnl:     pnl,
        pnl_label:     _btMoney(pnl),
        win_rate:      Number(r.win_rate || 0),
        win_rate_label: ((Number(r.win_rate || 0)) * 100).toFixed(1) + '%',
        trades:        Number(r.total_trades || 0),
        wins:          Number(r.wins || 0),
        losses:        Number(r.losses || 0),
        profit_factor: Number(r.profit_factor || 0),
        avg_rr:        Number(r.avg_rr_achieved || 0),
        avg_win_pips:  Number(r.avg_win_pips || 0),
        avg_loss_pips: Number(r.avg_loss_pips || 0),
        feature_importance: r.feature_importance || null,
        assumptions:   r.assumptions || null,
        max_drawdown:      Number(r.max_drawdown || 0),
        max_drawdown_pct:  Number(r.max_drawdown_pct || 0),
        equity_curve:      r.equity_curve || [],
        // Win/loss correlation breakdowns — rendered in the correlation
        // panel beneath the per-pair cards.
        by_env:                 r.by_env                 || {},
        by_side:                r.by_side                || {},
        by_hour:                r.by_hour                || {},
        by_dow:                 r.by_dow                 || {},
        confidence_buckets:     r.confidence_buckets     || {},
        pnl_distribution:       r.pnl_distribution       || {},
        component_correlations: r.component_correlations || {},
        by_quality:             r.by_quality             || {},
        by_exit_reason:         r.by_exit_reason         || {},
    };
}

// Render modelling-assumptions footer so the user can see what every P&L
// number is built on (entry fill model, spread, intrabar policy, etc).
function _btRenderAssumptions(perPair) {
    let host = document.getElementById('bt-assumptions');
    if (!host) {
        const body = document.getElementById('bt-pairs-body');
        if (!body || !body.parentNode) return;
        host = document.createElement('div');
        host.id = 'bt-assumptions';
        host.style.cssText = 'margin-top:12px;padding:10px 12px;border:1px dashed var(--bd,#2a2f3a);border-radius:6px;background:var(--bg0);font-size:10px;color:var(--txt3);line-height:1.5';
        body.parentNode.parentNode.appendChild(host);
    }
    const first = (perPair || []).find(p => p && p.assumptions);
    if (!first) { host.innerHTML = ''; return; }
    const a = first.assumptions || {};
    const spread = (a.spread_pips == null) ? 'default per-pair (1.0 majors / 2.0 JPY)' : (a.spread_pips + ' pips (override)');
    // Trade-management line gets its own colour so the "raw strategy"
    // mode is visually unmissable — this is the signal-quality number,
    // not the live-bot simulation.
    const tmRaw = /raw/i.test(String(a.trade_management || ''));
    const tmColour = tmRaw ? 'var(--accent,#5ac8fa)' : 'var(--txt2)';
    host.innerHTML = `
      <div style="font-weight:600;color:var(--txt2);margin-bottom:4px">📐 Modelling assumptions</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 16px">
        <div><b>Trade management:</b> <span style="color:${tmColour}">${_esc(a.trade_management || '—')}</span></div>
        <div><b>Lot size:</b> ${_esc(String(a.lot_size))}</div>
        <div><b>Spread on entry:</b> ${_esc(spread)}</div>
        <div><b>Intrabar TP/SL:</b> ${_esc(a.intrabar_policy || '—')}</div>
        <div><b>Entry fill:</b> ${_esc(a.entry_fill || '—')}</div>
        <div><b>Pip→USD model:</b> ${_esc(a.pip_usd_model || '—')}</div>
        <div><b>Commission:</b> ${_esc(a.commission || '—')}</div>
        <div><b>Swap:</b> ${_esc(a.swap || '—')}</div>
        <div><b>Lookahead:</b> ${_esc(a.lookahead || '—')}</div>
      </div>`;
}

// ── Win/Loss correlation visualisations ─────────────────────────────
//
// Builds a cluster of small multi-pair-aware charts that answer: "what
// correlated with my winning vs losing trades this period?" Drives off
// the breakdowns added to analyze_results (`by_env`, `by_hour`, `by_dow`,
// `by_side`, `confidence_buckets`, `pnl_distribution`, and the existing
// `component_correlations` / `by_quality` / `by_exit_reason`).
//
// Every chart is SVG-free so it renders fast and respects the dashboard's
// CSS variables. Win rate bars are green/amber/red-tinted based on the
// same thresholds the per-pair cards use, and count bars show relative
// volume so a rare-but-profitable bucket is visually distinguishable
// from a dominant one.

function _btColourForWR(wr) {
    // wr ∈ [0,1]
    if (wr >= 0.55) return 'var(--green,#2ecc71)';
    if (wr >= 0.45) return 'var(--amber,#f5a623)';
    return 'var(--red,#ff6b6b)';
}

function _btAggregateBuckets(perPair, field) {
    // Merge per-pair dicts of bucket → {count, wins, losses, total_pnl}
    // into a single aggregated dict the chart can read. Preserves input
    // order by walking pairs in order and using an array for keys.
    const out = {}; // key → stats
    const order = [];
    (perPair || []).forEach(p => {
        const b = (p && p[field]) || {};
        Object.keys(b).forEach(k => {
            const s = b[k] || {};
            if (!out[k]) {
                out[k] = {count: 0, wins: 0, losses: 0, total_pnl: 0, pips_sum: 0};
                order.push(k);
            }
            const n = Number(s.count || 0);
            out[k].count     += n;
            out[k].wins      += Number(s.wins || 0);
            out[k].losses    += Number(s.losses || 0);
            out[k].total_pnl += Number(s.total_pnl || 0);
            out[k].pips_sum  += Number(s.avg_pips || 0) * n;
        });
    });
    // Finalise derived fields
    order.forEach(k => {
        const s = out[k];
        s.win_rate = s.count > 0 ? s.wins / s.count : 0;
        s.avg_pips = s.count > 0 ? s.pips_sum / s.count : 0;
    });
    return {order, data: out};
}

// Render a horizontal bar chart where each row shows: label · win-rate
// bar (width = count share, colour = win-rate), numeric win-rate, and
// count label. Used by env/side/hour/dow/confidence breakdowns.
function _btBucketBarsHTML(agg, opts) {
    opts = opts || {};
    const order = opts.orderOverride || agg.order;
    if (!order || !order.length) {
        return '<div style="color:var(--txt3);font-size:10px;padding:6px 0">No data in this window.</div>';
    }
    const maxCount = Math.max(1, ...order.map(k => Number((agg.data[k] || {}).count || 0)));
    const labelFmt = opts.labelFmt || (k => k);
    return order.map(k => {
        const s = agg.data[k] || {};
        const n = Number(s.count || 0);
        if (!n) return '';
        const wr = Number(s.win_rate || 0);
        const wrLabel = (wr * 100).toFixed(0) + '%';
        const share = (n / maxCount) * 100;
        const colour = _btColourForWR(wr);
        const pnl = Number(s.total_pnl || 0);
        const pnlLabel = pnl >= 0 ? ('+$' + pnl.toFixed(0)) : ('-$' + Math.abs(pnl).toFixed(0));
        const pnlColor = pnl >= 0 ? 'var(--green,#2ecc71)' : 'var(--red,#ff6b6b)';
        return `
          <div style="display:grid;grid-template-columns:110px 1fr 48px 60px 64px;gap:8px;align-items:center;font-size:10px;margin-bottom:3px" title="${_esc(String(k))} · ${n} trade${n===1?'':'s'} · ${s.wins||0}W/${s.losses||0}L · avg ${(Number(s.avg_pips||0)).toFixed(1)}p">
            <div style="color:var(--txt2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${_esc(labelFmt(k))}</div>
            <div style="height:12px;background:var(--bg0);border-radius:3px;overflow:hidden;position:relative">
              <div style="position:absolute;left:0;top:0;bottom:0;width:${share.toFixed(1)}%;background:${colour};opacity:.85"></div>
              <div style="position:absolute;left:0;right:0;top:0;bottom:0;background:linear-gradient(to right, rgba(0,0,0,0) ${(wr*100).toFixed(1)}%, rgba(0,0,0,.35) ${(wr*100).toFixed(1)}%)"></div>
            </div>
            <div style="text-align:right;color:${colour};font-weight:600">${wrLabel}</div>
            <div style="text-align:right;color:var(--txt3)">${n}t</div>
            <div style="text-align:right;color:${pnlColor}">${pnlLabel}</div>
          </div>`;
    }).join('');
}

// Hour-of-day is 24 buckets so we want them in 0..23 order even if some
// hours are missing — fill gaps with zeros so the chart reads like a
// proper session map.
function _btHourBucketsHTML(perPair) {
    const agg = _btAggregateBuckets(perPair, 'by_hour');
    const hours = Array.from({length: 24}, (_, i) => String(i));
    hours.forEach(h => {
        if (!agg.data[h]) {
            agg.data[h] = {count: 0, wins: 0, losses: 0, total_pnl: 0, win_rate: 0, avg_pips: 0};
        }
    });
    return _btBucketBarsHTML({order: hours, data: agg.data}, {
        labelFmt: h => String(h).padStart(2, '0') + ':00',
    });
}

function _btDowBucketsHTML(perPair) {
    const agg = _btAggregateBuckets(perPair, 'by_dow');
    const order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        .filter(d => agg.data[d]);
    return _btBucketBarsHTML({order, data: agg.data});
}

function _btConfidenceBucketsHTML(perPair) {
    const agg = _btAggregateBuckets(perPair, 'confidence_buckets');
    const order = ['<50', '50-60', '60-70', '70-80', '80-90', '90-100']
        .filter(k => agg.data[k]);
    return _btBucketBarsHTML({order, data: agg.data}, {
        labelFmt: k => k + '%',
    });
}

function _btEnvBucketsHTML(perPair) {
    const agg = _btAggregateBuckets(perPair, 'by_env');
    // Sort by trade count descending so the dominant setup is on top
    const order = agg.order.slice().sort((a, b) => {
        return Number((agg.data[b] || {}).count || 0) - Number((agg.data[a] || {}).count || 0);
    });
    return _btBucketBarsHTML({order, data: agg.data});
}

function _btSideBucketsHTML(perPair) {
    const agg = _btAggregateBuckets(perPair, 'by_side');
    const order = ['BUY', 'SELL'].filter(k => agg.data[k]);
    return _btBucketBarsHTML({order, data: agg.data});
}

function _btExitReasonBucketsHTML(perPair) {
    const agg = _btAggregateBuckets(perPair, 'by_exit_reason');
    const order = ['tp', 'sl', 'timeout', 'partial+tp', 'partial+be', 'partial+timeout', 'be+tp', 'be+sl']
        .filter(k => agg.data[k])
        .concat(agg.order.filter(k => !['tp','sl','timeout','partial+tp','partial+be','partial+timeout','be+tp','be+sl'].includes(k)));
    return _btBucketBarsHTML({order, data: agg.data}, {
        labelFmt: k => ({
            'tp': 'Take Profit',
            'sl': 'Stop Loss',
            'timeout': 'Timeout',
            'partial+tp': 'Partial + TP',
            'partial+be': 'Partial + BE',
            'partial+timeout': 'Partial + Timeout',
            'be+tp': 'Runner TP',
            'be+sl': 'Runner BE',
        }[k] || k),
    });
}

function _btQualityBucketsHTML(perPair) {
    const agg = _btAggregateBuckets(perPair, 'by_quality');
    const order = ['strong', 'good', 'fair', 'weak'].filter(k => agg.data[k]);
    return _btBucketBarsHTML({order, data: agg.data}, {
        labelFmt: k => k.charAt(0).toUpperCase() + k.slice(1),
    });
}

// P&L distribution — stacked histogram where each bin is coloured by
// whether it represents a winning / losing outcome. Uses the first pair
// with data (aggregating histograms across pairs would require re-binning;
// instead we render one per pair side-by-side).
function _btPnlDistributionHTML(perPair) {
    const pairs = (perPair || []).filter(p => p && p.pnl_distribution && (p.pnl_distribution.bins || []).length);
    if (!pairs.length) return '<div style="color:var(--txt3);font-size:10px;padding:6px 0">No P&amp;L distribution to plot.</div>';

    return pairs.map(p => {
        const pd = p.pnl_distribution || {};
        const bins = pd.bins || [];
        const maxC = Math.max(1, ...bins.map(b => Number(b.count || 0)));
        const bars = bins.map(b => {
            const h = (Number(b.count || 0) / maxC) * 100;
            const col = b.sign === 'win' ? 'var(--green,#2ecc71)' : (b.sign === 'loss' ? 'var(--red,#ff6b6b)' : 'var(--txt3)');
            const midStr = b.mid >= 0 ? ('+' + Number(b.mid).toFixed(0)) : Number(b.mid).toFixed(0);
            return `
              <div style="flex:1;display:flex;flex-direction:column;align-items:center;min-width:0" title="${midStr}p · ${b.count} trade${b.count===1?'':'s'} (${Number(b.x0).toFixed(1)} → ${Number(b.x1).toFixed(1)})">
                <div style="width:100%;height:70px;display:flex;align-items:flex-end;justify-content:center">
                  <div style="width:70%;height:${h.toFixed(1)}%;background:${col};border-radius:2px 2px 0 0;min-height:${b.count>0?'2px':'0'}"></div>
                </div>
                <div style="font-size:9px;color:var(--txt3);margin-top:2px">${midStr}</div>
              </div>`;
        }).join('');
        return `
          <div style="background:var(--bg1);padding:10px;border-radius:6px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;font-size:10px;color:var(--txt3)">
              <span style="color:var(--txt);font-weight:600">${_esc(p.symbol)}</span>
              <span>n=${pd.n || 0} · median ${Number(pd.median || 0).toFixed(1)}p</span>
            </div>
            <div style="display:flex;gap:2px;align-items:flex-end;padding:0 4px">${bars}</div>
          </div>`;
    }).join('');
}

// Component-score correlations. Each component's coefficient is in
// [-1, 1] — positive means higher score → higher win probability.
function _btCorrelationsHTML(perPair) {
    // Average coefficients across pairs (weighted by trade count).
    const acc = {}; // component → {sum, weight}
    (perPair || []).forEach(p => {
        const cc = (p && p.component_correlations) || {};
        const w  = Number(p.trades || 0);
        Object.keys(cc).forEach(k => {
            const v = Number(cc[k] || 0);
            if (!isFinite(v)) return;
            if (!acc[k]) acc[k] = {sum: 0, weight: 0};
            acc[k].sum    += v * w;
            acc[k].weight += w;
        });
    });
    const keys = Object.keys(acc);
    if (!keys.length) return '<div style="color:var(--txt3);font-size:10px;padding:6px 0">No component correlations available (per-trade score breakdown not reported).</div>';

    const items = keys.map(k => ({
        component: k,
        coef: acc[k].weight > 0 ? acc[k].sum / acc[k].weight : 0,
    })).sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));

    return items.map(it => {
        // −1..+1 → 0..100 offset on a symmetric bar (centre at 50%).
        const centred = 50 + (it.coef * 50);
        const sign = it.coef >= 0 ? '+' : '-';
        const coefLabel = sign + Math.abs(it.coef).toFixed(2);
        const colour = it.coef > 0.15 ? 'var(--green,#2ecc71)'
                     : it.coef < -0.15 ? 'var(--red,#ff6b6b)'
                     : 'var(--amber,#f5a623)';
        const barLeft = it.coef >= 0 ? '50%' : centred.toFixed(1) + '%';
        const barWidth = Math.abs(it.coef * 50).toFixed(1) + '%';
        return `
          <div style="display:grid;grid-template-columns:160px 1fr 56px;gap:8px;align-items:center;font-size:10px;margin-bottom:3px">
            <div style="color:var(--txt2)">${_esc(it.component)}</div>
            <div style="height:12px;background:var(--bg0);border-radius:3px;position:relative">
              <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--bd,#2a2f3a)"></div>
              <div style="position:absolute;left:${barLeft};top:0;bottom:0;width:${barWidth};background:${colour};opacity:.85;border-radius:2px"></div>
            </div>
            <div style="text-align:right;color:${colour};font-weight:600">${coefLabel}</div>
          </div>`;
    }).join('');
}

// Top-level correlation panel. Lays out: env, side, hour, dow,
// confidence, exit reason, quality, component correlations, and P&L
// distribution in a responsive grid beneath the per-pair cards.
function _btRenderCorrelations(perPair) {
    let host = document.getElementById('bt-correlations');
    if (!host) {
        const body = document.getElementById('bt-pairs-body');
        if (!body || !body.parentNode) return;
        host = document.createElement('div');
        host.id = 'bt-correlations';
        host.style.cssText = 'margin-top:16px;border:1px solid var(--bd,#2a2f3a);padding:14px;border-radius:8px;background:var(--bg0)';
        // Inject after the pairs grid but *before* the ML insights panel
        // if that already exists, so the flow reads: cards → correlations → ML.
        const mlHost = document.getElementById('bt-ml-insights');
        if (mlHost && mlHost.parentNode) {
            mlHost.parentNode.insertBefore(host, mlHost);
        } else {
            body.parentNode.parentNode.appendChild(host);
        }
    }

    const withData = (perPair || []).filter(p => p && !p.error);
    if (!withData.length) {
        host.innerHTML = '<div style="color:var(--txt3);font-size:11px">📊 Win/Loss Correlations — no trades in this window.</div>';
        return;
    }

    // Diagnostic: we have pairs, but do they actually include the
    // breakdown fields? If every single pair is missing them AND has
    // trades, the server is running old Python code without the
    // forwarding — tell the user instead of showing eight empty panels.
    const totalTrades = withData.reduce((a, p) => a + Number(p.trades || 0), 0);
    const anyEnv   = withData.some(p => p.by_env  && Object.keys(p.by_env).length);
    const anyHour  = withData.some(p => p.by_hour && Object.keys(p.by_hour).length);
    const anySide  = withData.some(p => p.by_side && Object.keys(p.by_side).length);
    const hasAnyBreakdown = anyEnv || anyHour || anySide;
    if (totalTrades > 0 && !hasAnyBreakdown) {
        host.innerHTML = `
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="font-weight:600;color:var(--txt)">📊 Win / Loss Correlations</div>
          </div>
          <div style="padding:12px;background:var(--bg1);border-left:3px solid var(--amber,#f5a623);border-radius:4px;font-size:11px;color:var(--txt2);line-height:1.6">
            <b>Data pipeline mismatch.</b> The backtest returned ${totalTrades.toLocaleString()} trade${totalTrades===1?'':'s'}, but none of the per-bucket breakdowns
            (by environment, hour, confidence, etc.) came through. The Flask server is almost certainly
            running older Python code that doesn't yet emit these fields.
            <div style="margin-top:8px;color:var(--txt3);font-size:10px">Fix: restart the Flask server so <code>core/server.py</code> and <code>odl/backtest.py</code> reload, then run the backtest again.</div>
          </div>`;
        return;
    }

    const panel = (title, subtitle, bodyHTML) => `
      <div style="background:var(--bg1);padding:12px;border-radius:6px">
        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
          <div style="font-weight:600;color:var(--txt);font-size:11px">${_esc(title)}</div>
          <div style="font-size:9px;color:var(--txt3)">${_esc(subtitle || '')}</div>
        </div>
        ${bodyHTML}
      </div>`;

    const legend = `
      <div style="display:flex;gap:12px;font-size:9px;color:var(--txt3);margin-bottom:10px;flex-wrap:wrap">
        <span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--green,#2ecc71);border-radius:2px;display:inline-block"></span>WR ≥ 55%</span>
        <span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--amber,#f5a623);border-radius:2px;display:inline-block"></span>45–55%</span>
        <span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--red,#ff6b6b);border-radius:2px;display:inline-block"></span>&lt; 45%</span>
        <span style="margin-left:auto">bar width = trade volume · number = win rate · right col = total P&amp;L</span>
      </div>`;

    host.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <div style="font-weight:600;color:var(--txt)">📊 Win / Loss Correlations</div>
        <div style="font-size:10px;color:var(--txt3)">what tends to win — and lose — in this window</div>
      </div>
      ${legend}
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px">
        ${panel('By Environment',  'CHoCH-BUY@PDL, Continuation-SELL@PDH, …', _btEnvBucketsHTML(withData))}
        ${panel('By Side',         'BUY vs SELL',                              _btSideBucketsHTML(withData))}
        ${panel('By Signal Quality','weak / fair / good / strong',              _btQualityBucketsHTML(withData))}
        ${panel('By Exit Reason',  'how trades closed',                         _btExitReasonBucketsHTML(withData))}
        ${panel('By Confidence',   'model-assigned confidence bucket',          _btConfidenceBucketsHTML(withData))}
        ${panel('By Day of Week',  'entry day (UTC)',                           _btDowBucketsHTML(withData))}
        ${panel('By Hour of Day',  'entry hour, UTC — 00–23',                   _btHourBucketsHTML(withData))}
        ${panel('Component Correlations', 'structure / level / momentum / spread / env — +1 = wins, −1 = losses', _btCorrelationsHTML(withData))}
      </div>
      <div style="margin-top:12px">
        <div style="font-weight:600;color:var(--txt);font-size:11px;margin-bottom:6px">P&amp;L Distribution (pips, per pair)</div>
        ${_btPnlDistributionHTML(withData)}
      </div>
    `;
}

// Render feature-importance (sklearn) insights beneath the per-pair cards.
function _btRenderMLInsights(perPair) {
    let host = document.getElementById('bt-ml-insights');
    if (!host) {
        // Inject a container after the pairs grid if the HTML doesn't have one
        const body = document.getElementById('bt-pairs-body');
        if (!body || !body.parentNode) return;
        host = document.createElement('div');
        host.id = 'bt-ml-insights';
        host.style.cssText = 'margin-top:16px;border:1px solid var(--bd,#2a2f3a);padding:14px;border-radius:8px;background:var(--bg0)';
        body.parentNode.parentNode.appendChild(host);
    }

    const withFI = (perPair || []).filter(p => p && p.feature_importance);
    if (!withFI.length) {
        host.innerHTML = '<div style="color:var(--txt3);font-size:11px">🧠 ML Insights (sklearn) — no trained model (pip install scikit-learn, or too few decided trades).</div>';
        return;
    }

    const card = (p) => {
        const fi = p.feature_importance || {};
        if (fi.error) {
            return `
              <div style="background:var(--bg1);padding:10px;border-radius:6px;border-left:3px solid var(--amber,#f5a623)">
                <div style="font-weight:600;color:var(--txt);font-size:12px;margin-bottom:4px">${_esc(p.symbol)}</div>
                <div style="font-size:10px;color:var(--txt3)">${_esc(fi.error)}</div>
              </div>`;
        }
        const imps = Array.isArray(fi.importances) ? fi.importances.slice(0, 8) : [];
        const maxImp = Math.max(0.0001, ...imps.map(r => Number(r.importance || 0)));
        const rows = imps.map(r => {
            const v = Number(r.importance || 0);
            const pct = (v / maxImp) * 100;
            return `
              <div style="display:grid;grid-template-columns:130px 1fr 50px;gap:8px;align-items:center;font-size:10px;margin-bottom:4px">
                <div style="color:var(--txt2)">${_esc(r.feature)}</div>
                <div style="height:10px;background:var(--bg0);border-radius:3px;overflow:hidden">
                  <div style="height:100%;width:${pct.toFixed(1)}%;background:var(--accent,#5ac8fa)"></div>
                </div>
                <div style="text-align:right;color:var(--txt3)">${v.toFixed(3)}</div>
              </div>`;
        }).join('');

        // Prefer balanced accuracy (robust to class imbalance) + lift vs random.
        // Fall back to old fields for back-compat with older server responses.
        const bal = (fi.balanced_accuracy != null) ? fi.balanced_accuracy : null;
        const lift = (fi.lift_vs_random != null) ? fi.lift_vs_random : null;
        const liftLabel = (lift != null) ? ((lift >= 0 ? '+' : '') + (lift * 100).toFixed(1) + '%') : '—';
        const liftColour = (lift != null && lift > 0.02) ? 'var(--green)' : (lift != null && lift < -0.02 ? 'var(--red,#ff6b6b)' : 'var(--txt3)');
        const baseline = fi.baseline_win_rate != null ? (fi.baseline_win_rate * 100).toFixed(1) + '%' : '—';
        const balLabel = bal != null ? (bal * 100).toFixed(1) + '%' : '—';
        const prec = fi.win_precision != null ? (fi.win_precision * 100).toFixed(1) + '%' : '—';
        const rec  = fi.win_recall    != null ? (fi.win_recall    * 100).toFixed(1) + '%' : '—';

        return `
          <div style="background:var(--bg1);padding:12px;border-radius:6px">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
              <div style="font-weight:600;color:var(--txt);font-size:12px">${_esc(p.symbol)}</div>
              <div style="font-size:10px;color:var(--txt3)">n=${fi.n_trades}</div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:10px;color:var(--txt3);margin-bottom:10px">
              <div title="Share of trades that were WINS in this period">
                <div>Baseline WR</div><div style="color:var(--txt);font-size:11px">${baseline}</div>
              </div>
              <div title="Average of WIN-recall and LOSS-recall from out-of-bag predictions. 50% = coin flip.">
                <div>Balanced Acc</div><div style="color:var(--txt);font-size:11px">${balLabel}</div>
              </div>
              <div title="When the model predicts WIN, how often is it right?">
                <div>Win Precision</div><div style="color:var(--txt);font-size:11px">${prec}</div>
              </div>
              <div title="Of all actual WINs, how many did the model catch?">
                <div>Win Recall</div><div style="color:var(--txt);font-size:11px">${rec}</div>
              </div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;font-size:10px;color:var(--txt3)">
              <span>Lift vs. random guess</span>
              <span style="color:${liftColour};font-weight:600">${liftLabel}</span>
            </div>
            ${rows || '<div style="color:var(--txt3);font-size:10px">No importances returned.</div>'}
          </div>`;
    };

    host.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
        <div style="font-weight:600;color:var(--txt)">🧠 ML Insights</div>
        <div style="font-size:10px;color:var(--txt3)">(sklearn · RandomForest feature importance per pair)</div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px">
        ${withFI.map(card).join('')}
      </div>
    `;
}

// Stream backtest progress via Server-Sent Events. EventSource is GET-only,
// so we pass pairs/days/lot_size as query params.
function runBacktest() {
    const btn = document.getElementById('bt-run');
    const daysEl = document.getElementById('bt-days');
    const lotEl = document.getElementById('bt-lot');
    const days = Math.max(1, Math.min(60, Number((daysEl && daysEl.value) || 7)));
    const lot = Math.max(0.01, Number((lotEl && lotEl.value) || 0.5));
    const pairs = Array.from(document.querySelectorAll('.bt-pair'))
        .filter(c => c.checked).map(c => c.value);

    if (!pairs.length) {
        _btSetStatus('Select at least one pair to backtest.', true);
        return;
    }

    if (btn) { btn.disabled = true; btn.textContent = 'Running…'; }
    _btClearResults();
    _btSetStatus(`Backtesting ${pairs.length} pair${pairs.length > 1 ? 's' : ''} over ${days} day${days > 1 ? 's' : ''}…`);

    // Progressive state — fills in as pair_done events arrive
    const perPair = [];
    const pairIndex = {}; // symbol → index in perPair
    const qs = new URLSearchParams({
        pairs: pairs.join(','),
        days: String(days),
        lot_size: String(lot),
    });

    // Close any previous stream cleanly
    if (window.__btSSE) { try { window.__btSSE.close(); } catch (_) {} }
    const es = new EventSource('/api/backtest/stream?' + qs.toString());
    window.__btSSE = es;

    const cleanup = () => {
        try { es.close(); } catch (_) {}
        if (btn) { btn.disabled = false; btn.textContent = 'Run Backtest'; }
        if (window.__btSSE === es) window.__btSSE = null;
    };

    es.addEventListener('hello', () => {
        _btSetStatus(`Connected — starting ${pairs.length} pair${pairs.length > 1 ? 's' : ''}…`);
    });

    es.addEventListener('start', (e) => {
        try { JSON.parse(e.data); } catch (_) {}
    });

    es.addEventListener('pair_start', (e) => {
        try {
            const p = JSON.parse(e.data);
            _btSetStatus(`[${p.symbol}] starting…`);
        } catch (_) {}
    });

    es.addEventListener('progress', (e) => {
        try {
            const p = JSON.parse(e.data);
            const sym = p.symbol || '';
            if (p.type === 'stage') {
                _btSetStatus(`[${sym}] ${p.stage.replace(/_/g, ' ')}…`);
            } else if (p.type === 'bar') {
                const pct = p.total ? Math.round((p.bar / p.total) * 100) : 0;
                _btSetStatus(`[${sym}] signal gen ${p.bar}/${p.total} (${pct}%) · ${p.signals_so_far || 0} signals`);
            } else if (p.type === 'sim_progress') {
                const pct = p.total ? Math.round((p.done / p.total) * 100) : 0;
                _btSetStatus(`[${sym}] simulating trades ${p.done}/${p.total} (${pct}%)`);
            } else if (p.type === 'signals_done') {
                _btSetStatus(`[${sym}] ${p.signals} signals generated`);
            } else if (p.type === 'sim_done') {
                _btSetStatus(`[${sym}] ${p.trades} trades simulated`);
            }
        } catch (_) {}
    });

    es.addEventListener('pair_done', (e) => {
        try {
            const p = JSON.parse(e.data);
            const norm = _btNormalisePair(p);
            if (pairIndex[norm.symbol] != null) {
                perPair[pairIndex[norm.symbol]] = norm;
            } else {
                pairIndex[norm.symbol] = perPair.length;
                perPair.push(norm);
            }
            // Progressive render
            _btRenderPairCards(perPair);
            _btRenderPnlChart(perPair, perPair.reduce((a, r) => a + Number(r.total_pnl || 0), 0));
            _btRenderCorrelations(perPair);
            _btRenderAssumptions(perPair);
        } catch (_) {}
    });

    es.addEventListener('done', (e) => {
        try {
            const data = JSON.parse(e.data);
            data.per_pair = (data.per_pair || []).map(_btNormalisePair);
            _btRenderResults(data);
            _btRenderCorrelations(data.per_pair);
            _btRenderMLInsights(data.per_pair);
            _btRenderAssumptions(data.per_pair);
            const dur = data.duration_s != null ? ` in ${data.duration_s}s` : '';
            _btSetStatus(`Done${dur}.`);
        } catch (err) {
            _btSetStatus('Stream ended but payload was unreadable.', true);
        } finally {
            cleanup();
        }
    });

    es.addEventListener('error', (e) => {
        // EventSource fires 'error' both for HTTP errors and on close. Treat
        // it as fatal only if the connection is not OPEN.
        let msg = 'stream error';
        try {
            if (e && e.data) msg = (JSON.parse(e.data).error || msg);
        } catch (_) {}
        if (es.readyState === EventSource.CLOSED) {
            // Connection is dead — probe the endpoint to find out WHY.
            // EventSource swallows HTTP status codes so we re-fetch to expose
            // 404 / 409 / 500 / server-not-restarted scenarios to the user.
            fetch('/api/backtest/stream?' + qs.toString(), { method: 'GET' })
                .then(async (r) => {
                    if (r.status === 404) {
                        _btSetStatus(
                            'Backtest failed: server does not know /api/backtest/stream — restart the Flask server so the new code loads.',
                            true);
                    } else if (r.status === 409) {
                        _btSetStatus('Backtest failed: another backtest is still running.', true);
                    } else if (r.status >= 500) {
                        let body = '';
                        try { body = (await r.text()).slice(0, 300); } catch (_) {}
                        _btSetStatus(`Backtest failed: server error ${r.status}. ${body}`, true);
                    } else if (!r.ok) {
                        _btSetStatus(`Backtest failed: HTTP ${r.status}.`, true);
                    } else {
                        _btSetStatus('Backtest failed: ' + msg + ' (stream closed before any events).', true);
                    }
                })
                .catch((err) => {
                    _btSetStatus('Backtest failed: cannot reach server (' + String(err).slice(0, 120) + ').', true);
                })
                .finally(() => cleanup());
        } else if (es.readyState === EventSource.CONNECTING) {
            // transient — let the browser reconnect silently
        } else {
            _btSetStatus('Backtest failed: ' + msg, true);
            cleanup();
        }
    });
}
