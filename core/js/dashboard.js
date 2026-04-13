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

// Explicit helper so the AI Log tab can refresh on demand
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

// ── AI Log ────────────────────────────────────────────────────
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
        container.innerHTML = '<div style="padding:40px;text-align:center;color:var(--txt3)">No AI thoughts yet — start the bot to see live reasoning.</div>';
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
        return `
        <div style="padding:12px 16px;border-bottom:1px solid var(--line);font-size:11px;transition:background 0.15s" onmouseover="this.style.background='rgba(255,255,255,0.02)'" onmouseout="this.style.background=''">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:8px;flex-wrap:wrap">
                <div style="display:flex;align-items:center;gap:6px">
                    <strong style="color:var(--cyan)">${_esc(t.source || '—')}</strong>
                    <span style="background:var(--bg2);color:var(--txt2);padding:2px 6px;border-radius:3px">${_esc(t.stage || '')}</span>
                    ${confHtml}${actionHtml}
                </div>
                <span style="color:var(--txt3);white-space:nowrap">${new Date(t.ts).toLocaleTimeString()}</span>
            </div>
            <div style="color:var(--txt);margin-bottom:${t.detail ? 4 : 0}px;line-height:1.4">${_esc(t.summary || '')}</div>
            ${t.detail ? `<div style="color:var(--txt3);font-size:10px;margin-top:2px;line-height:1.4">📋 ${_esc(t.detail)}</div>` : ''}
        </div>`;
    }).join('');
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
            if (data.equity_curve && data.equity_curve.length > 0) {
                equityData = data.equity_curve;
                renderEquityCurve(equityData);
            }
        })
        .catch(() => renderKPIs({}));
}

function renderKPIs(kpis) {
    _setKPI('kpi-winrate',      (kpis.win_rate       || 0).toFixed(1) + '%');
    _setKPI('kpi-profitfactor', (kpis.profit_factor  || 0).toFixed(2));
    _setKPI('kpi-trades',       kpis.total_trades     || 0);
    _setKPI('kpi-return',       (kpis.equity_return  || 0).toFixed(2) + '%');
    _setKPI('kpi-drawdown',     (kpis.max_drawdown   || 0).toFixed(2) + '%');
    _setKPI('kpi-sharpe',       (kpis.sharpe         || 0).toFixed(2));
    _setKPI('kpi-avgratio',     (kpis.avg_win_loss_ratio || 0).toFixed(2));
    _setKPI('kpi-currentdd',    (kpis.current_drawdown   || 0).toFixed(2) + '%');
    _setKPI('kpi-recovery',     (kpis.recovery_factor    || 0).toFixed(2));
    _setKPI('kpi-sortino',      (kpis.sortino        || 0).toFixed(2));
}

function renderEquityCurve(data) {
    const canvas = document.getElementById('equityCanvas');
    if (!canvas || !data || data.length < 2) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = (rect.width  || 600) * dpr;
    canvas.height = (rect.height || 200) * dpr;
    ctx.scale(dpr, dpr);
    const W = canvas.width  / dpr;
    const H = canvas.height / dpr;
    const pad = { top: 20, right: 20, bottom: 28, left: 52 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top  - pad.bottom;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    ctx.fillStyle = '#0a0e14';
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(42,58,82,0.6)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH * i / 4);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        const val = max - (range * i / 4);
        ctx.fillStyle = '#6a7280';
        ctx.font = '9px "IBM Plex Mono",monospace';
        ctx.textAlign = 'right';
        ctx.fillText((val >= 0 ? '+' : '') + val.toFixed(0), pad.left - 4, y + 3);
    }

    const lastVal = data[data.length - 1];
    const lineCol = lastVal >= 0 ? '#50d963' : '#ff6b6b';
    const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    grad.addColorStop(0, lastVal >= 0 ? 'rgba(80,217,99,0.35)' : 'rgba(255,107,107,0.35)');
    grad.addColorStop(1, 'rgba(10,14,20,0)');

    const xStep = plotW / (data.length - 1);
    const xAt = i => pad.left + i * xStep;
    const yAt = v => pad.top + plotH - ((v - min) / range) * plotH;

    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(xAt(i), yAt(data[i]));
    ctx.lineTo(xAt(data.length - 1), H - pad.bottom);
    ctx.lineTo(xAt(0), H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(xAt(i), yAt(data[i]));
    ctx.strokeStyle = lineCol;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    if (min < 0 && max > 0) {
        const y0 = yAt(0);
        ctx.beginPath(); ctx.moveTo(pad.left, y0); ctx.lineTo(W - pad.right, y0);
        ctx.strokeStyle = 'rgba(160,170,184,0.3)'; ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    }

    ctx.fillStyle = '#6a7280';
    ctx.font = '9px "IBM Plex Mono",monospace';
    ctx.textAlign = 'center';
    ctx.fillText('start', pad.left, H - 6);
    ctx.fillText('mid', pad.left + plotW / 2, H - 6);
    ctx.fillText('now', W - pad.right, H - 6);
}

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
