// Auto-connect to MT5 on page load
window.addEventListener('load', () => {
    autoConnectMt5();
});

function autoConnectMt5() {
    // Silently attempt connection on load
    fetch('/api/mt5/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.connected) {
            updateMt5Display(data);
            const btn = document.getElementById('connect-btn');
            btn.style.background = 'var(--green)';
            btn.textContent = 'MT5 ✓';
        }
    })
    .catch(() => {});
}

function connectMt5() {
    // Check if this is the header button or modal button
    const headerBtn = document.getElementById('connect-btn');
    const modalBtn = document.getElementById('mt5-modal-btn');
    
    // If header button is clicked, attempt direct connection first
    if (event && event.target === headerBtn) {
        attemptDirectConnection();
        return;
    }
    
    // Modal connect button clicked - attempt connection with credentials
    const btn = modalBtn;
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = 'Connecting…';
    
    fetch('/api/mt5/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.connected) {
            mt5Connected = true;
            updateMt5Display(data);
            const connectBtn = document.getElementById('connect-btn');
            connectBtn.style.background = 'var(--green)';
            connectBtn.textContent = 'MT5 ✓';
            closeMt5Modal();
            
            // Place test trade after successful connection
            placeTestTrade();
        } else {
            const statusEl = document.getElementById('mt5-modal-status');
            statusEl.style.display = 'block';
            statusEl.style.background = 'rgba(255, 107, 107, 0.1)';
            statusEl.style.color = 'var(--red)';
            statusEl.textContent = 'Error: ' + (data.error || 'Connection failed');
            console.error('MT5 error:', data.error);
        }
    })
    .catch(error => {
        const statusEl = document.getElementById('mt5-modal-status');
        statusEl.style.display = 'block';
        statusEl.style.background = 'rgba(255, 107, 107, 0.1)';
        statusEl.style.color = 'var(--red)';
        statusEl.textContent = 'Error: ' + error.message;
        console.error('Connection error:', error);
    })
    .finally(() => {
        btn.disabled = false;
        btn.textContent = 'Connect';
    });
}

function attemptDirectConnection() {
    const btn = document.getElementById('connect-btn');
    btn.disabled = true;
    btn.textContent = 'Connecting…';
    
    fetch('/api/mt5/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.connected) {
            mt5Connected = true;
            updateMt5Display(data);
            btn.style.background = 'var(--green)';
            btn.textContent = 'MT5 ✓';
            
            // Place test trade after successful connection
            placeTestTrade();
        } else {
            // Connection failed, show modal for credentials
            document.getElementById('mt5-modal-overlay').style.display = 'flex';
            btn.disabled = false;
            btn.textContent = 'Connect MT5';
        }
    })
    .catch(error => {
        // Connection failed, show modal for credentials
        document.getElementById('mt5-modal-overlay').style.display = 'flex';
        btn.disabled = false;
        btn.textContent = 'Connect MT5';
    });
}

function placeTestTrade() {
    fetch('/api/mt5/test-trade', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            console.log('✓ Test trade placed:', data.ticket, '@', data.price);
        } else {
            console.warning('Test trade failed:', data.error);
        }
    })
    .catch(error => {
        console.error('Test trade error:', error);
    });
}

function updateMt5Display(data) {
    const dot = document.getElementById('mt5-dot');
    const statusTxt = document.getElementById('mt5-status-txt');
    const detail = document.getElementById('mt5-detail');
    
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
    
    document.getElementById('m-server').textContent = data.server || '—';
    document.getElementById('m-login').textContent = (data.login || '—') + (data.account_name ? ' · ' + data.account_name : '');
    
    const trEl = document.getElementById('m-trade');
    trEl.textContent = data.trade_allowed ? 'Allowed' : 'Disabled';
    trEl.style.color = data.trade_allowed ? 'var(--green)' : 'var(--red)';
    
    // Equity and Balance
    const equityEl = document.getElementById('m-equity');
    if (equityEl && data.equity !== undefined) {
        equityEl.textContent = '$' + data.equity.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
    
    const balanceEl = document.getElementById('m-balance');
    if (balanceEl && data.balance !== undefined) {
        balanceEl.textContent = '$' + data.balance.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
}

setInterval(() => {
    fetch('/api/mt5/status')
        .then(r => r.json())
        .then(data => updateMt5Display(data))
        .catch(() => {});
}, 2000);

// Bot Control
async function startBot() {
    const btn = document.getElementById('start-btn');
    btn.disabled = true;
    btn.textContent = 'Starting…';
    
    try {
        const r = await fetch('/bot/start', {method: 'POST'});
        const data = await r.json();
        if (data.ok && data.running) {
            document.getElementById('dot').className = 'dot live';
            document.getElementById('status-text').textContent = 'live';
            document.getElementById('start-btn').style.display = 'none';
            document.getElementById('stop-btn').style.display = 'inline-block';
        } else {
            alert('Error: ' + (data.error || 'Unknown'));
        }
    } catch (e) {
        alert('Failed to start bot: ' + e.message);
    }
    btn.disabled = false;
}

async function stopBot() {
    const btn = document.getElementById('stop-btn');
    btn.disabled = true;
    
    try {
        await fetch('/bot/stop', {method: 'POST'});
        document.getElementById('dot').className = 'dot';
        document.getElementById('status-text').textContent = 'idle';
        document.getElementById('start-btn').style.display = 'inline-block';
        document.getElementById('stop-btn').style.display = 'none';
    } catch (e) {
        alert('Failed to stop bot: ' + e.message);
    }
    btn.disabled = false;
}

// Other functions (placeholder)
function showTab(tab) { console.log('Showing tab:', tab); }
function toggleSym(el) { el.classList.toggle('active'); }
function stepInput(id, delta, min) { }
function setPoll(val, el) { }
function updateRR() { }
function clearThoughts() { }
function applyConfig() { }

function showTab(tab) {
    // Hide all tabs
    document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    // Show selected tab
    const tabEl = document.getElementById('tab-' + tab);
    if (tabEl) {
        tabEl.classList.add('active');
        event.target.classList.add('active');
    }
}

function toggleSym(el) {
    el.classList.toggle('active');
    updateSymbols();
}

function updateSymbols() {
    const active = Array.from(document.querySelectorAll('.sym-chip.active'))
        .map(el => el.textContent)
        .join(',');
    document.getElementById('cfg-symbols').value = active;
}

function stepInput(id, delta, min) {
    const el = document.getElementById(id);
    const val = parseFloat(el.value) + delta;
    el.value = Math.max(min, val);
}

function setPoll(val, el) {
    document.querySelectorAll('.poll-preset').forEach(e => e.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('cfg-poll').value = val;
    const display = val >= 60 ? Math.round(val/60) + 'm' : val + 's';
    document.getElementById('poll-display').textContent = display;
}

function updateRR() {
    const atr = parseFloat(document.getElementById('cfg-atr-mult').value);
    const sl = parseFloat(document.getElementById('cfg-sl-mult').value);
    const tp = parseFloat(document.getElementById('cfg-tp-mult').value);
    const ratio = (tp * atr) / (sl * atr);
    document.getElementById('rr-label').textContent = '1 : ' + ratio.toFixed(1);
    document.getElementById('rr-tp-bar').style.width = (ratio * 33) + '%';
}

function clearThoughts() {
    document.getElementById('thoughts').innerHTML = '';
    document.getElementById('thought-count').textContent = '0 entries';
}

function applyConfig() {
    console.log('Config applied');
}
// Modal functions
function openMt5Modal() {
    document.getElementById('mt5-modal-overlay').style.display = 'flex';
}

function closeMt5Modal() {
    document.getElementById('mt5-modal-overlay').style.display = 'none';
}

// Proper tab switching
function showTab(tab) {
    document.querySelectorAll('.tab-panel').forEach(el => {
        el.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('active');
    });
    
    const panel = document.getElementById('tab-' + tab);
    if (panel) panel.classList.add('active');
    
    event.target.classList.add('active');
}

// Complete other functions
function toggleSym(el) {
    el.classList.toggle('active');
    updateSymbols();
}

function updateSymbols() {
    const active = Array.from(document.querySelectorAll('.sym-chip.active'))
        .map(el => el.textContent)
        .join(',');
    document.getElementById('cfg-symbols').value = active;
}

function stepInput(id, delta, min) {
    const el = document.getElementById(id);
    let val = parseFloat(el.value) + delta;
    el.value = Math.max(min, val).toFixed(2);
}

function setPoll(val, el) {
    document.querySelectorAll('.poll-preset').forEach(e => e.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('cfg-poll').value = val;
    const display = val >= 60 ? Math.round(val/60) + 'm' : val + 's';
    document.getElementById('poll-display').textContent = display;
}

function updateRR() {
    const atr = parseFloat(document.getElementById('cfg-atr-mult').value);
    const sl = parseFloat(document.getElementById('cfg-sl-mult').value);
    const tp = parseFloat(document.getElementById('cfg-tp-mult').value);
    const ratio = (tp * atr) / (sl * atr);
    document.getElementById('rr-label').textContent = '1 : ' + ratio.toFixed(1);
    document.getElementById('rr-tp-bar').style.width = (ratio * 33) + '%';
}

function clearThoughts() {
    document.getElementById('thoughts').innerHTML = '';
    document.getElementById('thought-count').textContent = '0 entries';
}

function applyConfig() {
    console.log('Config applied:', {
        symbols: document.getElementById('cfg-symbols').value,
        volume: document.getElementById('cfg-volume').value,
        poll: document.getElementById('cfg-poll').value,
    });
}
// Fetch AI thoughts every 2 seconds
let lastThoughtTs = null;

setInterval(() => {
    fetch('/bot/ai_thoughts?limit=60')
        .then(r => r.json())
        .then(data => {
            if (!data.ok) return;
            const thoughts = data.thoughts || [];
            displayThoughts(thoughts);
        })
        .catch(() => {});
}, 2000);

function displayThoughts(thoughts) {
    const container = document.getElementById('thoughts');
    const count = document.getElementById('thought-count');
    
    if (thoughts.length === 0) {
        container.innerHTML = '<div class="empty-state">No AI thoughts yet</div>';
        count.textContent = '0 entries';
        return;
    }
    
    count.textContent = thoughts.length + ' entries';
    
    container.innerHTML = thoughts.map(t => `
        <div class="thought-entry" style="padding:10px; border-bottom:1px solid var(--line); font-size:11px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <strong style="color:var(--cyan)">${t.source}</strong>
                <span style="color:var(--txt2);">${t.stage}</span>
                <span style="color:var(--txt3);">${new Date(t.ts).toLocaleTimeString()}</span>
            </div>
            <div style="color:var(--txt); margin-bottom:4px;">${t.summary}</div>
            ${t.detail ? `<div style="color:var(--txt2); font-size:10px; margin-bottom:2px;">📋 ${t.detail}</div>` : ''}
            ${t.confidence !== null ? `<div style="color:var(--txt3);">Confidence: ${(t.confidence*100).toFixed(0)}%</div>` : ''}
        </div>
    `).join('');
}

// Portfolio Watch updater - displays real-time pair performance
function updatePortfolioWatch(data) {
    const pairs = ['GBPJPY', 'EURJPY', 'GBPUSD', 'EURUSD'];
    
    // Build pair stats from open trades
    const pairStats = {};
    pairs.forEach(pair => {
        pairStats[pair] = { trades: 0, pnl: 0, hasTrade: false };
    });
    
    if (data.open_trades && data.open_trades.length > 0) {
        data.open_trades.forEach(trade => {
            if (pairStats[trade.symbol]) {
                pairStats[trade.symbol].trades++;
                pairStats[trade.symbol].pnl += trade.profit || 0;
                pairStats[trade.symbol].hasTrade = true;
            }
        });
    }
    
    // Update each watch card
    pairs.forEach(pair => {
        const stats = pairStats[pair];
        
        // Status indicator
        const statusEl = document.getElementById(`status-${pair}`);
        if (statusEl) {
            const dot = statusEl.querySelector('.watch-dot');
            const state = statusEl.querySelector('.watch-state');
            
            if (stats.hasTrade) {
                dot.className = 'watch-dot trading';
                state.textContent = 'Trading';
            } else {
                dot.className = 'watch-dot idle';
                state.textContent = 'Idle';
            }
        }
        
        // Trades count
        const tradesEl = document.getElementById(`trades-${pair}`);
        if (tradesEl) {
            tradesEl.textContent = stats.trades;
        }
        
        // P&L with color
        const pnlEl = document.getElementById(`pnl-${pair}`);
        if (pnlEl) {
            const pnlStr = stats.pnl >= 0 ? `+$${stats.pnl.toFixed(0)}` : `-$${Math.abs(stats.pnl).toFixed(0)}`;
            pnlEl.textContent = pnlStr;
            pnlEl.style.color = stats.pnl >= 0 ? 'var(--green)' : 'var(--red)';
        }
    });
}

// Update Summary Snapshot every 3 seconds
setInterval(() => {
    fetch('/bot/status')
        .then(r => r.json())
        .then(data => {
            // Bot status
            const botStatus = data.running ? 'Running ✓' : 'Idle';
            const botStatusColor = data.running ? 'var(--green)' : 'var(--txt2)';
            document.getElementById('snap-bot-status').textContent = botStatus;
            document.getElementById('snap-bot-status').style.color = botStatusColor;
            
            // Market bias
            const bias = data.market_bias || '—';
            document.getElementById('snap-bias').textContent = bias;
            
            // Confidence
            const conf = data.confidence !== undefined ? (data.confidence * 100).toFixed(0) + '%' : '—';
            document.getElementById('snap-confidence').textContent = conf;
            
            // Last signal
            const lastSig = data.last_signal || 'None';
            document.getElementById('snap-last-signal').textContent = lastSig;
            
            // Trade details
            const tradeDetails = document.getElementById('snap-trade-details');
            if (data.open_trades && data.open_trades.length > 0) {
                tradeDetails.innerHTML = data.open_trades.map(t => `
                    <div style="padding:10px;background:var(--bg1);border-radius:6px">
                        <div style="font-weight:600;color:var(--cyan);margin-bottom:4px">${t.symbol} ${t.direction}</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:10px">
                            <div>Entry: <span style="color:var(--txt)">${t.entry_price}</span></div>
                            <div>SL: <span style="color:var(--red)">${t.stop_loss}</span></div>
                            <div>TP: <span style="color:var(--green)">${t.take_profit}</span></div>
                            <div>Profit: <span style="color:${t.profit > 0 ? 'var(--green)' : 'var(--red)'}">${t.profit > 0 ? '+' : ''}${t.profit.toLocaleString()}</span></div>
                        </div>
                    </div>
                `).join('');
            } else {
                tradeDetails.innerHTML = '<div style="padding:12px;background:var(--bg1);border-radius:6px;color:var(--txt2);text-align:center">No active trades</div>';
            }
            
            // Session summary
            const summary = data.session_summary || 'Waiting for bot to connect…';
            document.getElementById('snap-session-summary').textContent = summary;
            
            // Update Portfolio Watch visualization
            updatePortfolioWatch(data);
        })
        .catch(() => {
            // Silently fail if endpoint not available yet
        });
}, 3000);