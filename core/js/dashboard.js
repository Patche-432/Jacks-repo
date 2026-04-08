let mt5Connected = false;

function connectMt5() {
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
        } else {
            btn.textContent = 'Connect MT5';
            console.error('MT5 error:', data.error);
        }
    })
    .catch(error => {
        console.error('Connection error:', error);
        btn.textContent = 'Connect MT5';
    })
    .finally(() => {
        btn.disabled = false;
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