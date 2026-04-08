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