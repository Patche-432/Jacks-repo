// Simplified MT5 Connection Implementation

function connectMt5() {
    fetch('/api/mt5/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        // Update the sidebar with the connection status
        updateSidebar(data);
    })
    .catch(error => console.error('Error connecting to MT5:', error));
}

function updateSidebar(data) {
    // Implementation for updating the sidebar with MT5 status
    console.log('MT5 Status:', data);
}

// Auto-polling every 2 seconds to check MT5 status
setInterval(() => {
    fetch('/api/mt5/status')
        .then(response => response.json())
        .then(data => updateSidebar(data))
        .catch(error => console.error('Error fetching MT5 status:', error));
}, 2000);