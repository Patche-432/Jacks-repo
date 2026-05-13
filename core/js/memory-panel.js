// ── Trade Memory panel ─────────────────────────────────────────────────────────
// SQLite-backed cumulative stats panel — shown beneath ML Insights on Backtest tab.
// Injected into the DOM dynamically after bt-ml-insights.
// Triggered: on Backtest tab open, and after every completed backtest run.

function _btRenderMemory(data) {
    var host = document.getElementById("bt-memory-panel");
    if (!host) {
        var mlHost = document.getElementById("bt-ml-insights");
        if (!mlHost || !mlHost.parentNode) return;
        host = document.createElement("div");
        host.id = "bt-memory-panel";
        host.style.cssText = "margin-top:16px;border:1px solid var(--bd,#2a2f3a);padding:14px;border-radius:8px;background:var(--bg0)";
        mlHost.parentNode.insertBefore(host, mlHost.nextSibling);
    }
    if (!data || !data.ok) {
        host.innerHTML = "<div style=\"color:var(--txt3);font-size:11px\">\uD83D\uDCBE Trade Memory \u2014 SQLite not yet initialised (run first backtest).</div>";
        return;
    }
    var pairs = data.pairs || {};
    var syms  = Object.keys(pairs);
    if (!syms.length) {
        host.innerHTML = "<div style=\"color:var(--txt3);font-size:11px\">\uD83D\uDCBE Trade Memory \u2014 no data stored yet.</div>";
        return;
    }
    var totalRuns=0, totalTrades=0, totalWins=0, totalPnl=0;
    syms.forEach(function(s) {
        var p=pairs[s]||{};
        totalRuns   += Number(p.n_runs       ||0);
        totalTrades += Number(p.total_trades ||0);
        totalWins   += Number(p.total_wins   ||0);
        totalPnl    += Number(p.total_pnl    ||0);
    });
    var overallWR = totalTrades>0 ? totalWins/totalTrades : 0;
    var pnlSign   = totalPnl>=0 ? "+" : "-";
    var pnlAbs    = Math.abs(totalPnl).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
    var pnlC      = totalPnl>=0 ? "var(--green,#2ecc71)" : "var(--red,#ff6b6b)";
    var wrFmt     = (overallWR*100).toFixed(1)+"%";
    var wrC       = overallWR>=0.55 ? "var(--green,#2ecc71)" : overallWR>=0.45 ? "var(--amber,#f5a623)" : "var(--red,#ff6b6b)";
    var latestTs  = syms.map(function(s){return (pairs[s]||{}).latest;}).filter(Boolean).sort().slice(-1)[0];
    var earliest  = syms.map(function(s){return (pairs[s]||{}).earliest;}).filter(Boolean).sort()[0];
    var latFmt    = latestTs ? new Date(latestTs).toLocaleString()     : "\u2014";
    var earFmt    = earliest ? new Date(earliest).toLocaleDateString() : "\u2014";

    function memCard(sym) {
        var p=pairs[sym]||{};
        if (p.error) return "<div style=\"background:var(--bg1);padding:10px;border-radius:6px;border-left:3px solid var(--amber,#f5a623)\">" +
            "<div style=\"font-weight:600;font-size:12px;color:var(--txt);margin-bottom:4px\">" + _esc(sym) + "</div>" +
            "<div style=\"font-size:10px;color:var(--txt3)\">" + _esc(p.error) + "</div></div>";
        var runs   = Number(p.n_runs       ||0);
        var trades = Number(p.total_trades ||0);
        var wins   = Number(p.total_wins   ||0);
        var pnl    = Number(p.total_pnl    ||0);
        var wr     = trades>0 ? wins/trades : 0;
        var wrCl   = wr>=0.55 ? "var(--green,#2ecc71)" : wr>=0.45 ? "var(--amber,#f5a623)" : "var(--red,#ff6b6b)";
        var pnlCl  = pnl>=0  ? "var(--green,#2ecc71)" : "var(--red,#ff6b6b)";
        var pnlStr = (pnl>=0?"+":"-")+"$"+Math.abs(pnl).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
        var wrStr  = (wr*100).toFixed(1)+"%";
        // importances
        var imps   = Object.entries(p.importances||{}).sort(function(a,b){return b[1]-a[1];}).slice(0,5);
        var maxImp = imps.length ? imps[0][1] : 1;
        var impRows = imps.map(function(kv) {
            var pct=(kv[1]/maxImp*100).toFixed(1);
            return "<div style=\"display:grid;grid-template-columns:120px 1fr 42px;gap:6px;align-items:center;font-size:9px;margin-bottom:3px\">" +
                "<div style=\"color:var(--txt2)\">"+_esc(kv[0])+"</div>" +
                "<div style=\"height:8px;background:var(--bg0);border-radius:2px;overflow:hidden\">" +
                  "<div style=\"height:100%;width:"+pct+"%;background:var(--accent,#5ac8fa)\"></div></div>" +
                "<div style=\"text-align:right;color:var(--txt3)\">"+Number(kv[1]).toFixed(3)+"</div></div>";
        }).join("");
        // tuned params
        var tp=p.tuned_params||{};
        var tpK=Object.keys(tp);
        var tpGrid=tpK.length ? tpK.map(function(k){
            var lbl=k.replace(/_/g," ").replace("mult","\u00d7").replace("pips","p").replace("rr","R");
            return "<div style=\"font-size:9px\"><div style=\"color:var(--txt3);margin-bottom:1px\">"+_esc(lbl)+"</div>" +
                "<div style=\"color:var(--txt);font-weight:600\">"+Number(tp[k]).toFixed(2)+"</div></div>";
        }).join("") : "<div style=\"font-size:9px;color:var(--txt3)\">defaults</div>";
        var impSec = imps.length ? "<div style=\"margin-bottom:8px\">" +
            "<div style=\"font-size:9px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px\">Aggregated importances</div>" +
            impRows+"</div>" : "";
        var tpSec  = tpK.length ? "<div>" +
            "<div style=\"font-size:9px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px\">Tuned params</div>" +
            "<div style=\"display:grid;grid-template-columns:repeat(3,1fr);gap:4px\">"+tpGrid+"</div></div>" : "";
        return "<div style=\"background:var(--bg1);padding:12px;border-radius:6px\">" +
            "<div style=\"display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px\">" +
              "<div style=\"font-weight:600;font-size:13px;color:var(--txt)\">"+_esc(sym)+"</div>" +
              "<div style=\"font-size:9px;color:var(--txt3)\">"+runs+" run"+(runs!==1?"s":"")+"</div></div>" +
            "<div style=\"display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px;font-size:10px;text-align:center\">" +
              "<div style=\"background:var(--bg0);padding:6px;border-radius:4px\">" +
                "<div style=\"color:var(--txt3);font-size:9px\">Trades</div>" +
                "<div style=\"font-weight:600;color:var(--txt)\">"+trades+"</div></div>" +
              "<div style=\"background:var(--bg0);padding:6px;border-radius:4px\">" +
                "<div style=\"color:var(--txt3);font-size:9px\">Win Rate</div>" +
                "<div style=\"font-weight:600;color:"+wrCl+"\">"+wrStr+"</div></div>" +
              "<div style=\"background:var(--bg0);padding:6px;border-radius:4px\">" +
                "<div style=\"color:var(--txt3);font-size:9px\">Net P&amp;L</div>" +
                "<div style=\"font-weight:600;color:"+pnlCl+"\">"+pnlStr+"</div></div></div>" +
            impSec+tpSec+"</div>";
    }

    host.innerHTML =
        "<div style=\"display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap\">" +
          "<div style=\"font-weight:600;color:var(--txt)\">\uD83D\uDCBE Trade Memory</div>" +
          "<div style=\"font-size:10px;color:var(--txt3)\">SQLite \u2014 cumulative across all runs \u00b7 "+earFmt+" onwards</div>" +
          "<div style=\"margin-left:auto;display:flex;gap:10px;font-size:10px;font-family:var(--mono)\">" +
            "<span style=\"color:var(--txt3)\">runs <span style=\"color:var(--txt);font-weight:600\">"+totalRuns+"</span></span>" +
            "<span style=\"color:var(--txt3)\">trades <span style=\"color:var(--txt);font-weight:600\">"+totalTrades+"</span></span>" +
            "<span style=\"color:var(--txt3)\">WR <span style=\"color:"+wrC+";font-weight:600\">"+wrFmt+"</span></span>" +
            "<span style=\"color:var(--txt3)\">P&amp;L <span style=\"color:"+pnlC+";font-weight:600\">"+pnlSign+"$"+pnlAbs+"</span></span>" +
            "<span style=\"color:var(--txt3)\">updated <span style=\"color:var(--txt)\">"+latFmt+"</span></span>" +
          "</div></div>" +
        "<div style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px\">" +
        syms.map(memCard).join("")+"</div>";
}

function _btLoadMemory() {
    fetch("/api/backtest/memory")
        .then(function(r){return r.ok?r.json():Promise.reject(r.status);})
        .then(function(d){_btRenderMemory(d);})
        .catch(function(){
            var h=document.getElementById("bt-memory-panel");
            if(h) h.innerHTML="<div style=\"color:var(--txt3);font-size:11px\">\uD83D\uDCBE Trade Memory \u2014 could not reach /api/backtest/memory (restart server).</div>";
        });
}
