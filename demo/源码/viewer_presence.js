/* Each top-level Viewer page owns a lease; hidden tabs stay registered. */
(() => {
  const client = crypto.randomUUID().replaceAll('-', '');
  const endpoint = '/api/viewer-presence';
  let leaving = false;
  let sequence = 0;
  let timer;
  async function heartbeat() {
    if (leaving) return;
    try {
      await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({client,alive:true,sequence:++sequence}), cache:'no-store', keepalive:true});
    } catch (_) { /* A foreground/debug server may be restarting. */ }
  }
  function start() {
    leaving = false;
    clearInterval(timer);
    heartbeat();
    timer = setInterval(heartbeat, 15000);
  }
  window.addEventListener('pagehide', () => {
    leaving = true;
    clearInterval(timer);
    navigator.sendBeacon(endpoint, new Blob([JSON.stringify({client,alive:false,sequence:++sequence})], {type:'application/json'}));
  });
  // pageshow also covers back/forward cache restoration.
  window.addEventListener('pageshow', start);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) heartbeat(); });
  start();
})();
