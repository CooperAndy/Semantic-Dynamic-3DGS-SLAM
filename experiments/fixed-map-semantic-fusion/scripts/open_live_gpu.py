"""Open the active localhost GPU viewer or establish its interactive SSH tunnel."""
import urllib.request,webbrowser,runpy,sys
from pathlib import Path
try:
 with urllib.request.urlopen('http://127.0.0.1:8766/status',timeout=2) as r:assert r.status==200
except Exception:
 sys.argv=[str(Path(__file__).with_name('live_gpu_tunnel.py')),'--open']
 runpy.run_path(sys.argv[0],run_name='__main__')
else:webbrowser.open('http://127.0.0.1:8766/')
