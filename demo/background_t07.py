"""Short-lived Windows bootstrap; only the detached local server remains."""
from pathlib import Path
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import webbrowser

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / '运行日志'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def find_server(first_port):
    for port in range(first_port, min(first_port + 21, 65536)):
        base = f'http://127.0.0.1:{port}'
        try:
            with OPENER.open(base + '/api/status', timeout=0.15) as response:
                state = json.load(response)
            if state.get('app') == 't07-viewer' and state.get('root') == str(ROOT.resolve()) and not state.get('stopping'):
                return base, state
        except (OSError, ValueError):
            continue
    return None


def notify_launch(base):
    request = urllib.request.Request(base + '/api/launch', data=b'{}',
        headers={'Content-Type':'application/json', 'Origin':base})
    try:
        with OPENER.open(request, timeout=2) as response: response.read()
    except urllib.error.HTTPError as error:
        if error.code != 404: raise


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8911)
    parser.add_argument('--no-browser', action='store_true')
    args, extra = parser.parse_known_args(arguments)
    if not 1 <= args.port <= 65535: parser.error('Invalid port')
    LOGS.mkdir(exist_ok=True)
    # Serialize repeated double-clicks until the first service is ready.
    with (LOGS/'launcher.lock').open('a+b') as lock:
        lock.seek(0,2)
        if lock.tell() == 0: lock.write(b'0'); lock.flush()
        deadline = time.monotonic() + 30
        while True:
            lock.seek(0)
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1); break
            except OSError:
                if time.monotonic() >= deadline: raise RuntimeError('Another T07 launch did not finish; see runtime logs.')
                time.sleep(0.1)
        try:
            found = find_server(args.port)
            if found:
                base, state = found
                notify_launch(base)
            else:
                interpreter = Path(sys.executable)
                windowless = interpreter.with_name('pythonw.exe')
                if windowless.exists(): interpreter = windowless
                command = [str(interpreter),str(ROOT/'launch_t07.py'),'--managed','--no-browser','--port',str(args.port),*extra]
                env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
                with (LOGS/'t07-server.log').open('ab', buffering=0) as output:
                    process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                        stdout=output,stderr=subprocess.STDOUT,close_fds=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,env=env)
                deadline = time.monotonic() + 25
                while True:
                    found = find_server(args.port)
                    if found:
                        base,state = found
                        break
                    if process.poll() is not None:
                        raise RuntimeError(f'T07 service exited ({process.returncode}); see {LOGS / "t07-server.log"}')
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'T07 startup timed out; see {LOGS / "t07-server.log"}')
                    time.sleep(0.1)
            (LOGS/'last_launch.json').write_text(json.dumps(dict(url=base+'/interactive_viewer.html',
                pid=state.get('pid'),managed=state.get('managed',False),at=time.strftime('%Y-%m-%dT%H:%M:%S%z')),
                ensure_ascii=False,indent=2),encoding='utf-8')
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    url = base + '/interactive_viewer.html'
    print(f'T07: {url}',flush=True)
    if not args.no_browser: webbrowser.open(url)

