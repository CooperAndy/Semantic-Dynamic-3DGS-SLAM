from __future__ import annotations
import argparse, base64, hashlib, http.server, json, struct, threading, uuid, webbrowser
import urllib.request
import socket, re, os, subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, quote, urlsplit, parse_qs
import sys

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / '源码'
sys.path.insert(0, str(CODE))
import reference_eval
from viewer_lifecycle import ViewerLifecycle
CONFIG = json.loads((ROOT / '配置/原地图引用.json').read_text(encoding='utf-8'))
ASSETS = Path(CONFIG['asset_root']).resolve()
CAMERA_PICKER_LOCK = threading.Lock()
EXPORT_FOLDER_LOCK = threading.Lock()
CAPTURE_FOLDER_PATTERN = re.compile(
    r'(?P<day>\d{8})\(seq(?P<sequence>\d{2,})， (?P<start>\d+)-(?P<end>\d+)\)_(?P<number>\d{3,})')

def pick_camera_json():
    if os.name != 'nt':
        raise ValueError('The local camera file picker requires Windows.')
    if not CAMERA_PICKER_LOCK.acquire(blocking=False):
        raise ValueError('A camera file selection window is already open.')
    try:
        folder = (ROOT / '导出图片').resolve()
        folder.mkdir(parents=True, exist_ok=True)
        python = Path(sys.executable)
        if python.name.lower() == 'pythonw.exe':
            python = python.with_name('python.exe')
        result = subprocess.run([str(python), str(CODE / 'pick_camera_json.py'), str(folder)],
                                capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise ValueError('Could not open the local camera file selection window.')
        selected = json.loads(result.stdout)['path']
        if not selected:
            return {'cancelled': True}
        camera = Path(selected)
        if camera.suffix.lower() != '.json' or not camera.is_file() or camera.stat().st_size > 100000:
            raise ValueError('Choose a camera JSON file smaller than 100 KB.')
        meta = json.loads(camera.read_text(encoding='utf-8-sig'))
        if not isinstance(meta, dict) or meta.get('schema') != 't07-camera-v1':
            raise ValueError('Choose a saved camera.json file, rather than a SHA256 manifest.')
        return {'cancelled': False, 'name': camera.name, 'metadata': meta}
    finally:
        CAMERA_PICKER_LOCK.release()

def capture_scope(scene_id):
    catalog = json.loads((CODE / 'scene-catalog.json').read_text(encoding='utf-8'))
    scene = next((item for item in catalog['scenes'] if item['id'] == scene_id), None)
    if scene is None:
        raise ValueError('The capture scene is not in the catalog.')
    config = json.loads((CODE / scene['config']).read_text(encoding='utf-8'))
    sequence = re.fullmatch(r'kitti(\d+)(?:_.*)?', scene_id)
    frames = config.get('sourceFrames')
    if config.get('sceneId') != scene_id or sequence is None or not isinstance(frames, list) or len(frames) != 2 or not all(type(n) is int for n in frames) or not 0 <= frames[0] <= frames[1]:
        raise ValueError('Invalid capture scene or frame range.')
    return {'sceneId': scene_id, 'sequence': sequence[1].zfill(2), 'sourceFrames': frames}


def reserve_export_folder(parent, scope, day=None):
    """Use one daily counter across scenes, serialized across request threads."""
    day = day or datetime.now().strftime('%Y%m%d')
    pattern = re.compile(rf'{re.escape(day)}(?:\(seq\d{{2,}}， \d+-\d+\))?_(\d{{3,}})')
    start, end = scope['sourceFrames']
    with EXPORT_FOLDER_LOCK:
        parent.mkdir(parents=True, exist_ok=True)
        number = max((int(match.group(1)) for entry in parent.iterdir()
                      if entry.is_dir() and (match := pattern.fullmatch(entry.name))), default=0) + 1
        while True:
            folder = parent / f'{day}(seq{scope["sequence"]}， {start}-{end})_{number:03d}'
            try:
                folder.mkdir(exist_ok=False)
                return folder
            except FileExistsError:
                number += 1

def saved_captures(scene_id, source_frames=None):
    """List this Demo's own exports, without using the browser's remembered folder."""
    scope = capture_scope(scene_id)
    if source_frames is not None and source_frames != scope['sourceFrames']:
        raise ValueError('The scene frame range changed. Refresh the Demo.')
    parent = (ROOT / '导出图片').resolve()
    captures = []
    for folder in parent.iterdir() if parent.is_dir() else []:
        match = CAPTURE_FOLDER_PATTERN.fullmatch(folder.name)
        if not match or not folder.is_dir():
            continue
        if match['sequence'] != scope['sequence'] or [int(match['start']), int(match['end'])] != scope['sourceFrames']:
            continue
        camera = folder / 'camera.json'
        try:
            if not camera.resolve().is_relative_to(parent) or camera.stat().st_size > 100000:
                continue
            meta = json.loads(camera.read_text(encoding='utf-8'))
            if meta.get('schema') != 't07-camera-v1' or any(meta.get(key) != value for key, value in scope.items()):
                continue
            preview = folder / 'render.png'
            prefix = f'/outputs/{quote(folder.name, safe="")}/'
            captures.append({'name': folder.name, 'capturedAt': meta.get('capturedAt'),
                             'mode': meta.get('mode', 'rgb'),
                             'cameraUrl': prefix + 'camera.json',
                             'previewUrl': prefix + 'render.png'
                             if preview.is_file() and preview.resolve().is_relative_to(parent) else None})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    captures.sort(key=lambda item: (int(CAPTURE_FOLDER_PATTERN.fullmatch(item['name'])['day']),
                                   int(CAPTURE_FOLDER_PATTERN.fullmatch(item['name'])['number'])), reverse=True)
    return {'root': str(parent), **scope, 'captures': captures}

class T07Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = False
    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.ply': 'application/octet-stream', '.json': 'application/json; charset=utf-8'}
    reference_video_path = re.compile(r'/scene-data/reference-video/[a-z0-9_-]+\.mp4')
    def translate_path(self, path):
        rel = unquote(urlsplit(path).path).lstrip('/')
        if rel.startswith('scene-data/'):
            root, rel = ASSETS, rel[len('scene-data/'):]
        elif rel.startswith('outputs/'):
            root, rel = ROOT / '导出图片', rel[len('outputs/'):]
        else:
            root = CODE
        root = root.resolve()
        target = (root / rel).resolve()
        if not target.is_relative_to(root):
            return str(CODE / '__denied__')
        return str(target)
    def list_directory(self, path):
        self.send_error(403, 'Directory listing disabled')
        return None
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()
    def reply(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def serve_reference_video(self):
        try:
            video = open(self.translate_path(self.path), 'rb')
        except OSError:
            self.send_error(404, 'Reference video unavailable')
            return
        with video:
            size = os.fstat(video.fileno()).st_size
            start, end = 0, size - 1
            byte_range = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
            partial = byte_range and any(byte_range.groups())
            if partial:
                first, last = byte_range.groups()
                if first:
                    start = int(first)
                    if last:
                        end = min(end, int(last))
                else:
                    start = max(0, size - int(last))
                if start > end:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
            remaining = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(remaining))
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            video.seek(start)
            try:
                while remaining and (chunk := video.read(min(remaining, 256 * 1024))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except ConnectionError:
                return
    def do_HEAD(self):
        if self.reference_video_path.fullmatch(urlsplit(self.path).path):
            return self.serve_reference_video()
        return super().do_HEAD()
    def do_GET(self):
        if self.reference_video_path.fullmatch(urlsplit(self.path).path):
            return self.serve_reference_video()
        if urlsplit(self.path).path == '/api/captures':
            try:
                query = parse_qs(urlsplit(self.path).query)
                frames = [int(query['start'][0]), int(query['end'][0])] if 'start' in query or 'end' in query else None
                return self.reply(200, saved_captures(query.get('scene', [''])[0], frames))
            except (ValueError, KeyError, OSError) as error:
                return self.reply(400, {'error': str(error)})
        if urlsplit(self.path).path == '/api/status':
            lifecycle = getattr(self.server, 'viewer_lifecycle', None)
            return self.reply(200, {'app': 't07-viewer', 'root': str(ROOT.resolve()),
                                   'pid': os.getpid(),
                                   **(lifecycle.status() if lifecycle else {'managed': False})})
        if urlsplit(self.path).path == '/api/reference':
            try:
                query = parse_qs(urlsplit(self.path).query)
                return self.reply(200, reference_eval.reference(query['scene'][0], int(query['frame'][0])))
            except (ValueError, KeyError, OSError) as error:
                return self.reply(400, {'error': str(error)})
        return super().do_GET()
    def do_POST(self):
        lifecycle = getattr(self.server, 'viewer_lifecycle', None)
        if lifecycle: lifecycle.operation(1)
        try:
            if self.path not in ['/api/export', '/api/camera', '/api/evaluate', '/api/viewer-presence', '/api/launch', '/api/pick-camera']:
                return self.reply(404, {'error': 'Unknown endpoint'})
            origin = self.headers.get('Origin')
            expected = f'http://127.0.0.1:{self.server.server_port}'
            if origin != expected or self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
                return self.reply(403, {'error': 'Local same-origin requests required'})
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 40000000:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(size))
            if self.path == '/api/pick-camera':
                return self.reply(200, pick_camera_json())
            if self.path == '/api/launch':
                if lifecycle: lifecycle.launch()
                return self.reply(200, {'ok': True})
            if self.path == '/api/viewer-presence':
                if not isinstance(data, dict) or not isinstance(data.get('client'), str) or not re.fullmatch(r'[a-zA-Z0-9_-]{16,100}', data['client']) or type(data.get('alive')) is not bool or type(data.get('sequence')) is not int or not 0 <= data['sequence'] <= 2**53:
                    raise ValueError('Invalid Viewer presence')
                if lifecycle: lifecycle.presence(data['client'], data['alive'], data['sequence'])
                return self.reply(200, {'ok': True})
            if self.path == '/api/evaluate':
                return self.reply(200, reference_eval.evaluate(data))
            meta = data['metadata']
            if meta.get('schema') != 't07-camera-v1' or not isinstance(meta.get('camera'), dict):
                raise ValueError('Invalid camera metadata')
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
            png = None
            if self.path == '/api/export':
                scope = capture_scope(meta['sceneId'])
                if meta.get('sourceFrames', scope['sourceFrames']) != scope['sourceFrames']:
                    raise ValueError('The scene frame range changed. Refresh the Demo before saving.')
                meta.update(scope)
                png = base64.b64decode(data['png'], validate=True)
                if len(png) < 33 or png[:8] != b'\x89PNG\r\n\x1a\n':
                    raise ValueError('Invalid PNG')
                width, height = struct.unpack('>II', png[16:24])
                if width != meta['render']['width'] or height != meta['render']['height']:
                    raise ValueError('PNG dimensions differ from metadata')
                folder = reserve_export_folder(ROOT / '导出图片', scope)
            else:
                folder = ROOT / '相机参数' / stamp
                folder.mkdir(exist_ok=False)
            # These files are always new; old source assets are never a write target.
            meta['sourceAssetRoot'] = str(ASSETS)
            asset_url = meta.get('assetUrl', '')
            asset_rel = unquote(urlsplit(asset_url).path)
            if asset_rel.startswith('/scene-data/'):
                asset = (ASSETS / asset_rel[len('/scene-data/'):]).resolve()
                if asset.is_relative_to(ASSETS) and asset.is_file():
                    stat = asset.stat()
                    meta['sourceAsset'] = {'path': str(asset), 'bytes': stat.st_size,
                                           'mtime_ns': stat.st_mtime_ns}
            manifest_url = meta.get('assetManifest', '')
            manifest_rel = unquote(urlsplit(manifest_url).path)
            if manifest_rel.startswith('/scene-data/'):
                manifest = (ASSETS / manifest_rel[len('/scene-data/'):]).resolve()
                if manifest.is_relative_to(ASSETS) and manifest.is_file():
                    meta['sourceManifest'] = {'path': str(manifest),
                        'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
            if png:
                (folder / 'render.png').write_bytes(png)
                meta['png_sha256'] = hashlib.sha256(png).hexdigest()
            (folder / 'camera.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
            (folder / 'SHA256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
            self.reply(200, {'path': str(folder), 'files': list(hashes)})
        except (ValueError, KeyError, TypeError) as error:
            self.reply(400, {'error': str(error)})
        except OSError as error:
            self.reply(500, {'error': str(error)})
        finally:
            if lifecycle: lifecycle.operation(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8911)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--managed', action='store_true', help='Stop after the last Viewer page closes')
    parser.add_argument('--close-delay', type=float, default=10.0)
    parser.add_argument('--lease-timeout', type=float, default=180.0)
    parser.add_argument('--startup-timeout', type=float, default=120.0)
    args = parser.parse_args()
    if min(args.close_delay, args.lease_timeout, args.startup_timeout) <= 0:
        parser.error('Lifecycle timeouts must be positive')
    if not ASSETS.is_dir():
        raise FileNotFoundError(f'Original map assets missing: {ASSETS}')
    # Reuse only a T07 service for this exact workspace. Other occupied ports
    # are skipped, never terminated or mistaken for the requested Viewer.
    server = None
    for port in range(args.port, min(args.port + 21, 65536)):
        url = f'http://127.0.0.1:{port}/interactive_viewer.html'
        try:
            server = T07Server(('127.0.0.1', port), Handler)
            break
        except OSError as error:
            if getattr(error, 'winerror', None) not in (10048, 10013) and error.errno not in (98, 13, 10048):
                raise
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f'http://127.0.0.1:{port}/api/status', timeout=1) as response:
                    running = json.load(response)
                if running.get('app') == 't07-viewer' and running.get('root') == str(ROOT.resolve()):
                    print(f'T07 already running: {url}', flush=True)
                    if not args.no_browser: webbrowser.open(url)
                    return
            except (OSError, ValueError):
                pass
    if server is None:
        raise RuntimeError('No available local T07 port in the configured range.')
    finished = threading.Event()
    if args.managed:
        server.viewer_lifecycle = ViewerLifecycle(close_delay=args.close_delay,
            lease_timeout=args.lease_timeout, startup_timeout=args.startup_timeout)
        threading.Thread(target=server.viewer_lifecycle.watch, args=(server, finished), daemon=True).start()
    print(f'T07: {url}', flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        finished.set()
        server.server_close()

if __name__ == '__main__': main()
