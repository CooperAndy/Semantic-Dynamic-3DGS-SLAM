"""Local, full RGB seen-view diagnostics; never an offline benchmark substitute."""
import base64
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def reference(scene, frame):
    sources = json.loads((ROOT / '配置/参考帧测评.json').read_text(encoding='utf-8'))
    if scene not in sources or type(frame) is not int:
        raise ValueError('Invalid scene or frame number.')
    source = sources[scene]
    if not source['range'][0] <= frame <= source['range'][1]:
        raise ValueError('Frame number is outside the selected scene range.')
    path = Path(source['image_root']) / source['filename'].format(frame=frame)
    if not path.is_file():
        raise ValueError('The original RGB image for this frame is missing; evaluation is unavailable.')
    with Image.open(path) as image:
        width, height = image.size
    calib = Path(source['calibration'])
    line = next(line for line in calib.read_text().splitlines() if line.startswith('P2:'))
    p = [float(x) for x in line.split()[1:]]
    is_heldout = frame in source.get('heldout_frames', [])
    protocol = ('Viewer RGB full-image / original held-out frame / map-embedded camera / not the offline common-GT metric'
                if is_heldout else 'Viewer RGB full-image / seen reference frame / not offline Test')
    return {'sceneId': scene, 'frame': frame, 'width': width, 'height': height,
            'fx': p[0], 'fy': p[5], 'cx': p[2], 'cy': p[6],
            'protocol': protocol,
            'reference_path': str(path), 'reference_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'calibration_sha256': hashlib.sha256(calib.read_bytes()).hexdigest()}


def score_rgb(pred, target):
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError('Rendered and reference RGB dimensions differ. Resizing before evaluation is not allowed.')
    mse = float(np.mean(((pred.astype(np.float64) - target.astype(np.float64)) / 255.0) ** 2))
    return mse, None if mse == 0 else -10 * math.log10(mse)


def evaluate(data):
    meta = data['metadata']
    if meta.get('cameraPathVisible'):
        raise ValueError('Turn off Show camera path before RGB reference evaluation.')
    ref = reference(meta['sceneId'], data['frame'])
    if meta.get('mode') != 'rgb' or meta.get('variant') not in ('F1', 'F200'):
        raise ValueError('PSNR evaluates the full RGB map only. Switch to RGB Map.')
    camera_ref = meta.get('camera', {}).get('reference')
    if not isinstance(camera_ref, dict) or any(camera_ref.get(k) != ref[k] for k in ('sceneId','frame','width','height','fx','fy','cx','cy')):
        raise ValueError('Align and lock a reference-frame camera before evaluation. Free views cannot be scored.')
    sources = json.loads((ROOT / '配置/参考帧测评.json').read_text(encoding='utf-8'))
    pose_path = Path(sources[meta['sceneId']]['camera_poses'])
    poses = json.loads(pose_path.read_text(encoding='utf-8'))
    pose = next(p for p in poses if p['source_frame'] == data['frame'])
    expected = np.eye(4)
    expected[:3,:3] = np.diag([-1,-1,1]) @ np.array(pose['rotation']) @ np.diag([1,-1,-1])
    expected[:3,3] = np.diag([-1,-1,1]) @ np.array(pose['position'])
    actual = np.array(meta['camera'].get('worldMatrix',[]), dtype=np.float64)
    if actual.size != 16 or not np.isfinite(actual).all() or not np.allclose(actual.reshape(4,4,order='F'),expected,rtol=0,atol=1e-4):
        raise ValueError('The rendered camera matrix does not match the reference frame.')
    raw = base64.b64decode(data['png'], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.size != (ref['width'], ref['height']):
            raise ValueError('The evaluated PNG must have the original reference-image dimensions.')
        pred = np.array(image.convert('RGB'))
    target = np.array(Image.open(ref['reference_path']).convert('RGB'))
    mse, psnr = score_rgb(pred, target)
    result = {**ref, 'variant': meta['variant'], 'mse_rgb_0_1': mse, 'psnr_db': psnr,
              'identical': mse == 0, 'measuredAt': datetime.now().astimezone().isoformat(),
              'pose_sha256': hashlib.sha256(pose_path.read_bytes()).hexdigest(),
              'camera_matrix_max_error': float(np.max(np.abs(actual.reshape(4,4,order='F')-expected))),
              'metadata': meta, 'render_sha256': hashlib.sha256(raw).hexdigest()}
    parent = ROOT / '测评结果'
    parent.mkdir(exist_ok=True)
    folder = parent / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6])
    folder.mkdir()
    (folder / 'render.png').write_bytes(raw)
    (folder / 'score.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    (folder / 'SHA256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    return {**result, 'path': str(folder)}
