"""Download and verify exactly one TUM sequence, using CPU only. No inference."""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path('/root/autodl-tmp/dg_slam_chy_20261011')
DATA = ROOT / '02_data_masks'
TUM = DATA / 'TUM'
ARCHIVES = DATA / 'archives'
NAME = 'rgbd_dataset_freiburg3_walking_xyz'
URL = 'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/' + NAME + '.tgz'
EXPECTED_BYTES = 527550055
for p in [DATA, TUM, ARCHIVES]:
    p.mkdir(parents=True, exist_ok=True)
STATUS = DATA / 'preparation_status.json'
started = time.time()

def status(stage, **kwargs):
    item = {'stage': stage, 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2))
    tmp.replace(STATUS)
    print(json.dumps(item), flush=True)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def table(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]

try:
    archive = ARCHIVES / (NAME + '.tgz')
    part = archive.with_suffix('.tgz.part')
    free_before = shutil.disk_usage(ROOT).free
    if free_before < 3 * 1024**3:
        raise RuntimeError(f'Insufficient free space for one sequence: {free_before}')
    if not archive.exists():
        status('downloading', url=URL, expected_bytes=EXPECTED_BYTES, free_bytes=free_before)
        with (DATA / 'download.log').open('a') as log:
            result = subprocess.run(['curl', '-fL', '--retry', '4', '--retry-delay', '3', '--connect-timeout', '20', '--max-time', '1800', '-C', '-', '--dump-header', str(DATA/'download_headers.txt'), URL, '-o', str(part)], stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError(f'curl failed: {result.returncode}; partial download retained')
        if part.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f'Content length mismatch: {part.stat().st_size} != {EXPECTED_BYTES}')
        part.replace(archive)
    if archive.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError('Existing archive has unexpected size')
    status('verifying_archive', archive=str(archive))
    subprocess.run(['gzip', '-t', str(archive)], check=True)
    archive_hash = digest(archive)
    dataset = TUM / NAME
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        unpacked_bytes = sum(m.size for m in members if m.isfile())
        for m in members:
            relative = Path(m.name)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != NAME or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise RuntimeError(f'Unexpected archive member: {m.name}')
        if not dataset.exists():
            if shutil.disk_usage(ROOT).free < unpacked_bytes + 512*1024**2:
                raise RuntimeError('Insufficient extraction headroom')
            status('extracting', unpacked_file_bytes=unpacked_bytes, members=len(members))
            tf.extractall(TUM, filter='data')
    status('verifying_dataset')
    from PIL import Image
    import numpy as np
    rgb = table(dataset/'rgb.txt')
    depth = table(dataset/'depth.txt')
    gt = table(dataset/'groundtruth.txt')
    assert rgb and depth and gt
    for rows in [rgb, depth, gt]:
        ts = [float(row[0]) for row in rows]
        assert all(math.isfinite(x) for x in ts)
        assert all(b > a for a,b in zip(ts,ts[1:])), 'Timestamps not strictly increasing'
    for row in gt:
        values = [float(x) for x in row[1:]]
        assert len(values)==7 and all(math.isfinite(x) for x in values)
        assert abs(sum(x*x for x in values[3:])-1.0)<0.01
    all_files = []
    dimensions = {}
    modes = {}
    for kind, rows in [('rgb',rgb),('depth',depth)]:
        for row in rows:
            path = dataset/row[1]
            with Image.open(path) as im:
                dimensions[str(im.size)] = dimensions.get(str(im.size),0)+1
                modes[kind+':'+im.mode] = modes.get(kind+':'+im.mode,0)+1
                im.verify()
            all_files.append({'path':row[1], 'bytes':path.stat().st_size,'sha256':digest(path)})
    depth_times = np.array([float(x[0]) for x in depth])
    gt_times = np.array([float(x[0]) for x in gt])
    matches=[]
    for i,row in enumerate(rgb):
        t=float(row[0]);j=int(np.argmin(np.abs(depth_times-t)));k=int(np.argmin(np.abs(gt_times-t)))
        if abs(depth_times[j]-t)<0.08 and abs(gt_times[k]-t)<0.08:
            matches.append({'rgb_index':i,'depth_index':j,'gt_index':k,'rgb_timestamp':t,'rgb_path':row[1],'depth_path':depth[j][1],'gt_timestamp':float(gt[k][0]),'depth_dt_seconds':float(abs(depth_times[j]-t)),'gt_dt_seconds':float(abs(gt_times[k]-t))})
    selected=[]
    for row in matches:
        if not selected or row['rgb_timestamp']-selected[-1]['rgb_timestamp']>1.0/32:
            selected.append(row)
    assert len(selected)>=200, 'Not enough frames for requested smoke and small reproduction'
    for name, rows in [('associations_all_valid',matches),('official_loader_selected',selected),('smoke_first30',selected[:30]),('dev_first200',selected[:200])]:
        with (DATA/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    alias=TUM/'fr3_walk_xyz'
    if not alias.exists() and not alias.is_symlink():
        alias.symlink_to(NAME,target_is_directory=True)
    assert alias.resolve()==dataset.resolve(), 'Existing alias points elsewhere'
    for filename in ['rgb.txt','depth.txt','groundtruth.txt','accelerometer.txt']:
        path=dataset/filename
        if path.exists():all_files.append({'path':filename,'bytes':path.stat().st_size,'sha256':digest(path)})
    (DATA/'dataset_file_hashes.json').write_text(json.dumps(all_files,indent=2))
    report={'status':'DATASET_READY_MASKS_PENDING','sequence':NAME,'official_url':URL,'dataset_root':str(dataset),'optional_legacy_default_alias':str(alias),'official_config':'configs/TUM_RGBD/'+NAME+'.yaml','archive_bytes':archive.stat().st_size,'archive_sha256':archive_hash,'archive_gzip_integrity':'PASS','published_checksum_comparison':'NOT_AVAILABLE; SHA256 recorded locally; exact HTTP content length and gzip integrity checked','unpacked_file_bytes':unpacked_bytes,'rgb_count':len(rgb),'depth_count':len(depth),'gt_pose_count':len(gt),'all_indexed_png_verified':True,'dimensions':dimensions,'modes':modes,'association_max_dt_seconds':0.08,'official_loader_frame_rate':32,'association_count':len(matches),'official_loader_selected_count':len(selected),'max_depth_dt_seconds':max(x['depth_dt_seconds'] for x in selected),'max_gt_dt_seconds':max(x['gt_dt_seconds'] for x in selected),'raw_depth_units_per_metre':5000,'prediction_masks':'NOT_GENERATED; no fake empty masks created','gpu_used':False,'dependencies_installed':False,'free_bytes_before':free_before,'free_bytes_after':shutil.disk_usage(ROOT).free,'elapsed_seconds':time.time()-started}
    report['official_sequence_index']=1
    (DATA/'dataset_receipt.json').write_text(json.dumps(report,indent=2))
    status('complete',**report)
except Exception as exc:
    status('failed',error=str(exc))
    raise
