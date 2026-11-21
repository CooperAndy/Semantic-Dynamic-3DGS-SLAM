"""Bounded environment/data inventory. Does not install packages or alter old work."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/root/autodl-tmp/dg_slam_chy_20261011')
OUT = ROOT / '00_audit'
OUT.mkdir(parents=True, exist_ok=True)

def run(args, timeout=45):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as exc:
        return dict(error=str(exc))

report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
report['gpu_details'] = run(['bash', '-lc', 'command -v nvidia-smi\nls -l /usr/bin/nvidia-smi /dev/nvidia* /usr/local/cuda* 2>/dev/null\ncat /proc/driver/nvidia/version 2>/dev/null\n/usr/local/cuda/bin/nvcc --version 2>/dev/null'])
report['hardware'] = run(['bash', '-lc', 'lscpu | head -24\nfree -h\nls -ld /root/x1_monogs_venv /root/.conda/envs /root/autodl-tmp/SLAM /root/autodl-tmp/datasets 2>/dev/null'])
pythons = ['/root/miniconda3/bin/python', '/root/x1_monogs_venv/bin/python']
for parent in ['/root/venvs', '/root/.conda/envs', '/root/miniconda3/envs']:
    if Path(parent).is_dir():
        pythons += [str(p / 'bin/python') for p in Path(parent).iterdir() if p.is_dir()]
probe = r'''
import sys,json,importlib.util,importlib.metadata as m,os
names=['torch','torchvision','numpy','opencv-python','scipy','open3d','faiss-gpu','faiss-cpu','pytorch3d','lietorch','evo','ninja','PyYAML','transformers','detectron2','timm','scikit-image','einops','trimesh','plyfile','lpips','torch-scatter']
r={'executable':sys.executable,'python':sys.version,'packages':{}}
for n in names:
 try:r['packages'][n]=m.version(n)
 except m.PackageNotFoundError:r['packages'][n]=None
for n in ['droid_backends','diff_gaussian_rasterization','simple_knn','faiss','lietorch','pytorch3d','detectron2','OneFormer']:
 try:s=importlib.util.find_spec(n);r[n]=None if s is None else s.origin
 except Exception as e:r[n]=str(e)
try:
 import torch
 r['torch_runtime']={'version':torch.__version__,'built_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}
 if torch.cuda.is_available():
  r['torch_runtime']['device']=torch.cuda.get_device_name(0)
  r['torch_runtime']['capability']=torch.cuda.get_device_capability(0)
except Exception as e:r['torch_runtime']={'error':str(e)}
print(json.dumps(r))
'''
report['python_environments'] = {p: run([p, '-c', probe], 50) for p in pythons if Path(p).exists()}
scan_roots = ['/root/autodl-tmp', '/root/raft_stereo_models', '/root/.cache', '/root/venvs', '/root/x1_monogs_venv']
skip = {'.git','node_modules','__pycache__','site-packages','.vscode-server','.nv'}
dir_pattern = re.compile(r'tum|bonn|freiburg|rgbd_dataset|replica|scannet|vkitti|kitti|oneformer|droid|dg.?slam|room0', re.I)
file_pattern = re.compile(r'(^droid.*\.(pth|pt)$|oneformer|^groundtruth\.txt$|^associations?\.txt$|rgbd_dataset.*\.(tgz|gz|zip)$)',re.I)
report['data_and_code_matches'] = []
report['asset_matches'] = []
report['scan'] = []
started = time.monotonic()
for start in scan_roots:
    if not Path(start).is_dir():
        continue
    count = 0
    errors = []
    for directory, dirs, files in os.walk(start, followlinks=False, onerror=lambda e: errors.append(str(e))):
        count += 1
        depth = len(Path(directory).relative_to(start).parts)
        dirs[:] = sorted(d for d in dirs if d not in skip and not Path(directory,d).is_symlink()) if depth < 7 else []
        if dir_pattern.search(Path(directory).name):
            report['data_and_code_matches'].append({'path':directory,'children': sorted(dirs)[:30], 'files_sample':sorted(files)[:12]})
        for filename in files:
            if file_pattern.search(filename):
                p=Path(directory,filename)
                try:report['asset_matches'].append({'path':str(p),'bytes':p.stat().st_size})
                except OSError:pass
        if count>=50000 or time.monotonic()-started>90:
            report['scan'].append({'root':start,'visited_dirs':count,'truncated':True,'errors':errors[:8]})
            break
    else:
        report['scan'].append({'root':start,'visited_dirs':count,'truncated':False,'errors':errors[:8]})
report['network'] = {}
for url in ['https://github.com/fudan-zvg/DG-SLAM.git','https://raw.githubusercontent.com/fudan-zvg/DG-SLAM/main/run_tum.py','https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz']:
    report['network'][url] = run(['curl','-IL','--connect-timeout','8','--max-time','18',url],25)
(OUT/'environment_data_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'saved':str(OUT/'environment_data_inventory.json'),'environments':list(report['python_environments']),'directory_matches':len(report['data_and_code_matches']),'asset_matches':len(report['asset_matches']),'scan':report['scan']}))
