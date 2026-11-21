"""Targeted dataset checks and official source receipts; no dependency changes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT=Path('/root/autodl-tmp/dg_slam_chy_20261011')
out=ROOT/'00_audit'
source=ROOT/'01_source_env'
source.mkdir(exist_ok=True)

def run(cmd,timeout=30):
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except Exception as e:return dict(error=str(e))

r={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cuda_devices':sorted(str(p) for p in Path('/dev').glob('nvidia*')),'datasets':{}}
r['replica_room0']={}
for root in ['/root/autodl-tmp/SplaTAM/data/Replica/room0','/root/autodl-tmp/SplaTAM_e6b/data/Replica/room0']:
    p=Path(root)
    entry={'exists':p.exists(),'resolved':str(p.resolve()),'is_symlink':p.is_symlink()}
    if p.exists():
        result=p/'results'
        fs=sorted(result.iterdir()) if result.is_dir() else []
        entry['rgb_count']=sum(f.name.startswith('frame') for f in fs)
        entry['depth_count']=sum(f.name.startswith('depth') for f in fs)
        traj=p/'traj.txt'
        if traj.is_file():
            entry['trajectory_lines']=sum(bool(l.strip()) for l in traj.open())
            entry['trajectory_sha256']=hashlib.sha256(traj.read_bytes()).hexdigest()
        entry['sample_files']=[{'name':f.name,'bytes':f.stat().st_size} for f in fs[:4]]
    r['replica_room0'][root]=entry
started=time.monotonic()
sequences=[]
scanerrors=[]
count=0
for directory,dirs,files in os.walk('/root/autodl-tmp',followlinks=False,onerror=lambda e:scanerrors.append(str(e))):
    count+=1
    depth=len(Path(directory).relative_to('/root/autodl-tmp').parts)
    dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','site-packages','node_modules'} and not Path(directory,d).is_symlink()] if depth<10 else []
    if 'rgb.txt' in files or 'depth.txt' in files:
        p=Path(directory)
        sequences.append({'path':directory,'rgb_txt':'rgb.txt' in files,'depth_txt':'depth.txt' in files,'gt':'groundtruth.txt' in files or 'pose.txt' in files,'mask_directory':(p/'seg_mask').is_dir()})
    if count>=60000 or time.monotonic()-started>60:break
r['tum_format_candidates']=sequences
r['scan']={'root':'/root/autodl-tmp','max_depth':10,'follow_symlinks':False,'visited_dirs':count,'time_seconds':time.monotonic()-started,'errors':scanerrors[:10]}
api=source/'official_commit_api.json'
r['commit_api']=run(['curl','-fsSL','--connect-timeout','6','--max-time','15','https://api.github.com/repos/fudan-zvg/DG-SLAM/commits/main','-o',str(api)],20)
commit=None
if r['commit_api'].get('returncode')==0:
    try:commit=json.loads(api.read_text())['sha']
    except Exception as e:r['commit_parse_error']=str(e)
r['official_commit']=commit
revision=commit or 'main'
receipt=[]
for filename in ['README.md','environment.yaml','run_tum.py','.gitmodules','dg_slam/gs_tracking_mapping.py','dg_slam/gaussian/gaussian_render.py','dg_slam/gaussian_model.py','configs/dg_slam.yaml','configs/TUM_RGBD/fr3_walk_xyz.yaml','scripts/download_tum.sh']:
    target=source/'official_reference'/filename
    target.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/fudan-zvg/DG-SLAM/{revision}/{filename}'
    item={'relative_path':filename,'url':url,'result':run(['curl','-fsSL','--connect-timeout','5','--max-time','12',url,'-o',str(target)],17)}
    if item['result'].get('returncode')==0:
        item['sha256']=hashlib.sha256(target.read_bytes()).hexdigest();item['bytes']=target.stat().st_size
    receipt.append(item)
r['source_files']=receipt
r['source_snapshot_complete_repository']=False
(out/'targeted_data_source_inventory.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps({'saved':str(out/'targeted_data_source_inventory.json'),'cuda_devices':r['cuda_devices'],'tum_format_candidates':sequences,'replica_room0':r['replica_room0'],'official_commit':commit,'source_files_ok':sum(x['result'].get('returncode')==0 for x in receipt)}))
