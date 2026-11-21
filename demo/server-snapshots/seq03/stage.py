"""Stage and hash seq03 image window and code without using the GPU."""
from pathlib import Path
import hashlib,json,shutil,zipfile,re,sys
from PIL import Image
P=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116');ZIP=Path('/root/autodl-tmp/slam_project/data/kitti/data_odometry_color.zip');OLD=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499');S23=Path('/root/autodl-tmp/s23_effects_20261105_p16247');BATCH=OLD.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 if (P/'staging_receipt.json').exists():raise FileExistsError('Already staged')
 assert shutil.disk_usage(P).free>18*2**30
 seq=P/'sequences/03';seq.mkdir(parents=True);records=[]
 with zipfile.ZipFile(ZIP) as z:
  for cam in (2,3):
   folder=seq/f'image_{cam}';folder.mkdir()
   for i in range(801):
    source=f'dataset/sequences/03/image_{cam}/{i:06}.png';target=folder/f'{i:06}.png'
    with z.open(source) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
    with Image.open(target) as im:
     if i==0 and cam==2:width,height=im.size
     assert im.size==(width,height);im.verify()
    records.append(dict(local=i,source=i,camera=cam,sha256=sha(target)))
   print('STAGED CAMERA',cam,flush=True)
  for name in ('calib.txt','times.txt'):(seq/name).write_bytes(z.read(f'dataset/sequences/03/{name}'))
 assert len((seq/'times.txt').read_text().splitlines())==801 and len(records)==1602
 (P/'input_manifest.json').write_text(json.dumps(records,indent=2));job=dict(sequence='03',start=0,end=800,count=801,width=width,height=height,job='seq03_000_800_queue',remote_root=str(P));(P/'input_spec.json').write_text(json.dumps(job,indent=2))
 scripts=P/'scripts';scripts.mkdir()
 for src in (OLD/'scripts').glob('*.py'):
  text=src.read_text().replace(str(OLD),str(P)).replace('sequences/09','sequences/03');text=re.sub(r'\b1226\b',str(width),text);text=re.sub(r'\b370\b',str(height),text);(scripts/src.name).write_text(text)
 shutil.copytree(OLD/'vendor',P/'vendor');shutil.copytree(S23/'candidate_v4_fp64accum',P/'candidate',ignore=shutil.ignore_patterns('__pycache__'))
 # Existing environment binaries and dependencies remain on the server; their
 # hashes are pinned before the queue is allowed to execute.
 dependencies=[BATCH/'server_prepare.py',S23/'run_registered_matrix.py',S23/'experiment.py',S23/'technical25/source/configs/kitti/TECH_TMD25.py',S23/'raster_fp64_accum_v1/diff_gaussian_rasterization/__init__.py',Path('/root/raft_stereo_models/raftstereo-middlebury.pth'),BATCH/'orb_cpu_source/Examples/Stereo/stereo_kitti',Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2/ORB_SLAM3_4452a3c4/Vocabulary/ORBvoc.txt')]
 for p in dependencies:assert p.is_file(),str(p)
 for folder in (P/'candidate',scripts,Path('/root/autodl-tmp/slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo/core')):
  dependencies.extend(p for p in folder.rglob('*.py') if p.is_file() and '__pycache__' not in p.parts)
 dependencies.extend(p for p in (S23/'raster_fp64_accum_v1').rglob('*.so') if p.is_file())
 locks={str(p):sha(p) for p in dependencies};(P/'dependency_sha256.json').write_text(json.dumps(locks,indent=2))
 status=dict(status='PASS',frames=801,stereo_images=1602,image_dimensions=[width,height],image_manifest_sha256=sha(P/'input_manifest.json'),calib_sha256=sha(seq/'calib.txt'),times_sha256=sha(seq/'times.txt'),dependency_count=len(locks),dependency_manifest_sha256=sha(P/'dependency_sha256.json'),zip_source=str(ZIP),zip_bytes=ZIP.stat().st_size,GT_runtime=False)
 (P/'staging_receipt.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if __name__=='__main__':main()
