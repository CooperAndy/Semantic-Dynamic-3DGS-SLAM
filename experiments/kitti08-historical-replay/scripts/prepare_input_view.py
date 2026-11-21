"""Prepare immutable08 input view and inherited front-end recipes; CPU only."""
from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np
from PIL import Image
B=Path('/root/autodl-tmp');R=Path('/root/autodl-tmp/stage3_g14_20261027_p30915/data_pipeline');assert R.exists()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=B/'slam_project/data/kitti/dataset/sequences/08';view=R/'input_view'
for cam in ['image_2','image_3']:(view/cam).mkdir()
rows=[];shape=None
for local,i in enumerate(range(3000,4000)):
 row=dict(frame_id=local,source_frame_id=i)
 for cam in ['image_2','image_3']:
  p=source/cam/f'{i:06d}.png';assert p.is_file()
  with Image.open(p) as im:
   hw=(im.height,im.width);shape=hw if shape is None else shape;assert hw==shape
  (view/cam/f'{local:06d}.png').symlink_to(p)
  row[cam]=dict(source=str(p),sha256=sha(p))
 rows.append(row)
shutil.copy2(source/'calib.txt',view/'calib.txt')
times=np.loadtxt(source/'times.txt')[3000:4000];assert len(times)==1000 and np.all(np.diff(times)>0)
np.savetxt(view/'times.txt',times-times[0],fmt='%.9f')
for alias,cam in [('image_0','image_2'),('image_1','image_3')]:(view/alias).symlink_to(view/cam,target_is_directory=True)
P={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ').reshape(3,4) for s in (view/'calib.txt').read_text().splitlines() if s.startswith(('P2:','P3:'))}
K=P['P2'][:,:3];baseline=abs(P['P2'][0,3]/K[0,0]-P['P3'][0,3]/P['P3'][0,0]);beta=K[0,0]*baseline
old=B/'t04_chy_kitti00_200_20261012/orb_matched_v2';prep=json.loads((old/'preparation.json').read_text())
binary=old/'stereo_kitti_headless';assert sha(binary)=='df94fa506f2bd13c0697143449ef560cb0ada307ae232992b83b591f61fc95b8'
shutil.copy2(binary,R/'pipeline/stereo_kitti_headless')
vocab=Path(next(k for k in prep['hashes'] if k.endswith('/ORBvoc.txt')));assert sha(vocab)==prep['hashes'][str(vocab)]
repair=prep['compatibility_repair'];assert sha(Path(repair['library']))==repair['sha256']
libs=[str(Path(repair['library']).parent)]+[x[2:] for x in prep['build_command'] if x.startswith('-L')]
text=(old/'settings.yaml').read_text();values={'Camera1.fx':K[0,0],'Camera1.fy':K[1,1],'Camera1.cx':K[0,2],'Camera1.cy':K[1,2],
 'Camera.width':shape[1],'Camera.height':shape[0],'Camera.fps':10,'Camera.RGB':0,'Stereo.b':baseline,'ORBextractor.nFeatures':2000}
out=[]
for line in text.splitlines():
 key=line.split(':',1)[0]
 out.append(f'{key}: {values[key]}' if key in values else line)
(R/'orb/settings.yaml').write_text('\n'.join(out)+'\n')
sources={
 'left.py':B/'slam_project/stage1_driving_adaptation/scripts/generate_raft_depth_kitti.py',
 'right.py':B/'slam_project/stage1r_confidence_mapping/scripts/generate_raft_reverse_depth_kitti.py',
 'confidence.py':B/'slam_project/stage1r_confidence_mapping/scripts/build_depth_confidence_kitti.py',
}
source_locks=[]
for name,p in sources.items():
 s=p.read_text();original=sha(p)
 # Only captions and unused disparity-file output differ from upstream recipes.
 s=s.replace('KITTI 00 frame','KITTI08 local frame').replace('KITTI seq00 frames 000000-000199','KITTI08 source3000-3999, local0-999')
 if name=='right.py':s='\n'.join(l for l in s.splitlines() if 'np.save(args.output_dir / f"{reference.stem}_disp.npy"' not in l)+'\n'
 q=R/'pipeline'/name;q.write_text(s);compile(s,str(q),'exec')
 source_locks.append(dict(source=str(p),original_sha256=original,copied_path=str(q),copied_sha256=sha(q)))
checkpoint=Path('/root/raft_stereo_models/raftstereo-middlebury.pth');assert checkpoint.is_file()
reg=dict(task_id='T3-G14-INPUT',state='REGISTERED_INPUT_PREPARATION_NOT_EXECUTED',at=datetime.datetime.now().astimezone().isoformat(),
 candidate_entry_gate_waived=True,window={'sequence':'08','source_start':3000,'frame_count':1000,'exposure':'EXPOSED_CROSS_WINDOW_DEV'},
 shape=list(shape),K=K.tolist(),projection_beta=float(beta),equivalent_baseline_m=float(baseline),
 source_calib_sha256=sha(source/'calib.txt'),source_times_sha256=sha(source/'times.txt'),rows=rows,
 binary_sha256=sha(R/'pipeline/stereo_kitti_headless'),vocabulary=str(vocab),vocabulary_sha256=sha(vocab),library_paths=libs,
 pipeline_sources=source_locks,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),raft_root=str(B/'slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo'),
 inference_iterations=32,raw_depth_range_m=[.5,80],confidence_valid_range_m=[.5,40],confidence_threshold=.8,
 input_preparation_budget_seconds=2400,output_scope='Private symlink view and new outputs, no old experiment resumed')
(R/'input_registration.json').write_text(json.dumps(reg,indent=2));print(json.dumps({k:v for k,v in reg.items() if k not in ['rows','pipeline_sources']}))
