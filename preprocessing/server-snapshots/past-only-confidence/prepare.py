from common import *
import numpy as np,shutil,importlib.util,time
start=time.monotonic();assert not (R/'registration.json').exists();a=json.loads((F/'receipts/audit.json').read_text());assert a['status']=='PASS_CAUSAL1000_INPUT_POSES_AND_FAILURE_LEDGER' and a['GT_forbidden_opens']==0
assert sha(F/'output/external_causal_P2_c2w.txt')==a['primary_sha256'];poses=np.loadtxt(F/'output/external_causal_P2_c2w.txt');assert poses.shape==(1000,12) and np.isfinite(poses).all()
for name in ['provenance','causal_pose_records','receipts']:(R/name).mkdir()
for filename in ['confidence.py','left.py','right.py','generate_cross_inputs.py']:shutil.copy2(P/'pipeline'/filename,R/'provenance'/('legacy_'+filename))
shutil.copy2(P/'input_view/calib.txt',R/'calib.txt');shutil.copy2(F/'receipts/audit.json',R/'provenance/frontend_audit.json');shutil.copy2(F/'registration.json',R/'provenance/frontend_registration.json')
spec=importlib.util.spec_from_file_location('legacy',R/'provenance/legacy_confidence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);calib=m.parse_calibration(R/'calib.txt')
old=json.loads((P/'input_registration.json').read_text());assert abs(calib['fx']*calib['baseline_m']-old['projection_beta'])<1e-9
rows=[]
for i,pose in enumerate(poses):
 T=np.eye(4);T[:3]=pose.reshape(3,4);assert np.max(np.abs(T@np.linalg.inv(T)-np.eye(4)))<1e-9;path=R/f'causal_pose_records/{i:06d}.npy';np.save(path,T,allow_pickle=False)
 paths=dict(left=P/f'input_view/image_2/{i:06d}.png',right=P/f'input_view/image_3/{i:06d}.png',depth_left=P/f'raw_left/{i:06d}_depth.npy',depth_right=P/f'raw_right/{i:06d}_depth.npy',pose=path)
 rows.append(dict(index=i,source_frame=i+3000,inputs={k:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for k,p in paths.items()}))
 assert time.monotonic()-start<600
 if i%100==0:print('REGISTER',i,flush=True)
sources={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.suffix in ['.py','.md']};dump(R/'registration.json',dict(status='REGISTERED_PAST_ONLY_CONFIDENCE_NOT_EXECUTED',at=now(),window=old['window'],shape=old['shape'],K=old['K'],calibration=calib,calib_sha256=sha(R/'calib.txt'),frontend_primary_sha256=a['primary_sha256'],source_files=sources,rows=rows,first_frame_temporal_multiplier=1.,temporal_neighbors='previous only',min_depth=.5,max_depth=40.,seed_threshold=.8,loss_threshold=.2,GT_runtime=False,quality_adoption=False));print('REGISTERED',len(rows),flush=True)
