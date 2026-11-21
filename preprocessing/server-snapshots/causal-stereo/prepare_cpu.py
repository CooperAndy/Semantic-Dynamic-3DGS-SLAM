from common import *
import ast,shutil,time,numpy as np
start=time.monotonic();assert not (R/'registration_cpu.json').exists();old=json.loads((P14/'input_registration.json').read_text());inputs=json.loads((I/'registration.json').read_text());assert RAFTPY.is_file() and CPU.is_file();(R/'runtime').mkdir();(R/'source/RAFT-Stereo').mkdir(parents=True);(R/'provenance').mkdir();(R/'templates').mkdir()
checkpoint=Path(old['checkpoint']);assert sha(checkpoint)==old['checkpoint_sha256'];shutil.copy2(checkpoint,R/'runtime/raftstereo-middlebury.pth');assert sha(R/'runtime/raftstereo-middlebury.pth')==old['checkpoint_sha256'];root=Path(old['raft_root']);core=[]
for x in old['raft_core_file_locks']:
 p=Path(x['path']);assert sha(p)==x['sha256'];target=R/'source/RAFT-Stereo'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);core.append(dict(path=target.relative_to(R).as_posix(),bytes=target.stat().st_size,sha256=sha(target)))
for name in ['LICENSE','LICENSE.txt']:
 p=root/name
 if p.is_file():shutil.copy2(p,R/'provenance'/name)
for name in ['left.py','right.py','confidence.py']:
 src=P14/'pipeline'/name;dst=R/'source'/name;shutil.copy2(src,dst);assert sha(src)==sha(dst);ast.parse(dst.read_bytes())
for srcname,name in [('producer.py','producer.py'),('contract_v2.py','contract.py'),('audit.py','audit_parent.py')]:
 src=C/srcname;dst=R/'templates'/name;shutil.copy2(src,dst);assert sha(src)==sha(dst);ast.parse(dst.read_bytes())
shutil.copy2(C/'receipts/audit.json',R/'provenance/G19C_audit.json');shutil.copy2(I/'registration.json',R/'provenance/G21_frontend_registration.json');windows=[]
for win in inputs['windows']:
 wid=win['id'];W=I/'windows'/wid;a=json.loads((W/'audit.json').read_text());assert a['status']=='PASS_CAUSAL1000_FRONTEND_INPUT_AND_LEDGER' and a['GT_forbidden_opens']==0;primary=W/'output/external_causal_P2_c2w.txt';assert sha(primary)==a['primary_sha256'];poses=np.loadtxt(primary);assert poses.shape==(1000,12) and np.isfinite(poses).all();assert sha(W/'input_view/calib.txt')==win['calib_sha256'] and sha(W/'input_view/times.txt')==win['times_sha256']
 for x in win['inputs']:assert sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes']
 windows.append(dict(id=wid,sequence=win['sequence'],source_start=win['source_start'],frame_count=1000,shape=win['shape'],K=win['K'],projection_beta=win['projection_beta'],input_view=str(W/'input_view'),input_registration_sha256=sha(W/'registration.json'),frontend_audit_sha256=sha(W/'audit.json'),primary_pose_path=str(primary),primary_pose_sha256=a['primary_sha256'],GT_quality_consulted=False))
for p in R.rglob('*.py'):ast.parse(p.read_bytes())
files={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file()};reg=dict(status='PASS_CPU_STEREO_PIPELINE_SOURCE_WEIGHT_AND4000_IMAGE_LOCKS_PENDING_GPU',at=now(),windows=windows,source_files=files,raft_core_files=core,checkpoint_sha256=old['checkpoint_sha256'],runtime_python=str(RAFTPY),model_parameters='ParentG14 unchanged;32iters,mixed_precision,corr_alt',confidence_formula_source_sha256=sha(R/'templates/producer.py'),depth_range_m=[.5,80],confidence_selection_range_m=[.5,40],seed_threshold=.8,loss_threshold=.2,seconds=time.monotonic()-start,GPU_calls=0,GT_scoring=False,quality_adoption=False);dump(R/'registration_cpu.json',reg);print(json.dumps({k:v for k,v in reg.items() if k not in ['source_files','raft_core_files']}))
