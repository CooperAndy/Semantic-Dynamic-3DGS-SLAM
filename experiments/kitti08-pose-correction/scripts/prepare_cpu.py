from common import *
import ast,copy,shutil,subprocess,time,numpy as np
start=time.monotonic();assert not (R/'source').exists() and not (R/'plan.json').exists()
ca=json.loads((C/'receipts/audit.json').read_text());fa=json.loads((F/'receipts/audit.json').read_text());assert ca['status']=='PASS_1000_PAST_ONLY_CONFIDENCE_INPUT_REPLAY' and fa['status']=='PASS_CAUSAL1000_INPUT_POSES_AND_FAILURE_LEDGER';assert ca['GT_forbidden_opens']==fa['GT_forbidden_opens']==0
assert sha(C/'input_manifest.json')==ca['input_manifest_sha256'];manifest=json.loads((C/'input_manifest.json').read_text());assert manifest['frame_count']==1000 and manifest['pose_source']=='ORB_estimate' and manifest['GT_runtime'] is False;assert sha(manifest['pose_file'])==fa['primary_sha256']==manifest['pose_sha256']
for name in ['source','native']:shutil.copytree(D/name,R/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['receipts','provenance','runs']:(R/name).mkdir()
shutil.copy2(D/'registration.json',R/'provenance/D_registration.json');shutil.copy2(C/'registration.json',R/'provenance/G19C_registration.json');shutil.copy2(C/'receipts/audit.json',R/'provenance/G19C_audit.json');shutil.copy2(F/'receipts/audit.json',R/'provenance/G19_audit.json');shutil.copy2(C/'calib.txt',R/'provenance/calib.txt')
assert sha(R/'source/scripts/splatam_external_tm.py')=='6d65efd83b61c414c82231e67928c0c8f4dd564c9e7cb56440241b8343f544b7';assert sha(R/'native/diff_gaussian_rasterization/_C.cpython-312-x86_64-linux-gnu.so')=='b6d51ef9c8e68cc12675ffb429ad40a036fb203cb9104cb18ce839904b56e40b'
fixed=['backend.py','factor_hooks.py','feature_pose.py','pose_chain.py','visibility.py']
for name in fixed:assert sha(R/'scripts'/name)==sha(D/'scripts'/name),name
assert sha(R/'scripts/stable_birth.py')==sha(R.parent/'stage3_g18e_20261029_p47149/stable_birth.py')
runtime=R/'inputs/runtime';runtime.mkdir(parents=True);shutil.copy2(manifest['pose_file'],runtime/'external_causal_P2_c2w.txt');manifest['pose_file']=str(runtime/'external_causal_P2_c2w.txt');dump(runtime/'input_manifest.json',manifest);locks=[]
for row in manifest['rows']:
 for key in ['rgb','depth','confidence']:
  p=Path(row[key]);assert sha(p)==row['hashes'][key];locks.append(dict(path=str(p),bytes=p.stat().st_size,sha256=row['hashes'][key]))
 dep=np.load(row['depth']);conf=np.load(row['confidence']);assert dep.dtype==conf.dtype==np.float32 and dep.shape==conf.shape==(manifest['H'],manifest['W']) and np.isfinite(dep).all() and np.isfinite(conf).all();assert ((dep==0)|((dep>=.5)&(dep<=40))).all() and ((dep==0)|(conf>=.8)).all()
 assert time.monotonic()-start<1200
variants=json.loads((D/'plan.json').read_text())['variants'];hold=list(range(19,1000,20));yaml=R/'source/configs/data/g20_manifest.yaml';K=manifest['K'];yaml.write_text(f"dataset_name: kitti\ninput_manifest: {runtime/'input_manifest.json'}\ncamera_params:\n  image_height: {manifest['H']}\n  image_width: {manifest['W']}\n  fx: {K[0][0]}\n  fy: {K[1][1]}\n  cx: {K[0][2]}\n  cy: {K[1][2]}\n  png_depth_scale: 1.0\n  crop_edge: 0\n")
for v in variants:
 c=ast.literal_eval(ast.parse((D/'source/configs/kitti'/f'{v}.py').read_text()).body[0].value);assert c['seed']==0 and c['data']['num_frames']==1000 and c['eval_only_frames']==hold
 c.update(workdir=str(R/'runs'),task_id='T3-G20',scope='Full1000 causal-input stable-birth native tracking/history DEV;not formal adoption');c['data']['gradslam_data_cfg']=str(yaml);(R/'source/configs/kitti'/f'{v}.py').write_text('config = '+repr(c)+'\n')
dump(R/'plan.json',dict(frames=1000,variants=variants,holdout=hold,seed=0,donor='B00_ORB_S0',mapping_iters=60,source_input_locks=locks,budget=dict(per_run_seconds=9000,total_seconds=57000,points=6000000,free_disk_bytes=6*2**30,new_bytes=12*2**30,combined_new_bytes=16*2**30,start_min_free_bytes=23*2**30),causal_input_manifest_sha256=ca['input_manifest_sha256'],stable_birth_sha256=sha(R/'scripts/stable_birth.py'),quality_adoption=False))
with (E/'cpu_contract.log').open('x') as log:subprocess.run([PY,str(R/'scripts/cpu_contract.py')],env=dict(env(),CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
dump(R/'receipts/preparation_cpu.json',dict(status='PASS_3000_CAUSAL_INPUT_LOCKS_AND_1024_PROPAGATION_CONTRACTS_PENDING_NATIVE_GPU',at=now(),source_inputs=len(locks),fixed_method_hashes={n:sha(R/'scripts'/n) for n in fixed},stable_birth_sha256=sha(R/'scripts/stable_birth.py'),seconds=time.monotonic()-start,GT_reads=0,GPU_calls=0,quality_adoption=False));print((R/'receipts/preparation_cpu.json').read_text())
