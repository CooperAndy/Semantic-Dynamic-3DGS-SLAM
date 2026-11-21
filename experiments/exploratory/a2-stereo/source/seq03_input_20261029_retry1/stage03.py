from pathlib import Path
import json,hashlib,os,sys,shutil,importlib.util
import numpy as np
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');OLD=O/'recheck08_v1';I=O/'seq03_input_20261029_retry1';R=O/'seq03_recheck_v1';D=O/'reference_seq03_000000_000199_lidar'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rep(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
assert not R.exists() and shutil.disk_usage(O).free>8*2**30
inputm=json.loads((I/'input_manifest.json').read_text());audit=json.loads((I/'independent_input_audit.json').read_text());assert audit['status']=='PASS_INDEPENDENT_INPUT_AUDIT' and audit['manifest_sha256']==sha(I/'input_manifest.json')
cal=I/'input_view/calib.txt';assert sha(cal)=='d76b5668c898c4c6eba9b1024eeba43694cf9026362a2d85b893e3039f8934d0'
vals={k:np.array([float(x) for x in v.split()]) for line in cal.read_text().splitlines() if ':' in line for k,v in [line.split(':',1)]}
P2=vals['P2'].reshape(3,4);P3=vals['P3'].reshape(3,4);c2=-np.linalg.solve(P2[:,:3],P2[:,3]);c3=-np.linalg.solve(P3[:,:3],P3[:,3]);fb=float(P2[0,0]*abs(c3[0]-c2[0]));assert np.allclose(P2[:,:3],inputm['K']) and .1<fb/P2[0,0]<1
lid=json.loads((D/'manifest.json').read_text());assert [r['source_frame'] for r in lid['frames']]==list(range(19,200,20))
for r in lid['frames']:
 assert '/sequences/03/velodyne/' in r['member'] and sha(D/r['filename'])==r['sha256']
R.mkdir();(R/'runs').mkdir();(R/'source').symlink_to(OLD/'source',target_is_directory=True)
inputm['mapping_holdout_frames']=list(range(19,200,20));(R/'input_manifest.json').write_text(json.dumps(inputm,indent=2))
pose=np.loadtxt(inputm['pose_file']).reshape(200,3,4);T=np.tile(np.eye(4),(200,1,1));T[:,:3,:]=pose
np.savez_compressed(R/'orb_adapter_reference.npz',frame_ids=np.arange(200),c2w=T,w2c=np.linalg.inv(T))
priorreg=json.loads((OLD/'run_registration.json').read_text())
for x in priorreg['file_locks']:
 assert sha(x['path'])==x['sha256'],x['path']
scriptnames=['recheck08_worker.py','recheck08_audit.py','recheck08_score.py','recheck08_export.py','recheck08_run25.py','recheck08_run200.py']
for seed in (0,42,2026):
 S=R/f'seed_{seed}';S.mkdir();(S/'runs').mkdir();(S/'source').symlink_to(OLD/'source',target_is_directory=True)
 yaml='camera_params:\n  crop_edge: 0\n  cx: '+str(P2[0,2])+'\n  cy: '+str(P2[1,2])+'\n  fx: '+str(P2[0,0])+'\n  fy: '+str(P2[1,1])+'\n  image_height: 375\n  image_width: 1242\n  png_depth_scale: 1.0\ndataset_name: kitti\ninput_manifest: '+str(R/'input_manifest.json')+'\n';(S/'data.yaml').write_text(yaml)
 for name in scriptnames:
  s=(OLD/name).read_text()
  s=s.replace("O/'recheck08_v1'",f"Path('{S}')").replace("Path('/root/autodl-tmp/A2_stereo_robust_geometry/recheck08_v1')",f"Path('{S}')")
  s=s.replace("BASE/'inputs/runtime/input_manifest.json'",f"Path('{R}/input_manifest.json')").replace("B/'inputs/runtime/input_manifest.json'",f"Path('{R}/input_manifest.json')")
  if name=='recheck08_worker.py':
   s=rep(s,"Path('/root/autodl-tmp/stage3_g05B_20261025/input_view/calib.txt')",f"Path('{cal}')")
   s=rep(s,'4abedb688d60249c36353b9893570f993ebfdc5ef7938f52f3a7b9e1edbfffb1',sha(cal))
   s=rep(s,'378.6112506585076',repr(fb))
   s=rep(s,"assert cfg['data']['sequence']=='08' and manifest['H']==370 and manifest['W']==1226",f"cfg['seed']={seed};cfg['mapping']['sampling_seed']={seed};cfg['mapping']['residual_fb']=fb;cfg['data'].update(sequence='03',desired_image_height=375,desired_image_width=1242,gradslam_data_cfg='{S}/data.yaml');assert manifest['H']==375 and manifest['W']==1242")
   s=rep(s,'core.seed_everything(0)',f'core.seed_everything({seed})')
   s=rep(s,'cap=3000,seed=0,log_path',f'cap=3000,seed={seed},log_path')
   s=rep(s,"receipt={'status':'PASS_ENGINEERING','arm':ARM",f"receipt={{'seed':{seed},'status':'PASS_ENGINEERING','arm':ARM")
  if name=='recheck08_audit.py':
   s=rep(s,"B/'inputs/runtime/orb_adapter_reference.npz'",f"Path('{R}/orb_adapter_reference.npz')")
  s=s.replace('exposed KITTI08 B200 DEV','KITTI03 candidate test; mapper-only holdout')
  compile(s,str(S/name),'exec');(S/name).write_text(s)
 locks=[]
 for x in priorreg['file_locks']:
  if '/source/' in x['path']:locks.append(x)
 for name in scriptnames+['data.yaml']:
  p=S/name;locks.append({'path':str(p),'sha256':sha(p)})
 for p in [R/'input_manifest.json',cal,R/'orb_adapter_reference.npz']:
  locks.append({'path':str(p),'sha256':sha(p)})
 reg={'state':'APPROVED_FOR_EXECUTION_AFTER_AUDIT','source':'KITTI03 0-199 candidate test; scope audit limited','seed':seed,'arms':['C0','C2'],'C2_gamma':0.7633236491505211,'C2_fb':fb,'file_locks':locks,'quality_adoption':False,'execution':'not started'}
 (S/'run_registration.json').write_text(json.dumps(reg,indent=2))
summary={'status':'STAGED_NO_MAPPING_RUN','sequence':'03','frames':200,'seeds':[0,42,2026],'fb':fb,'T':0.8,'input_manifest_sha256':sha(R/'input_manifest.json'),'pose_sha256':sha(inputm['pose_file']),'old_registration_sha256':sha(OLD/'run_registration.json'),'quality_adoption':False}
(R/'staging_receipt.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
