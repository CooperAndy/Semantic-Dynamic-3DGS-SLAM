from pathlib import Path
import json, hashlib, numpy as np
R=Path('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_input_20261029_retry1')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
result=json.loads((R/'result.json').read_text());m=json.loads((R/'input_manifest.json').read_text())
assert result['status']=='PASS_INPUT_PREPARATION' and result['manifest_sha256']==sha(R/'input_manifest.json') and len(m['rows'])==200
assert m['confidence_threshold']==0.8 and m['pose_sha256']==sha(m['pose_file'])
summary=[]
for row in m['rows']:
 i=row['frame_id'];assert i==row['source_frame_id']
 for k in ('rgb','right','depth','confidence'):assert sha(row[k])==row['hashes'][k]
 raw=np.load(R/f'raw_left/{i:06d}_depth.npy');d=np.load(row['depth']);c=np.load(row['confidence'])
 assert raw.shape==d.shape==c.shape==(375,1242) and np.isfinite(raw).all() and np.isfinite(d).all() and np.isfinite(c).all()
 expected=(raw>=.5)&(raw<=40)&(c>=.8)
 assert np.array_equal(d>0,expected),i
 assert np.array_equal(d[expected],raw[expected]),i
 summary.append({'frame':i,'raw_valid_fraction':float(((raw>=.5)&(raw<=40)).mean()),'confidence_valid_fraction':float(expected.mean())})
T=np.loadtxt(m['pose_file']).reshape(200,3,4);assert np.isfinite(T).all()
steps=np.linalg.norm(np.diff(T[:,:,3],axis=0),axis=1)
orb=(R/'logs/ORB.log').read_text(errors='replace')
obj={'status':'PASS_INDEPENDENT_INPUT_AUDIT','frames':200,'checked_sha_files':800,'threshold':.8,'depth_mask_exact_match':True,'retained_fraction_mean':float(np.mean([x['confidence_valid_fraction'] for x in summary])),'retained_fraction_min':float(min(x['confidence_valid_fraction'] for x in summary)),'retained_fraction_max':float(max(x['confidence_valid_fraction'] for x in summary)),'translation_step_min_m':float(steps.min()),'translation_step_max_m':float(steps.max()),'consecutive_identical_pose_count':int(np.sum(np.max(np.abs(np.diff(T,axis=0)),axis=(1,2))<1e-9)),'orb_lost_mentions':sum('lost' in x.lower() for x in orb.splitlines()),'manifest_sha256':sha(R/'input_manifest.json'),'pose_sha256':sha(m['pose_file']),'frames_detail':summary}
(R/'independent_input_audit.json').write_text(json.dumps(obj,indent=2))
print(json.dumps({k:v for k,v in obj.items() if k!='frames_detail'},indent=2))
