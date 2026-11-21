from common import *
import numpy as np,re,time
start=time.monotonic();reg=json.loads((R/'registration.json').read_text());assert json.loads((R/'receipts/contract.json').read_text())['status'].startswith('PASS_');assert json.loads((R/'producer_receipt.json').read_text())['frames']==1000
for name,digest in reg['source_files'].items():assert sha(R/name)==digest,name
assert sha(P/'pipeline/confidence.py')==sha(R/'provenance/legacy_confidence.py');assert sha(R/'calib.txt')==reg['calib_sha256'];assert sha(F/'output/external_causal_P2_c2w.txt')==reg['frontend_primary_sha256']
rows=[json.loads(line) for line in (R/'per_frame.jsonl').open()];assert len(rows)==1000;outrows=[];mins=[];source_paths={};pose_rows=[]
for i,(source,row) in enumerate(zip(reg['rows'],rows)):
 assert row['index']==row['max_input_index']==i and row['previous_index']==(i-1 if i else None)
 for name,x in source['inputs'].items():
  p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];source_paths[str(p)]=(name,i)
 d=np.load(source['inputs']['depth_left']['path'],allow_pickle=False);arrays={}
 for name,x in row['outputs'].items():
  p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];arrays[name]=np.load(p,allow_pickle=False);assert arrays[name].shape==tuple(reg['shape'])
 c,s,l,h=[arrays[x] for x in ['confidence','seed_mask','loss_mask','depth_confhard']];assert c.dtype==h.dtype==np.float32 and s.dtype==l.dtype==np.bool_ and np.isfinite(c).all() and np.isfinite(h).all();assert np.min(c)>=0 and np.max(c)<=1;assert np.array_equal(s,c>=.8) and np.array_equal(l,c>=.2);assert np.all(c[(d<.5)|(d>40)]==0);assert np.array_equal(h,np.where(s,d,0).astype(np.float32));assert np.all((h==0)|((h>=.5)&(h<=40)));assert abs(float(c.mean())-row['mean_confidence'])<1e-12 and abs(float(s.mean())-row['seed_ratio'])<1e-12
 T=np.load(source['inputs']['pose']['path']);pose_rows.append(T[:3].reshape(12));mins.append(row['seed_ratio']);outrows.append(dict(frame_id=i,source_frame_id=i+reg['source_start'],rgb=source['inputs']['left']['path'],right=source['inputs']['right']['path'],depth=row['outputs']['depth_confhard']['path'],confidence=row['outputs']['confidence']['path'],hashes=dict(rgb=source['inputs']['left']['sha256'],right=source['inputs']['right']['sha256'],depth=row['outputs']['depth_confhard']['sha256'],confidence=row['outputs']['confidence']['sha256'])))
 assert time.monotonic()-start<600
opened=set();sealed=-1;seals=[];calls=0;forbidden=[]
path_pattern=re.compile(r'openat\([^,]*, "([^"]+)"');seal_pattern=re.compile(r'write\(\d+<[^>]+/seals\.log>, "sealed (\d+)\\n"')
for line in (R/'file_activity.log').open():
 if '/poses/' in line or 'groundtruth.txt' in line or '/evaluation_only/' in line:forbidden.append(line[:500])
 m=path_pattern.search(line)
 if m and m.group(1) in source_paths:
  name,i=source_paths[m.group(1)];assert i<=sealed+1,('Future input',name,i,sealed);opened.add((name,i));calls+=1
 m=seal_pattern.search(line)
 if m:
  i=int(m.group(1));assert i==sealed+1 and all((name,i) in opened for name in ['left','right','depth_left','depth_right','pose']);sealed=i;seals.append(i)
assert not forbidden and seals==list(range(1000)) and len(opened)==5000
assert np.array_equal(np.array(pose_rows),np.loadtxt(F/'output/external_causal_P2_c2w.txt'))
manifest=dict(task_id='T3-G21D-PAST-INPUT',pose_source='ORB_estimate',pose_file=str(F/'output/external_causal_P2_c2w.txt'),pose_sha256=reg['frontend_primary_sha256'],frame_convention='P2 c2w, immediate returned pose in window map0, metres',frame_count=1000,K=reg['K'],H=reg['shape'][0],W=reg['shape'][1],depth_units='metres',valid_depth_range_m=[.5,80],confidence_threshold=.8,GT_runtime=False,temporal_scope='Cached replay with causally generated poses and previous-only confidence;not live end-to-end/FPS',rows=outrows)
dump(R/'input_manifest.json',manifest);dump(R/'receipts/audit.json',dict(status='PASS_1000_PAST_ONLY_CONFIDENCE_INPUT_REPLAY',at=now(),frames=1000,unique_source_files=len(opened),source_open_calls=calls,sealed_outputs=len(seals),GT_forbidden_opens=0,mean_seed_ratio=float(np.mean(mins)),min_seed_ratio=float(np.min(mins)),first_seed_ratio=mins[0],confidence_depth_dtype='float32',mask_dtype='bool',source_confidence_byte_unchanged=True,input_manifest_sha256=sha(R/'input_manifest.json'),trace_sha256=sha(R/'file_activity.log'),seconds=time.monotonic()-start,quality_adoption=False));print((R/'receipts/audit.json').read_text())
