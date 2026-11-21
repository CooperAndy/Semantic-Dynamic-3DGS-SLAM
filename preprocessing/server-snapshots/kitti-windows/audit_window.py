from common import *
import sys,csv,re,collections,time,numpy as np,shutil
start=time.monotonic();wid=sys.argv[1];assert wid in [w['id'] for w in WINDOWS];W=R/'windows'/wid;reg=json.loads((W/'registration.json').read_text());allreg=json.loads((R/'registration.json').read_text());fr=allreg['parent_G19'];assert sha(F/'bin/causal_stereo')==fr['binary_sha256'] and sha(F/'runtime/ORBvoc.txt')==fr['vocabulary_sha256'];assert sha(W/'settings.yaml')==reg['settings_sha256']
for rel,digest in allreg['source_files'].items():assert sha(R/rel)==digest
for x in fr['libraries']:assert sha(F/x['path'])==x['sha256']
for x in reg['inputs']:assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256']
assert sha(W/'input_view/calib.txt')==reg['calib_sha256'] and sha(W/'input_view/times.txt')==reg['times_sha256'];rows=list(csv.DictReader((W/'output/causal_ledger.csv').open()));assert len(rows)==1000 and [int(x['frame']) for x in rows]==list(range(1000));times=np.loadtxt(W/'input_view/times.txt');raw=np.loadtxt(W/'output/raw_c2w_NOT_FOR_MAPPING.txt');assert raw.shape==(1000,12)
max_inverse=0.;non_ok=[];invalid=[];maps=set();changed=[]
for i,row in enumerate(rows):
 assert int(row['source_frame'])==reg['source_start']+i and int(row['max_input_index'])==i and abs(float(row['timestamp'])-times[i])<1e-12
 tcw=np.array([float(row[f'tcw_{r}{c}']) for r in range(4) for c in range(4)]).reshape(4,4)
 if int(row['tracking_state'])!=2:non_ok.append(i)
 if int(row['valid_pose']):
  assert np.isfinite(tcw).all() and np.array_equal(tcw[3],[0,0,0,1]);expected=np.linalg.inv(tcw);actual=np.eye(4);actual[:3]=raw[i].reshape(3,4);error=float(np.max(np.abs(expected-actual)));assert error<1e-9;max_inverse=max(max_inverse,error);assert int(row['map_id'])>=0;maps.add(int(row['map_id']))
 else:invalid.append(i)
 if int(row['map_changed']):changed.append(i)
trace=W/'output/file_activity.log';opened=set();sealed=-1;emitted=[];calls=0;forbidden=[];image_pattern=re.compile(r'/input_view/image_([23])/(\d{6})\.png');write_pattern=re.compile(r'write\(\d+<[^>]+/causal_ledger\.csv>,\s*"((?:\\.|[^"\\])*)"')
for line in trace.open():
 if '/poses/' in line or 'groundtruth.txt' in line or '/evaluation_only/' in line:forbidden.append(line[:500])
 if 'openat(' in line:
  m=image_pattern.search(line)
  if m:
   camera,i=map(int,m.groups());assert 0<=i<1000 and i<=sealed+1,('Future image',i,sealed);opened.add((camera,i));calls+=1
 m=write_pattern.search(line)
 if m:
  for row in json.loads('"'+m.group(1)+'"').splitlines():
   if row.startswith('frame,'):continue
   i=int(row.split(',',1)[0]);assert i==sealed+1 and (2,i) in opened and (3,i) in opened;sealed=i;emitted.append(i)
assert not forbidden and emitted==list(range(1000)) and len(opened)==2000;ready=not non_ok and not invalid and len(maps)==1
if ready:shutil.copy2(W/'output/raw_c2w_NOT_FOR_MAPPING.txt',W/'output/external_causal_P2_c2w.txt')
final=np.loadtxt(W/'output/offline_final_NOT_FOR_MAPPING.txt');sensitivity=None
if final.shape==(1000,12) and np.isfinite(final).all() and np.isfinite(raw).all():
 x=raw.reshape(1000,3,4)[:,:,3];y=final.reshape(1000,3,4)[:,:,3];u,s,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)));d=np.diag([1.,1.,np.linalg.det(vt.T@u.T)]);rotation=vt.T@d@u.T;t=y.mean(0)-rotation@x.mean(0);errors=np.linalg.norm(x@rotation.T+t-y,axis=1);sensitivity=dict(scope='Two estimates;not GT accuracy;later optimization/gauge effects',SE3_positions_no_scale_RMS_m=float(np.sqrt(np.mean(errors**2))),max_after_SE3_m=float(errors.max()),scale=1.)
dump(W/'audit.json',dict(status='PASS_CAUSAL1000_FRONTEND_INPUT_AND_LEDGER' if ready else 'COMPLETE_1000_FAILURE_LEDGER_PRIMARY_WITHHELD',at=now(),window=wid,frames=1000,states=dict(collections.Counter(x['tracking_state'] for x in rows)),non_OK_frames=non_ok,invalid_pose_frames=invalid,map_ids=sorted(maps),map_changed_frames=changed,max_inverse_error=max_inverse,trace_unique_images=2000,trace_image_open_calls=calls,trace_pose_seals=1000,GT_forbidden_opens=0,trace_sha256=sha(trace),ledger_sha256=sha(W/'output/causal_ledger.csv'),primary_sha256=sha(W/'output/external_causal_P2_c2w.txt') if ready else None,post_shutdown_sensitivity=sensitivity,seconds=time.monotonic()-start,quality_adoption=False));print((W/'audit.json').read_text())
