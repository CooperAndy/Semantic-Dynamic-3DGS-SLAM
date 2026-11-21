from common import *
import csv,re,shutil,collections,time,numpy as np
start=time.monotonic();reg=json.loads((R/'registration.json').read_text());assert sha(R/'bin/causal_stereo')==reg['binary_sha256'] and sha(R/'inputs/settings.yaml')==reg['settings_sha256']
for x in reg['libraries']:assert sha(R/x['path'])==x['sha256']
for x in reg['inputs']:assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256']
rows=list(csv.DictReader((R/'output/causal_ledger.csv').open()));assert len(rows)==1000 and [int(x['frame']) for x in rows]==list(range(1000));times=np.loadtxt(R/'inputs/times.txt');raw=np.loadtxt(R/'output/raw_c2w_NOT_FOR_MAPPING.txt');assert raw.shape==(1000,12)
max_inverse=0.;non_ok=[];invalid=[];maps=set();changed=[]
for i,row in enumerate(rows):
 assert int(row['source_frame'])==i+3000 and int(row['max_input_index'])==i and abs(float(row['timestamp'])-times[i])<1e-12
 tcw=np.array([float(row[f'tcw_{r}{c}']) for r in range(4) for c in range(4)]).reshape(4,4)
 if int(row['tracking_state'])!=2:non_ok.append(i)
 if int(row['valid_pose']):
  assert np.isfinite(tcw).all() and np.allclose(tcw[3],[0,0,0,1],atol=0,rtol=0)
  expected=np.linalg.inv(tcw);actual=np.eye(4);actual[:3]=raw[i].reshape(3,4);err=float(np.max(np.abs(actual-expected)));assert err<1e-9;max_inverse=max(max_inverse,err);maps.add(int(row['map_id']));assert int(row['map_id'])>=0
 else:invalid.append(i)
 if int(row['map_changed']):changed.append(i)
trace=R/'output/file_activity.log';opened=set();sealed=-1;emitted=[];image_calls=0;forbidden=[]
image_pattern=re.compile(r'/input_view/image_([23])/(\d{6})\.png')
write_pattern=re.compile(r'write\(\d+<[^>]+/causal_ledger\.csv>,\s*"((?:\\.|[^"\\])*)"')
for line in trace.open(errors='strict'):
 if '/poses/' in line or 'groundtruth.txt' in line:forbidden.append(line[:500])
 if 'openat(' in line:
  match=image_pattern.search(line)
  if match:
   camera,i=int(match.group(1)),int(match.group(2));assert 0<=i<1000 and i<=sealed+1,('Future image before current pose sealed',i,sealed);opened.add((camera,i));image_calls+=1
 match=write_pattern.search(line)
 if match:
  text=json.loads('"'+match.group(1)+'"')
  for record in text.splitlines():
   if record.startswith('frame,'):continue
   i=int(record.split(',',1)[0]);assert i==sealed+1 and (2,i) in opened and (3,i) in opened;sealed=i;emitted.append(i)
assert not forbidden and emitted==list(range(1000)) and len(opened)==2000
ready=not non_ok and not invalid and len(maps)==1
if ready:shutil.copy2(R/'output/raw_c2w_NOT_FOR_MAPPING.txt',R/'output/external_causal_P2_c2w.txt')
final=np.loadtxt(R/'output/offline_final_NOT_FOR_MAPPING.txt');sensitivity=None
if final.shape==(1000,12) and np.isfinite(final).all() and np.isfinite(raw).all():
 x=raw.reshape(1000,3,4)[:,:,3];y=final.reshape(1000,3,4)[:,:,3];u,sv,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)));d=np.diag([1.,1.,np.linalg.det(vt.T@u.T)]);rotation=vt.T@d@u.T;translation=y.mean(0)-rotation@x.mean(0);errors=np.linalg.norm(x@rotation.T+translation-y,axis=1);sensitivity=dict(scope='Two estimates,not GT accuracy;includes later optimization and gauge effects',raw_position_RMS_m=float(np.sqrt(np.mean(np.sum((x-y)**2,axis=1)))),SE3_positions_no_scale_RMS_m=float(np.sqrt(np.mean(errors**2))),max_after_SE3_m=float(errors.max()),scale=1.)
dump(R/'receipts/audit.json',dict(status='PASS_CAUSAL1000_INPUT_POSES_AND_FAILURE_LEDGER' if ready else 'COMPLETE_LEDGER_PRIMARY_WITHHELD_FAILURES',at=now(),frames=1000,states=dict(collections.Counter(x['tracking_state'] for x in rows)),non_OK_frames=non_ok,invalid_pose_frames=invalid,map_ids=sorted(maps),map_changed_frames=changed,max_inverse_error=max_inverse,trace_unique_image_files=len(opened),trace_image_open_calls=image_calls,trace_pose_seals=len(emitted),GT_forbidden_opens=len(forbidden),source_registration_sha256=sha(R/'registration.json'),trace_sha256=sha(trace),ledger_sha256=sha(R/'output/causal_ledger.csv'),primary_sha256=sha(R/'output/external_causal_P2_c2w.txt') if ready else None,post_shutdown_sensitivity=sensitivity,seconds=time.monotonic()-start,quality_adoption=False))
print((R/'receipts/audit.json').read_text())
