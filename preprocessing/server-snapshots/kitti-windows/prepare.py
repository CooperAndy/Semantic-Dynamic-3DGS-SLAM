from common import *
import numpy as np,shutil,struct,time
start=time.monotonic();assert not (R/'registration.json').exists();assert json.loads((F/'receipts/audit.json').read_text())['status']=='PASS_CAUSAL1000_INPUT_POSES_AND_FAILURE_LEDGER';fr=json.loads((F/'registration.json').read_text())
for rel,digest in [('bin/causal_stereo',fr['binary_sha256']),('runtime/ORBvoc.txt',fr['vocabulary_sha256']),('inputs/settings.yaml',fr['settings_sha256'])]:assert sha(F/rel)==digest
for x in fr['libraries']:assert sha(F/x['path'])==x['sha256']
settings=(F/'inputs/settings.yaml').read_text();kv={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in settings.splitlines() if ':' in line};calib_sources=[];records=[]
dump(R/'window_selection.json',dict(at=now(),status='WINDOWS_FIXED_BEFORE_FRONTEND_OR_GT_SCORING',windows=WINDOWS,selection='First1000 of existing05/10;no outcome based replacement',GT_reads=0,quality_adoption=False))
for win in WINDOWS:
 W=R/'windows'/win['id'];view=W/'input_view';(view/'image_2').mkdir(parents=True);(view/'image_3').mkdir();(W/'output').mkdir();raw=DATA/win['sequence'];cal={}
 for line in (raw/'calib.txt').read_text().splitlines():
  k,*v=line.split()
  if k in ['P2:','P3:']:cal[k[:-1]]=np.array(v,dtype=np.float64).reshape(3,4)
 p2,p3=cal['P2'],cal['P3'];assert np.array_equal(p2[:,:3],p3[:,:3]);assert np.allclose([p2[0,0],p2[1,1],p2[0,2],p2[1,2]],[float(kv[k]) for k in ['Camera1.fx','Camera1.fy','Camera1.cx','Camera1.cy']],rtol=0,atol=1e-10);beta=p2[0,3]-p3[0,3];assert abs(beta/p2[0,0]-float(kv['Stereo.b']))<1e-12
 times=np.loadtxt(raw/'times.txt',dtype=np.float64)[:1000];assert times.shape==(1000,) and np.isfinite(times).all() and (np.diff(times)>0).all();times=times-times[0];np.savetxt(view/'times.txt',times,fmt='%.17g');shutil.copy2(raw/'calib.txt',view/'calib.txt');shutil.copy2(F/'inputs/settings.yaml',W/'settings.yaml');inputs=[]
 for i in range(1000):
  for cam in ['image_2','image_3']:
   src=raw/cam/f'{i:06d}.png';p=view/cam/src.name
   with src.open('rb') as handle:header=handle.read(24)
   assert header[:8]==bytes([137,80,78,71,13,10,26,10]);width,height=struct.unpack('>II',header[16:24]);assert (height,width)==(int(kv['Camera.height']),int(kv['Camera.width']));p.symlink_to(src);inputs.append(dict(path=str(p),source=str(src),bytes=src.stat().st_size,sha256=sha(src)))
  assert time.monotonic()-start<600
 record=dict(**win,shape=[height,width],K=p2[:,:3].tolist(),projection_beta=float(beta),settings_sha256=sha(W/'settings.yaml'),calib_sha256=sha(view/'calib.txt'),times_sha256=sha(view/'times.txt'),source_calib_sha256=sha(raw/'calib.txt'),source_times_sha256=sha(raw/'times.txt'),inputs=inputs);dump(W/'registration.json',record);records.append(record);print('REGISTERED',win['id'],len(inputs),flush=True)
affinity=sorted(os.sched_getaffinity(0))[:4];assert affinity;dump(R/'registration.json',dict(at=now(),status='REGISTERED_TWO_CAUSAL_FRONTEND_WINDOWS_NOT_EXECUTED',windows=records,source_files={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.md']},parent_G19=dict(root=str(F),binary_sha256=fr['binary_sha256'],vocabulary_sha256=fr['vocabulary_sha256'],libraries=fr['libraries']),affinity=affinity,GT_allowed=False,quality_adoption=False))
