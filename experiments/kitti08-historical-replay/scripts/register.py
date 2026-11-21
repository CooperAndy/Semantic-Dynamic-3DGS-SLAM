from common import *
import numpy as np,torch,scipy,subprocess,os,sys,importlib.util,shutil
assert not (R/'registration.json').exists()
logical_before=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
remaining_R=max(0,16*2**30-logical_before);needed=remaining_R+12*2**30+5.15*2**30
assert shutil.disk_usage(R).free>needed
dump(R/'receipts/resource_preflight.json',dict(status='REGISTERED_REMAINING_R_PLUS_ARCHIVE_AND_DISK_FLOOR_COVERED',at=now(),logical_R_before=logical_before,remaining_R_budget=remaining_R,archive_budget=12*2**30,required_free_bytes=needed,actual_free_bytes=shutil.disk_usage(R).free))
assert json.loads((R/'receipts/hook_cpu_contract.json').read_text())['status'].startswith('PASS72')
assert torch.cuda.get_device_capability(0)==(12,0) and (torch.__version__,np.__version__,scipy.__version__)==('2.8.0+cu128','1.26.4','1.17.1')
with (R/'receipts/native_preflight.log').open('x') as f:
 subprocess.run([PY,str(R/'scripts/native_preflight.py')],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),timeout=90,check=True)
sys.path.insert(0,str(R/'source'));spec=importlib.util.spec_from_file_location('adapter_core',R/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
from external_tracking import set_external,camera
m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());raw=np.loadtxt(m['pose_file']).reshape(N,3,4);poses=np.tile(np.eye(4,dtype=np.float32),(N,1,1));poses[:,:3]=raw.astype(np.float32)
params=dict(cam_unnorm_rots=torch.zeros((1,4,N),device='cuda'),cam_trans=torch.zeros((1,3,N),device='cuda'));ws=[]
for i in range(N):set_external(params,torch.linalg.inv(torch.tensor(poses[i],device='cuda')),i,core);ws.append(camera(params,i,core).cpu().numpy())
ws=np.stack(ws);cs=np.linalg.inv(ws);np.savez_compressed(R/'inputs/runtime/orb_adapter_reference.npz',frame_ids=np.arange(N),w2c=ws,c2w=cs)
delta=float(np.abs(cs-poses).max());bound=2e-5+32*np.finfo(np.float32).eps*max(1,float(np.abs(poses).max()));assert delta<=bound
dump(R/'receipts/adapter_bridge.json',dict(status='PASS_NEW_SM120_NATIVE1000_ADAPTER',at=now(),source_max_abs=delta,derived_bound=bound))
native=json.loads((R/'receipts/native_preflight.json').read_text());assert native['sha256']==NATIVE_SHA and sha(R/'source/scripts/splatam_external_tm.py')==CORE_SHA
shutil.copy2(native['rasterizer'],R/'provenance/native_sm120.so')
locks_list=[]
for base in ['source','scripts','inputs','provenance','data_pipeline/pipeline']:
 for p in sorted((R/base).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':locks_list.append(dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
for p in [R/'PROTOCOL.md',R/'plan.json',R/'data_pipeline/input_manifest.json',R/'data_pipeline/input_registration.json',R/'data_pipeline/input_result.json',R/'receipts/input_preparation.json',R/'receipts/hook_cpu_contract.json',R/'receipts/native_preflight.json',R/'receipts/adapter_bridge.json',R/'receipts/resource_preflight.json']:
 locks_list.append(dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
for row in m['rows']:
 for k in ['rgb','right','depth','confidence']:
  assert sha(row[k])==row['hashes'][k];locks_list.append(dict(path=row[k],bytes=Path(row[k]).stat().st_size,sha256=row['hashes'][k]))
locks_list.append(dict(path=native['rasterizer'],bytes=Path(native['rasterizer']).stat().st_size,sha256=native['sha256']))
for path,digest in [('/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth','7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'),('/root/miniconda3/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth','df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0')]:
 assert sha(path)==digest
 locks_list.append(dict(path=path,bytes=Path(path).stat().st_size,sha256=digest))
dump(R/'registration.json',dict(**json.loads((R/'plan.json').read_text()),state='REGISTERED_RECOVERY_FOUR_INHERITED_TWO_FRESH1000_SM120',inherited_variants=VARIANTS[:4],execution_variants=VARIANTS[4:],original_registration_sha256=sha(R/'provenance/original_registration.json'),recovery_manifest_sha256=sha(R/'provenance/recovery_manifest.json'),registered_at=now(),file_locks=locks_list,core_sha256=CORE_SHA,native_kernel_sha256=NATIVE_SHA,native_kernel_path=native['rasterizer'],donor_inputs_sealed_per_pair_before_H3=True))
print('REGISTERED',len(locks_list),flush=True)
