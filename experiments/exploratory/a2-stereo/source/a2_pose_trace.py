"""Trace original ORB to adapter and emitted trajectory, no map mutation."""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');R=O/'mapper_v1';B=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def compare(a,b):
 assert a.shape==b.shape
 diff=np.abs(a.astype(np.float64)-b.astype(np.float64));ix=np.unravel_index(np.argmax(diff),diff.shape)
 return {'exact':bool(np.array_equal(a,b)),'different_elements':int(np.count_nonzero(a!=b)),'max_abs':float(diff[ix]),'max_index':list(map(int,ix)),'a_at_max':float(a[ix]),'b_at_max':float(b[ix])}
def main():
 start=time.monotonic();torch.set_num_threads(2);out=O/'diagnostic_20261028'/'pose_trace.json';assert not out.exists();out.parent.mkdir(exist_ok=True)
 seal=json.loads((R/'A2_DEV_evidence_seal_20261028.json').read_text());locked={x['path']:x['sha256'] for x in seal['files']}
 for a in ('B0','B1','B2','B3','B4'):
  p=R/'runs'/f'{a}_200'/'params.npz';assert sha(p)==locked[str(p)]
 m=json.loads((B/'inputs/runtime/input_manifest.json').read_text());path=Path(m['pose_file']);assert sha(path)==m['pose_sha256']
 refpath=B/'inputs/runtime/orb_adapter_reference.npz';assert sha(refpath)=='9b1bc315b6b2bd7a957f818a770ef9323046d0e2eaf96eff093ca49516adb46e'
 with np.load(refpath) as z:refw=z['w2c'].copy();refc=z['c2w'].copy()
 with np.load(R/'runs/B0_200/estimated_trajectory.npz') as z:actualw=z['w2c'].copy();actualc=z['c2w'].copy()
 with np.load(R/'runs/B0_200/params.npz') as z:external=z['external_w2c_all_frames'].copy();savedq=z['cam_unnorm_rots'].copy();savedt=z['cam_trans'].copy()
 five={}
 for a in ('B0','B1','B2','B3','B4'):
  with np.load(R/'runs'/f'{a}_200'/'estimated_trajectory.npz') as z:five[a]=compare(z['c2w'],actualc)
 raw=np.tile(np.eye(4),(200,1,1));raw[:,:3]=np.loadtxt(path).reshape(200,3,4)
 sys.path.insert(0,str(R/'source'))
 from utils.slam_helpers import matrix_to_quaternion
 from utils.slam_external import build_rotation
 from external_tracking import set_external,camera
 core=SimpleNamespace(matrix_to_quaternion=matrix_to_quaternion,build_rotation=build_rotation)
 final_params={'cam_unnorm_rots':torch.tensor(savedq,device='cuda'),'cam_trans':torch.tensor(savedt,device='cuda')}
 mapw=np.stack([camera(final_params,i,core).cpu().numpy() for i in range(200)])
 params={'cam_unnorm_rots':torch.nn.Parameter(torch.zeros((1,4,200),device='cuda')),'cam_trans':torch.nn.Parameter(torch.zeros((1,3,200),device='cuda'))}
 oldw=[];recreated=[]
 for i in range(200):
  w=torch.linalg.inv(torch.tensor(raw[i],dtype=torch.float32,device='cuda'));oldw.append(w.detach().cpu().numpy());set_external(params,w,i,core);recreated.append(camera(params,i,core).cpu().numpy())
 oldw=np.stack(oldw);recreated=np.stack(recreated);newc=np.linalg.inv(recreated)
 result={'status':'COMPLETE','numpy_version':np.__version__,'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(),
 'input_sha256':sha(path),'reference_sha256':sha(refpath),'five_arms_vs_B0_c2w':five,
 'steps':{'recreated_GPU_inv_vs_saved_external_w2c':compare(oldw,external),
          'sealed_map_w2c_vs_saved_trajectory_w2c':compare(mapw,actualw),
          'recreated_adapter_w2c_vs_A2_w2c':compare(recreated,actualw),
          'recreated_adapter_w2c_vs_original_reference_w2c':compare(recreated,refw),
          'A2_w2c_vs_original_reference_w2c':compare(actualw,refw),
          'A2_w2c_numpy_inverse_vs_A2_c2w':compare(np.linalg.inv(actualw),actualc),
          'original_reference_w2c_numpy_inverse_vs_reference_c2w':compare(np.linalg.inv(refw),refc),
          'recreated_c2w_vs_A2_c2w':compare(newc,actualc),
          'A2_c2w_vs_reference_c2w':compare(actualc,refc)},
 'limits':'Exact origin may be shared runtime or hardware-dependent arithmetic; this experiment cannot isolate historical generator environment.',
 'elapsed_s':time.monotonic()-start,'quality_adoption':False}
 for a in ('B0','B1','B2','B3','B4'):
  p=R/'runs'/f'{a}_200'/'params.npz';assert sha(p)==locked[str(p)]
 out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
