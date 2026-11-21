"""Original final states, all200 native renders; keep only sparseGT samples."""
import json,sys,hashlib,time
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,native_render_variables
from native_semantic_render import render_mass,semantic_features,decode_mass
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/semantickitti08_v1';B=R/'02_基础因子DEV与Oracle诊断/semantickitti08_gt_v1';OLD=R/'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1';GROUPS=['S0','S1','S-G','S2','S2-H']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 man=json.loads((D/'evaluation_input_manifest.json').read_text());proto=R/'protocol/SemanticKITTI08稀疏GT评价预声明_20261007_v1.json';assert sha(proto)==man['protocol_sha256'];assert sha(R/'inputs/base_map/params.npz')==man['base_sha256'];B.mkdir(parents=True,exist_ok=True);out=B/'renders';assert not out.exists();out.mkdir();sys.path.insert(0,'/root/autodl-tmp/SplaTAM')
 import torch
 import torch.nn.functional as F
 from diff_gaussian_rasterization import GaussianRasterizer as Renderer
 from utils.recon_helpers import setup_camera
 from utils.slam_helpers import transform_to_frame
 dependencies=json.loads((R/'01_语义表示_初始化与渲染审计/native_audit_20261007_v1/source_manifest.json').read_text())[:4];assert all(sha(v['source'])==v['sha256'] for v in dependencies);dump(B/'native_dependencies.json',dependencies)
 with np.load(R/'inputs/base_map/params.npz') as b:base={k:b[k] for k in b.files}
 ids=base['source_gaussian_index'];params={k:torch.as_tensor(base[k],device='cuda',dtype=torch.float32) for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']};camera=setup_camera(int(base['org_width']),int(base['org_height']),base['intrinsics'],base['w2c']);start=time.time();torch.cuda.reset_peak_memory_stats();shared=[];oldchecks=[]
 def forward(t,c):return Renderer(raster_settings=camera)(**native_render_variables(params,t,c,torch.zeros_like,lambda q:F.normalize(q,dim=1),torch.exp,torch.sigmoid))[0]
 for g in GROUPS:
  sp=OLD/'states'/f'{g}_final.npz';assert sha(sp)==man['state_sha256'][g];state=Evidence.load(sp,ids);features=semantic_features(state);folder=out/g;folder.mkdir()
  for rec in man['records']:
   f=rec['local_frame'];assert sha(rec['GT_file']['path'])==rec['GT_file']['sha256']
   with np.load(rec['GT_file']['path']) as gt:pixel=np.flatnonzero(gt['point_index'].ravel()>=0)
   with torch.no_grad():
    t=transform_to_frame(params,f,False,False);z=t['means3D'][:,2];mom=forward(t,torch.stack([z,torch.ones_like(z),z*z],1)).cpu().numpy();alpha=mom[1];mapdepth=np.divide(mom[0],alpha,out=np.zeros_like(alpha),where=alpha>0);decoded=decode_mass(render_mass(params,t,features,camera),reference_alpha=alpha)
   if f%10==9:
    with np.load(OLD/'renders'/g/f'{f:06d}.npz') as prev:
     assert np.array_equal(prev['label'],decoded['label']) and np.array_equal(prev['alpha'],decoded['alpha']) and np.array_equal(prev['score'],decoded['score'])
    oldchecks.append({'group':g,'local_frame':f,'original20_render_bitwise_identical':True})
   np.savez_compressed(folder/f'{rec["original_frame"]:06d}.npz',pixel_index=pixel,label=decoded['label'].ravel()[pixel],score=decoded['score'].ravel()[pixel],alpha=alpha.ravel()[pixel],map_depth=mapdepth.ravel()[pixel])
   shared.append({'group':g,'frame':f,'alpha_sha256':hashlib.sha256(alpha.ravel()[pixel].tobytes()).hexdigest()});print(json.dumps({'phase':'native_render','group':g,'frame':rec['original_frame']}),flush=True)
  assert sha(sp)==man['state_sha256'][g]
 for k,t in params.items():assert np.array_equal(t.cpu().numpy(),base[k].astype(np.float32))
 assert sha(R/'inputs/base_map/params.npz')==man['base_sha256'];dump(B/'original20_render_reproduction.json',oldchecks);dump(B/'shared_alpha_identity.json',shared);dump(B/'render_result.json',{'status':'COMPLETE_NATIVE_RENDER_ALL200_FIVE_FIXED_STATES','views':200,'renders':1000,'old100_renders_bitwise_identical':True,'base_states_and_geometry_unchanged':True,'wall_seconds':time.time()-start,'peak_torch_GPU_bytes':torch.cuda.max_memory_allocated(),'GT_not_fused':True,'GT_only_selects_saved_sampling_locations':True,'evaluation':'sparse projectedGT samples, exposedDEV','full_image_GT_claim':False})
if __name__=='__main__':main()
