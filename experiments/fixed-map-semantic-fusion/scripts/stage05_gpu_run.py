"""Independent Z condition and native render of DEV stress/V/Z, fixed geometry."""
import json,hashlib,sys,time
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,native_render_variables
from native_semantic_render import semantic_features,render_mass,decode_mass
R=Path(__file__).resolve().parents[1];D=R/'05_条件轻量候选与历史诊断';O=D/'dev_diagnostics_20261007_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert (O/'CPU_result.json').exists() and not (O/'GPU_result.json').exists()
 protocol={'status':'FROZEN_BEFORE_DEPTH_DIAGNOSTIC','DEV_only':True,'z_map':'native alpha normalized camera-z mean in metres; fixed map','z_obs':'SGBM predicted camera z in metres, valid .5..80','common_valid':'original predicted labels/confidence/depth/alpha validity; GT never changes candidate updates','GT_diagnostic_only':'fusion GT depth .5..80, nonmoving, native depth agreement <=.1+.01*zGT; not calibrated noise or guaranteed correct association','depth_bins_m':[[.5,10],[10,30],[30,80.0001]],'Z_trigger':'far and near trusted diagnostic sets each >=1000 pixels; baseline residual>1m rate far exceeds near by >=.10 in either DEV window','Z_abs_m':1.,'Z_relative':.02,'Z_parameters':'single predeclared heuristic, no grid search','Z_combined_with_V':False,'final_test':'not accessed; none newly isolated'}
 dump(O/'depth_protocol.json',protocol);sys.path.insert(0,'/root/autodl-tmp/SplaTAM')
 import torch
 import torch.nn.functional as F
 from utils.recon_helpers import setup_camera
 from utils.slam_helpers import transform_to_frame
 from diff_gaussian_rasterization import GaussianRasterizer as Renderer
 man=json.loads((R/'inputs/vkitti2_gt_dev_v2/manifest.json').read_text());depthrows=[];renderrows=[];locks=[];started=time.time();torch.cuda.reset_peak_memory_stats()
 for seq in man['sequences']:
  name=seq['name'];folder=O/name;old=R/'02_基础因子DEV与Oracle诊断/gt_dev_v2'/f'{name}_full';d=R/'inputs/vkitti2_gt_dev_v2'/name;bp=d/'base_map.npz';bh=sha(bp)
  with np.load(bp) as raw:base={k:raw[k] for k in raw.files}
  ids=base['source_gaussian_index'];n=len(ids);params={k:torch.as_tensor(base[k],device='cuda',dtype=torch.float32) for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']};camera=setup_camera(int(base['org_width']),int(base['org_height']),base['intrinsics'],base['w2c'])
  def forward(t,c):return Renderer(raster_settings=camera)(**native_render_variables(params,t,c,torch.zeros_like,lambda q:F.normalize(q,dim=1),torch.exp,torch.sigmoid))[0]
  def inputs(f):
   paths={'labels':d/'prediction/labels'/f'{f:06d}_semantic.npy','confidence':d/'prediction/confidence'/f'{f:06d}_confidence.npy','depth':Path(seq['records'][f]['files']['depth_sgbm']['path'])}
   arrays={k:np.load(p) for k,p in paths.items()};locks.extend({'path':p.relative_to(R).as_posix(),'sha256':sha(p)} for p in paths.values())
   return {k:torch.as_tensor(v,device='cuda',dtype=torch.int64 if k=='labels' else torch.float32) for k,v in arrays.items()}
  def geom(f,x):
   t=transform_to_frame(params,f,False,False);z=t['means3D'][:,2];m=forward(t,torch.stack([z,torch.ones_like(z),z*z],1));alpha=m[1];md=m[0]/alpha.clamp_min(1e-8);dep=x['depth'];conf=x['confidence'];labels=x['labels'];valid=(labels>=0)&(labels<19)&torch.isfinite(conf)&(conf>=0)&(conf<=1)&torch.isfinite(dep)&(dep>=.5)&(dep<=80)&(alpha>=.05)&torch.isfinite(md)&(md>0)
   return t,md,dep,valid
  counts=np.zeros(3,np.int64);low=np.zeros(3,np.int64);residual=np.zeros(3,np.float64)
  for f in seq['fusion']:
   with torch.no_grad():
    x=inputs(f);t,md,dep,valid=geom(f,x);gdp=Path(seq['records'][f]['files']['gt_depth']['path']);gmp=Path(seq['records'][f]['files']['gt_moving']['path']);locks.extend({'path':p.relative_to(R).as_posix(),'sha256':sha(p)} for p in [gdp,gmp]);gt=torch.as_tensor(np.load(gdp),device='cuda');moving=torch.as_tensor(np.load(gmp),device='cuda',dtype=torch.bool);trusted=valid&torch.isfinite(gt)&(gt>=.5)&(gt<=80)&~moving&((md-gt).abs()<=.1+.01*gt);r=(md-dep).abs()
    for b,(a,z) in enumerate(protocol['depth_bins_m']):mask=trusted&(gt>=a)&(gt<z);counts[b]+=int(mask.sum());low[b]+=int((mask&(r>1)).sum());residual[b]+=float(r[mask].sum(dtype=torch.float64))
  rates=low/np.maximum(counts,1);trigger=bool(counts[0]>=1000 and counts[2]>=1000 and rates[2]-rates[0]>=.10)
  depthrows.append({'sequence':name,'trusted_GT_depth_proxy_counts':counts.tolist(),'baseline_geo_below_half_counts':low.tolist(),'rates':rates.tolist(),'mean_abs_residual_m':(residual/np.maximum(counts,1)).tolist(),'S2_Z_triggered':trigger});dump(O/'depth_trigger_diagnostic.json',depthrows)
  if trigger:
   state=Evidence(ids,19);geo_folder=folder/'Z_geometry_observations';geo_folder.mkdir()
   for f in seq['fusion']:
    with torch.no_grad():x=inputs(f);t,md,dep,valid=geom(f,x);geo=1/(1+((md-dep).abs()/(1+.02*dep))**2);m=torch.stack([valid.float()*geo,valid.float()/(1+(md-dep).abs()**2),valid.float()])
    cols=torch.zeros((n,3),device='cuda',requires_grad=True);grad=torch.autograd.grad(forward(t,cols),cols,grad_outputs=m)[0].detach().cpu().numpy();den=grad[:,2]
    with np.load(old/'observations'/f'{f:06d}_prediction.npz') as c:i=c['gaussian_index'];dist=c['distribution'];sem=c['semantic_reliability'];oldgeo=c['geometry_reliability'];ver=str(c['input_version'])
    assert (den[i]>=.00099).all();assert np.allclose(np.clip(grad[i,1]/den[i],0,1),oldgeo,atol=1e-4,rtol=1e-4);g=np.clip(grad[i,0]/den[i],0,1);np.savez_compressed(geo_folder/f'{f:06d}.npz',gaussian_id=ids[i],gaussian_index=i,geometry_reliability=g,baseline_geo_max_abs_error=np.array(np.max(np.abs(grad[i,1]/den[i]-oldgeo))))
    full=np.zeros((n,19),np.float32);full[i]=dist;w=np.zeros(n,np.float64);w[i]=sem.astype(float)*g.astype(float);source={'dataset':'VKITTI2','sequence':seq['scene']+'/clone','camera':'Camera_0','original_frame':seq['records'][f]['original_frame']};state.update_frame(source,ver+':Z_abs1_rel02_v1',full,w)
    print(json.dumps({'phase':'05_Z_fusion','sequence':name,'frame':f}),flush=True)
   state.save(folder/'states/S2-Z_final.npz')
  # Native evaluation of all perturbed states and separately triggered candidates.
  for sp in sorted((folder/'states').glob('*.npz')):
   group=sp.stem.removesuffix('_final');s=Evidence.load(sp,ids);features=semantic_features(s);out=folder/'renders'/group;assert not out.exists();out.mkdir()
   for f in seq['evaluation']:
    with torch.no_grad():t=transform_to_frame(params,f,False,False);alpha=forward(t,torch.ones_like(params['rgb_colors']))[0].cpu().numpy();dec=decode_mass(render_mass(params,t,features,camera),reference_alpha=alpha)
    np.savez_compressed(out/f'{f:06d}.npz',**{k:v for k,v in dec.items() if k!='class_mass'})
    with np.load(old/'renders/S2'/f'{f:06d}.npz') as b:assert np.array_equal(alpha,b['alpha'])
    renderrows.append({'sequence':name,'group':group,'frame':f,'shared_alpha_identical':True})
   print(json.dumps({'phase':'05_native_render','sequence':name,'group':group}),flush=True)
  for k,t in params.items():assert np.array_equal(t.cpu().numpy(),base[k].astype(np.float32))
  assert sha(bp)==bh
 dump(O/'GPU_input_locks.json',locks);dump(O/'native_render_checks.json',renderrows);dump(O/'GPU_result.json',{'status':'COMPLETE_05_NATIVE_DEV_RENDER','renders':len(renderrows),'GPU':torch.cuda.get_device_name(),'wall_seconds':time.time()-started,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'geometry_RGB_camera_opacity_unchanged':True,'Z_trigger_rows':depthrows,'combined_candidate_run':False,'Test_accessed':False})
if __name__=='__main__':main()
