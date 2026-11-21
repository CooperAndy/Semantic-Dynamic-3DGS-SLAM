"""Full frozen-map native semantic GPU viewer. No mapping, training or GT tuning."""
import argparse,io,json,time,sys,hashlib,threading,platform
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'04_系统整合与最终归档/live_gpu_20261007_v1'
sys.path.insert(0,'/root/autodl-tmp/SplaTAM')
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizer
from utils.recon_helpers import setup_camera
from utils.slam_helpers import transform_to_frame
from semantic_cpu import Evidence
from native_semantic_render import semantic_features,render_mass,decode_mass
LOCK=threading.Lock()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Viewer:
 def __init__(self):
  self.bp=R/'inputs/base_map/params.npz';self.sp=R/'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1/states/S0_final.npz'
  self.hashes={'base':sha(self.bp),'state':sha(self.sp)}
  with np.load(self.bp) as b:self.base={k:b[k] for k in b.files}
  b=self.base;self.n=len(b['means3D']);assert self.n==907488 and b['log_scales'].shape[1]==1
  self.p={k:torch.as_tensor(b[k],device='cuda',dtype=torch.float32) for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']}
  self.state=Evidence.load(self.sp,b['source_gaussian_index']);self.features=semantic_features(self.state)
  self.c=[torch.as_tensor(np.pad(self.features,((0,0),(0,1)))[:,i:i+3].copy(),device='cuda') for i in range(0,21,3)]
  self.common={'means2D':torch.zeros_like(self.p['means3D']),'rotations':F.normalize(self.p['unnorm_rotations'],dim=1),'scales':torch.exp(self.p['log_scales']).repeat(1,3),'opacities':torch.sigmoid(self.p['logit_opacities'])}
  self.palette=torch.tensor(json.loads((R/'04_系统整合与最终归档/final_delivery_20261007_v1/viewer_data.json').read_text())['palette'],device='cuda',dtype=torch.uint8)
  self.cameras={}
 def camera(self,scale):
  if scale not in self.cameras:
   w=max(1,round(int(self.base['org_width'])*scale));h=max(1,round(int(self.base['org_height'])*scale));k=self.base['intrinsics'].copy();k[:2]*=scale
   self.cameras[scale]=setup_camera(w,h,k,self.base['w2c'])
  return self.cameras[scale]
 @torch.inference_mode()
 def forward(self,frame=99,yaw=0.,mode='semantic',scale=1.):
  t=transform_to_frame(self.p,frame,False,False);pts=t['means3D']
  if yaw:
   angle=np.deg2rad(yaw);rot=torch.tensor([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],device='cuda',dtype=torch.float32);pts=pts@rot.T
  camera=self.camera(scale);renderer=GaussianRasterizer(raster_settings=camera)
  var=dict(self.common,means3D=pts)
  if mode=='rgb':return renderer(**dict(var,colors_precomp=self.p['rgb_colors']))[0]
  mass=torch.cat([renderer(**dict(var,colors_precomp=c))[0] for c in self.c],0)[:20]
  return mass
 @staticmethod
 def decode(mass):
  alpha=mass.sum(0);classes=mass[:19];unknown=mass[19];mx,label=classes.max(0);supported=alpha>=.05
  labelled=supported&(classes.sum(0)>0)&(unknown/alpha.clamp_min(1e-20)<.5)
  pred=torch.full_like(label,-2);pred[supported]=-1;pred[labelled]=label[labelled]
  return pred,mx/alpha.clamp_min(1e-20),alpha
 def image(self,frame,yaw,mode,scale):
  start=time.perf_counter();torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
  x=self.forward(frame,yaw,mode,scale)
  if mode=='rgb':im=(x.clamp(0,1)*255).to(torch.uint8).permute(1,2,0)
  else:
   label,score,_=self.decode(x);im=self.palette[label.clamp(0,18)].clone();im[label==-1]=90;im[label==-2]=0
   if mode=='score':v=(score.clamp(0,1)*255).to(torch.uint8);im=torch.stack([v,v,v],-1);im[label==-2]=0
  z.record();torch.cuda.synchronize();gpu=a.elapsed_time(z);arr=im.cpu().numpy();bio=io.BytesIO();Image.fromarray(arr).save(bio,format='JPEG',quality=88)
  return bio.getvalue(),{'gpu_ms':gpu,'server_ms':(time.perf_counter()-start)*1000,'gaussians':self.n,'shape':list(arr.shape)}
 def audit(self):
  O.mkdir(parents=True,exist_ok=False);results=[]
  torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
   for f in [9,99,199]:
    mass=self.forward(f);ref=render_mass(self.p,transform_to_frame(self.p,f,False,False),self.features,self.camera(1.));assert np.allclose(mass.cpu().numpy(),ref,atol=5e-5,rtol=1e-4)
    pred,score,alpha=self.decode(mass);dec=decode_mass(ref);assert np.array_equal(pred.cpu().numpy(),dec['label'])
    old=R/f'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1/renders/S0/{f:06d}.npz'
    with np.load(old) as v:assert np.array_equal(pred.cpu().numpy(),v['label']);assert np.allclose(score.cpu().numpy(),v['score'],atol=5e-5,rtol=1e-4)
    ones=GaussianRasterizer(raster_settings=self.camera(1.))(**dict(self.common,means3D=transform_to_frame(self.p,f,False,False)['means3D'],colors_precomp=torch.ones_like(self.p['rgb_colors'])))[0][0]
    error=float((alpha-ones).abs().max());assert torch.allclose(alpha,ones,atol=5e-5,rtol=1e-4)
    for mode in ['rgb','semantic']:
     for _ in range(3):self.image(f,0,mode,1.)
     samples=[];server=[]
     for _ in range(12):data,stats=self.image(f,0,mode,1.);samples.append(stats['gpu_ms']);server.append(stats['server_ms'])
     (O/f'{3000+f}_{mode}.jpg').write_bytes(data)
     results.append({'frame':3000+f,'mode':mode,'resolution':[int(self.base['org_width']),int(self.base['org_height'])],'passes':7 if mode=='semantic' else 1,'GPU_median_ms':float(np.median(samples)),'GPU_p95_ms':float(np.percentile(samples,95)),'server_median_ms':float(np.median(server)),'server_p95_ms':float(np.percentile(server,95)),'GPU_median_fps':float(1000/np.median(samples)),'full_frame_server_median_fps':float(1000/np.median(server)),'shared_alpha_max_abs_error':error})
   for yaw in [-5,5]:
    data,stats=self.image(99,yaw,'semantic',1.);(O/f'3099_yaw{yaw:+}_semantic.jpg').write_bytes(data)
   # Controlled unknown foreground using the same cached native path and GPU decoder.
   cam=setup_camera(129,129,np.array([[100,0,64],[0,100,64],[0,0,1]],float),np.eye(4));means=torch.tensor([[0.,0.,2.],[0.,0.,4.]],device='cuda');feat=torch.zeros((2,21),device='cuda');feat[0,19]=1;feat[1,0]=1
   var={'means3D':means,'means2D':torch.zeros_like(means),'rotations':torch.tensor([[1.,0,0,0],[1.,0,0,0]],device='cuda'),'scales':torch.tensor([[.12]*3,[.24]*3],device='cuda'),'opacities':torch.full((2,1),.99,device='cuda')}
   m=torch.cat([GaussianRasterizer(raster_settings=cam)(**dict(var,colors_precomp=feat[:,i:i+3].contiguous()))[0] for i in range(0,21,3)])[:20];assert int(self.decode(m)[0][64,64])==-1
   var['opacities'][0]=0;leak=torch.cat([GaussianRasterizer(raster_settings=cam)(**dict(var,colors_precomp=feat[:,i:i+3].contiguous()))[0] for i in range(0,21,3)])[:20];assert int(self.decode(leak)[0][64,64])==0
  assert sha(self.bp)==self.hashes['base'] and sha(self.sp)==self.hashes['state']
  result={'status':'PASS_FULL_NATIVE_GPU_RENDERING','gaussians':self.n,'all_Gaussians_active_no_sampling':True,'unknown_foreground_test_and_negative_control':True,'fixed_original_labels_match':True,'base_state_hashes_before_after':self.hashes,'no_GT_or_tuning':True,'performance':results,'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_reserved_bytes':torch.cuda.memory_reserved(),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'30fps_semantic_server_pass':all(r['server_p95_ms']<=1000/30 for r in results if r['mode']=='semantic'),'timing_scope':'warm GPU rendering/decoding and server CPU JPEG; excludes network/browser; sequential requests; no backward'}
  (O/'gpu_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result),flush=True)
V=None
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  try:
   q=urlparse(self.path)
   if q.path=='/':data=(Path(__file__).with_name('live_semantic_gpu.html')).read_bytes();kind='text/html; charset=utf-8';stats={}
   elif q.path=='/status':data=json.dumps(json.loads((O/'gpu_audit.json').read_text())).encode();kind='application/json';stats={}
   elif q.path=='/render':
    p=parse_qs(q.query);frame=int(p.get('frame',['99'])[0]);yaw=float(p.get('yaw',['0'])[0]);mode=p.get('mode',['semantic'])[0];scale=float(p.get('scale',['1'])[0]);assert 0<=frame<200 and -15<=yaw<=15 and mode in ['rgb','semantic','score'] and scale in [1.,.5]
    with LOCK:data,stats=V.image(frame,yaw,mode,scale)
    kind='image/jpeg'
   else:self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Render-Stats',json.dumps(stats));self.end_headers();self.wfile.write(data)
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as exc:self.send_error(400,str(exc))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');p.add_argument('--serve-existing-audit',action='store_true');a=p.parse_args();V=Viewer()
 if not a.serve_existing_audit:V.audit()
 if not a.audit_only:print('READY 127.0.0.1:18766',flush=True);ThreadingHTTPServer(('127.0.0.1',18766),Handler).serve_forever()
