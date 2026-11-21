import json,time,hashlib,torch,numpy as np,sys
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file
from pi3.models.pi3 import Pi3
scene=sys.argv[1];start=int(sys.argv[2]);end=int(sys.argv[3]); assert (scene,start,end) in [('Scene01',0,49),('Scene06',200,249)]
root=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted')/scene/'clone/frames/rgb/Camera_0'; out=Path('/root/autodl-tmp/stage2_new_model_pi3mos/run_20261105')/scene;out.mkdir(parents=True,exist_ok=False)
conf={'scene':scene,'start':start,'end':end,'window':8,'stride':1,'target': 'last_frame_t','prediction_target':'gt frame_t to frame_t+1; model uses only frames <= frame_t','height':280,'width':910,'resize':'torchvision bilinear antialias true; back bilinear align_corners false','precision':'bfloat16 autocast','vehicle_aggregation':'median of finite dynamic_masks probability in predicted mask','vehicle_threshold':0.5,'pixel_threshold_visualization':0.25,'weight_sha256':'1a536b1bb6e9c6c16a77b7ed544649351d387e281a16f3d946e0786cdcf23ded','code_commit':'5e709d27324a935e364e01efc56f3128ac00bf24','range':'GT postscore only','ground_truth_input':False}
(out/'config.json').write_text(json.dumps(conf,indent=2))
m=Pi3().eval();m.load_state_dict(load_file('../checkpoints/pi3mos.safetensors',device='cpu'),strict=True);m=m.cuda(); imgs=[]
for i in range(start,end+1):
 image=Image.open(root/f'rgb_{i:05d}.jpg').convert('RGB'); t=TF.to_tensor(image);t=TF.resize(t,[280,910],interpolation=InterpolationMode.BILINEAR,antialias=True);imgs.append(t)
results=[];torch.cuda.reset_peak_memory_stats()
for t in range(start+7,end+1):
 x=torch.stack(imgs[t-start-7:t-start+1]).unsqueeze(0).cuda(); torch.cuda.synchronize();ts=time.perf_counter()
 with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16): y=m(x)
 p=y['dynamic_masks'][0,-1].squeeze().float(); p=torch.nn.functional.interpolate(p[None,None],size=(375,1242),mode='bilinear',align_corners=False)[0,0];torch.cuda.synchronize(); elapsed=time.perf_counter()-ts
 if not torch.isfinite(p).all() or p.min()<0 or p.max()>1: raise RuntimeError('invalid probability')
 a=p.cpu().numpy();np.savez_compressed(out/f'{t:06d}_prob.npz',prob=a)
 results.append({'frame_t':t,'frames':list(range(t-7,t+1)),'seconds':elapsed,'min':float(a.min()),'max':float(a.max())})
 if (t-start)%10==0: print(scene,t,round(elapsed,2),flush=True)
 del x,y,p
(out/'run.json').write_text(json.dumps({'frames':results,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30},indent=2));print('DONE',scene,len(results),flush=True)
