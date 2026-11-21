import json,sys,time,glob,torch,numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file
from pi3.models.pi3 import Pi3

S=('Scene01','Scene02','Scene06','Scene18','Scene20')
R=Path('/root/autodl-tmp/stage2_new_model_pi3mos');V=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted');v=Path(glob.glob('/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_*/vehicle_only_v2')[0]);O=R/'run_all_clone_20261105';O.mkdir(exist_ok=True)
P={s:v/'g2_online_dev_full_v2/prediction_root'/s/'clone/instances' for s in S[:-1]};P['Scene20']=v/'scene20_formal_recovery_one_shot_20260929_v3/test_output_scene20_vehicle_20260929_recovery_03/prediction_root/Scene20/clone/instances'
conf={'scenes':list(S),'variant':'clone','mode':'causal current and seven historical frames','warmup_frames':7,'pair_label':'t to t+1; final frame excluded','resolution':[280,910],'threshold_vehicle':0.5,'aggregation':'median dynamic probability over predicted instance mask','depth_filter_inference':False,'scoring_ranges':['40m GT any valid pixel','all labelled objects'],'weight_sha256':'1a536b1bb6e9c6c16a77b7ed544649351d387e281a16f3d946e0786cdcf23ded','pretrained_scene_split':'unknown','pi3_code_commit':'5e709d27324a935e364e01efc56f3128ac00bf24','mask_sources':{s:str(P[s]) for s in S},'warning':'Scene20 user-authorized exposed evaluation; not formal M3 reacceptance. Previous M3 remains unchanged.'}
f=O/'config.json'
if f.exists():
 if json.loads(f.read_text())!=conf:raise RuntimeError('Existing config differs; stop')
else:f.write_text(json.dumps(conf,indent=2))
m=Pi3().eval();m.load_state_dict(load_file(str(R/'checkpoints/pi3mos.safetensors'),device='cpu'),strict=True);m=m.cuda();torch.cuda.reset_peak_memory_stats();res=[]
for sc in S:
 imgs=sorted((V/sc/'clone/frames/rgb/Camera_0').glob('rgb_*.jpg'));n=len(imgs);out=O/sc;out.mkdir(exist_ok=True);cache={}
 for t in range(7,n-1):
  dest=out/f'{t:06d}.npz'
  if dest.exists():continue
  for k in range(t-7,t+1):
   if k not in cache:
    im=Image.open(imgs[k]).convert('RGB');cache[k]=TF.resize(TF.to_tensor(im),[280,910],interpolation=InterpolationMode.BILINEAR,antialias=True)
  cache={k:a for k,a in cache.items() if k>=t-7};x=torch.stack([cache[k] for k in range(t-7,t+1)]).unsqueeze(0).cuda();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):y=m(x)
  p=y['dynamic_masks'][0,-1].squeeze().float();p=torch.nn.functional.interpolate(p[None,None],size=(375,1242),mode='bilinear',align_corners=False)[0,0];torch.cuda.synchronize();seconds=time.perf_counter()-start
  if not torch.isfinite(p).all() or p.min()<0 or p.max()>1:raise RuntimeError('invalid output')
  a=p.cpu().numpy();meta=json.loads((P[sc]/f'{t:06d}.json').read_text())['instances'];masks=np.load(P[sc]/f'{t:06d}_masks.npz')['masks'];ids=[];scores=[]
  for item in meta:
   if item.get('category','').lower() not in {'car','truck','bus','van'}:continue
   mask=masks[int(item['mask_index'])].astype(bool);vals=a[mask];ids.append(int(item['track_id']));scores.append(float(np.median(vals)) if len(vals) else float('nan'))
  if len(ids)!=len(set(ids)):raise ValueError('duplicate track')
  native=y['dynamic_masks'][0,-1].squeeze().float().cpu().numpy().astype(np.float16)
  tmp=out/f'{t:06d}.tmp.npz';np.savez_compressed(tmp,prob=native,track_ids=np.array(ids,dtype=np.int64),scores=np.array(scores,dtype=np.float32),seconds=np.array(seconds));tmp.replace(dest)
  del x,y,p
  if t%25==0:print(sc,t,'/',n-2,'sec',round(seconds,3),'peakGiB',round(torch.cuda.max_memory_allocated()/2**30,2),flush=True)
 print('DONE',sc,'frames',n,flush=True)
(O/'inference_complete.json').write_text(json.dumps({'completed':True,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30},indent=2))
