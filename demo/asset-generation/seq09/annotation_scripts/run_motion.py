"""Causal 8-frame Pi3MOS inference and per-vehicle score aggregation."""
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file
from pi3.models.pi3 import Pi3
from resource_guard import wait_for_headroom
R=Path(__file__).resolve().parent.parent
B=R.parent

def main():
    out=R/'motion';out.mkdir(exist_ok=True)
    model=Pi3().eval();model.load_state_dict(load_file('/root/autodl-tmp/stage2_new_model_pi3mos/checkpoints/pi3mos.safetensors',device='cpu'),strict=True);model.cuda()
    cache={};counts={'static':0,'dynamic':0,'unknown':0};started=time.perf_counter();inference=0
    reused=0
    for t in range(200):
        previous=out/f'{t:06d}.json'
        if previous.exists() and (t<7 or (out/f'{t:06d}.npz').exists()):
            try:
                cached=json.loads(previous.read_text())
                assert cached['frame']==t
                assert cached['window']==(list(range(t-7,t+1)) if t>=7 else [])
                assert all(o['raw_state'] in counts for o in cached['objects'])
                if t>=7:
                    with np.load(out/f'{t:06d}.npz',allow_pickle=False) as c: saved=c['prob']
                    assert saved.shape==(280,910) and np.isfinite(saved).all() and saved.min()>=0 and saved.max()<=1
            except (ValueError,OSError,KeyError,AssertionError):
                for suffix in ('json','npz'):
                    bad=out/f'{t:06d}.{suffix}'
                    if bad.exists():bad.rename(out/f'{t:06d}.{time.time_ns()}.{suffix}.interrupted')
            else:
                for obj in cached['objects']:counts[obj['raw_state']]+=1
                reused+=1
                if t%20==0:print('MOTION_REUSED',t,counts,flush=True)
                continue

        records=[];window=list(range(t-7,t+1)) if t>=7 else []
        meta=json.loads((R/f'instances/{t:06d}.json').read_text());masks=np.load(R/f'instances/{t:06d}_masks.npz')['masks']
        depth=np.load(B/f'raft_left/{t:06d}_depth.npy')
        if t>=7:
            wait_for_headroom('Pi3MOS',t)
            for i in window:
                if i not in cache:
                    im=Image.open(B/f'sequences/09/image_2/{i:06d}.png').convert('RGB');assert im.size==(1226,370)
                    cache[i]=TF.resize(TF.to_tensor(im),[280,910],interpolation=InterpolationMode.BILINEAR,antialias=True)
            cache={i:x for i,x in cache.items() if i>=t-7}
            x=torch.stack([cache[i] for i in window])[None].cuda();torch.cuda.synchronize();tic=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):y=model(x)
            prob=y['dynamic_masks'][0,-1].squeeze().float()
            up=F.interpolate(prob[None,None],size=(370,1226),mode='bilinear',align_corners=False)[0,0]
            assert torch.isfinite(up).all() and up.min()>=0 and up.max()<=1
            probs=up.cpu().numpy();np.savez_compressed(out/f'{t:06d}.npz',prob=prob.cpu().numpy().astype(np.float16));torch.cuda.synchronize();inference+=time.perf_counter()-tic
            del x,y,prob,up
            torch.cuda.empty_cache()
        for obj in meta['instances']:
            if obj['category'].lower() not in ('car','truck','bus','van'):continue
            k=int(obj['mask_index']);mask=masks[k].astype(bool);values=probs[mask] if t>=7 else np.array([])
            score=float(np.median(values)) if len(values) else None
            dv=depth[mask];dv=dv[np.isfinite(dv)&(dv>=.5)&(dv<=40)]
            z=float(np.median(dv)) if len(dv)>=20 else None
            state=('dynamic' if score>=.5 else 'static') if score is not None else 'unknown';counts[state]+=1
            records.append(dict(track_id=int(obj['track_id']),category=obj['category'],bbox=obj['bbox'],mask_index=k,mask_area=int(mask.sum()),score=score,raw_state=state,depth_m=z,valid_depth_pixels=int(len(dv)),range_40m=z is not None,reason='insufficient_real_history' if t<7 else None))
        (out/f'{t:06d}.json').write_text(json.dumps(dict(frame=t,window=window,image_size=[1226,370],objects=records),allow_nan=False))
        if t%10==0 or t==199:print('MOTION',t,counts,'peak_gib',round(torch.cuda.max_memory_allocated()/2**30,3),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(reused_complete_frames=reused,frames=200,inferred_frames=193,unknown_history_frames=list(range(7)),counts=counts,window=8,input_size=[910,280],precision='BF16',threshold=.5,depth='same-map RAFT-Stereo left metric Z, 0.5-40m',scope='prediction distribution, not accuracy; independent offline overlay; no map filtering',elapsed_seconds=time.perf_counter()-started,forward_and_probability_io_seconds=inference,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),indent=2))

if __name__=='__main__':main()
