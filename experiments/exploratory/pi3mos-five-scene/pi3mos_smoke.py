import sys,time,json,torch,numpy as np
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from pi3.models.pi3 import Pi3
n=int(sys.argv[1]); h=int(sys.argv[2]); w=int(sys.argv[3])
root=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted/Scene01/clone/frames/rgb/Camera_0')
imgs=[]
for i in range(n):
 im=Image.open(root/f'rgb_{i:05d}.jpg').convert('RGB').resize((w,h),Image.Resampling.BILINEAR)
 imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255)
t0=time.perf_counter(); m=Pi3().eval(); m.load_state_dict(load_file('../checkpoints/pi3mos.safetensors',device='cpu'),strict=True); m=m.cuda(); x=torch.stack(imgs).unsqueeze(0).cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16): y=m(x)
torch.cuda.synchronize(); p=y['dynamic_masks']; print(json.dumps({'frames':n,'height':h,'width':w,'shape':list(p.shape),'finite':bool(torch.isfinite(p).all()),'min':float(p.min()),'max':float(p.max()),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'forward_s':time.perf_counter()-t,'total_s':time.perf_counter()-t0}),flush=True)
