"""RAFT-Large forward flow for the existing FlowWarpTracker input contract."""
from pathlib import Path
import json,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large
from resource_guard import wait_for_headroom
R=Path(__file__).resolve().parent.parent
B=R.parent

def image(i):
    x=torch.from_numpy(np.asarray(Image.open(B/f'sequences/09/image_2/{i:06d}.png').convert('RGB')).copy()).permute(2,0,1).float()[None].cuda()/127.5-1
    return F.pad(x,(0,6,0,6),mode='replicate')

def main():
    out=R/'flow';out.mkdir(exist_ok=False)
    model=raft_large(weights=None,progress=False)
    state=torch.load('/root/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model.eval().cuda()
    previous=image(0);t=time.perf_counter()
    for i in range(1,200):
        wait_for_headroom('flow',i)
        current=image(i)
        with torch.inference_mode():flow=model(previous,current,num_flow_updates=20)[-1][0,:,:370,:1226].permute(1,2,0).float().cpu().numpy()
        assert flow.shape==(370,1226,2) and np.isfinite(flow).all()
        np.savez_compressed(out/f'{i-1:06d}_{i:06d}.npz',observed_flow=flow.astype(np.float16))
        previous=current
        if i==1 or i%20==0:print('FLOW',i,'/199',round(time.perf_counter()-t,2),flush=True)
    (out/'summary.json').write_text(json.dumps({'pairs':199,'model':'torchvision RAFT Large C_T_SKHT_V2','updates':20,'input_normalization':'RGB / 127.5 - 1','padding':'replicate right/bottom 6 pixels to 1232x376','output':'float16 H,W,2 forward flow, restored original 1226x370','elapsed_seconds':time.perf_counter()-t},indent=2))

if __name__=='__main__':main()
