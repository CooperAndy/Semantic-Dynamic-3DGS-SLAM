from pathlib import Path
import numpy as np
D=Path(__file__).parent;I=D.parent/'trace_contract'
for i in range(3):
 for side in ['left','right']:(I/side/f'{i:06d}.png').read_bytes()
 np.save(D/f'{i:06d}_depth.npy',np.arange(128*1024,dtype=np.float32).reshape(128,1024),allow_pickle=False)
