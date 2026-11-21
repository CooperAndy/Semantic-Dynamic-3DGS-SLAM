"""C01: explicit 200-frame P2 ORB replay; no GT fallback or padded trajectories."""
from pathlib import Path
import json,hashlib,numpy as np,torch
from PIL import Image

class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self,config_dict,device='cuda:0',desired_height=376,desired_width=1241,start=0,end=-1,stride=1,**kwargs):
        path=Path(config_dict['input_manifest']);self.manifest=json.loads(path.read_text())
        m=self.manifest
        assert m['pose_source']=='ORB_estimate' and m['GT_runtime'] is False
        assert start==0 and stride==1 and end in [-1,m['frame_count']]
        assert (desired_height,desired_width)==(m['H'],m['W'])
        self.rows=m['rows'];assert [r['frame_id'] for r in self.rows]==list(range(m['frame_count']))
        raw=np.loadtxt(m['pose_file'],dtype=np.float64);assert raw.shape==(len(self.rows),12)
        assert hashlib.sha256(Path(m['pose_file']).read_bytes()).hexdigest()==m['pose_sha256']
        self.poses=np.tile(np.eye(4),(len(raw),1,1));self.poses[:,:3]=raw.reshape(-1,3,4)
        assert np.max(np.abs(self.poses[0]-np.eye(4)))<1e-6
        self.device=device;self.K=np.eye(4,dtype=np.float32);self.K[:3,:3]=m['K']
        self.name='kitti_manifest_P2';self.retained_inds=torch.arange(len(self.rows))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        if index<0 or index>=len(self):raise IndexError(index)
        row=self.rows[index]
        for key in ['rgb','depth']:
            assert hashlib.sha256(Path(row[key]).read_bytes()).hexdigest()==row['hashes'][key],row[key]
        with Image.open(row['rgb']) as im:rgb=np.array(im.convert('RGB'),copy=True)
        depth=np.load(row['depth'],allow_pickle=False)
        assert depth.dtype==np.float32 and depth.shape==rgb.shape[:2]
        assert np.isfinite(depth).all() and ((depth==0)|((depth>=.5)&(depth<=80))).all()
        return (torch.tensor(rgb,dtype=torch.float32,device=self.device),
                torch.tensor(depth[...,None],device=self.device),
                torch.tensor(self.K,device=self.device),
                torch.tensor(self.poses[index],dtype=torch.float32,device=self.device))
