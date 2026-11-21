import hashlib,json,sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install_gt_guard():
    audit={'blocked_gt_accesses':0,'scope':'Python open hook, runtime view contains no poses'}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=args[0].decode() if isinstance(args[0],bytes) else args[0]
            if 'groundtruth' in name.lower() or 'poses/00.txt' in name.lower() or 'orb_kitti' in name.lower():
                audit['blocked_gt_accesses']+=1;raise PermissionError('Runtime cannot read reference poses')
    sys.addaudithook(guard);return audit
def load_r1_window():
    import cv2,numpy as np,torch
    view=ROOT/'runtime200';protocol=view/'protocol.json';p=json.loads(protocol.read_text())
    for name,digest in p['input_hashes'].items(): assert sha256(view/name)==digest,name
    data=[]
    for row in p['rows']:
        i=row['frame_id'];rgb=cv2.cvtColor(cv2.imread(str(view/row['rgb_path'])),cv2.COLOR_BGR2RGB)
        depth=np.load(view/row['depth_path'],allow_pickle=False)
        assert rgb.shape==(192,640,3) and depth.shape==(192,640) and np.isfinite(depth).all()
        data.append((i,torch.from_numpy(rgb[None]).permute(0,3,1,2),torch.from_numpy(depth),torch.tensor(p['K'],dtype=torch.float32),None,torch.zeros((192,640),dtype=torch.uint8)))
    return data,p['rows'],{'runtime_protocol_sha256':sha256(protocol),'mask_mode':'NO_SEMANTIC_EXCLUSION_DIAGNOSTIC'}
