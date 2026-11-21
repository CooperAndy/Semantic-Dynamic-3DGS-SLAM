"""GT-free runtime view: the accepted 200-frame order, RGB-D and masks only."""
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tum_window import ROOT,sha256

VIEW=ROOT/'02_data_masks/R1_runtime200'
DATA=ROOT/'02_data_masks/TUM/rgbd_dataset_freiburg3_walking_xyz'
def install_gt_guard():
    audit={'guard':'Python audit open hook; code and input view audited separately','blocked_gt_accesses':0}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=args[0].decode() if isinstance(args[0],bytes) else args[0]
            if Path(path).name.lower()=='groundtruth.txt':
                audit['blocked_gt_accesses']+=1
                raise PermissionError('R1 runtime may not open groundtruth.txt')
    sys.addaudithook(guard)
    return audit

def load_r1_window():
    index=VIEW/'frames.csv'
    with index.open() as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['frame_id','rgb_timestamp','rgb_path','depth_path']
        rows=list(reader)
    assert len(rows)==200
    assert not (VIEW/'groundtruth.txt').exists()
    manifest=VIEW/'seg_mask/manifest.json'
    meta=json.loads(manifest.read_text())
    assert meta['model_revision']=='4a5bac8e64f82681a12db2e151a4c2f4ce6092b2'
    assert meta['model_weights_sha256']=='c0b2fe11dfecee6f2f1f315f466946e96f4e94813f3f6d660ff3747b83c28cc9'
    assert meta['person_label']==12
    records={x['file']:x for x in meta['masks']}
    data=[]
    for i,row in enumerate(rows):
        assert int(row['frame_id'])==i
        rgb_path=VIEW/row['rgb_path'];depth_path=VIEW/row['depth_path']
        assert rgb_path.resolve().is_relative_to((DATA/'rgb').resolve())
        assert depth_path.resolve().is_relative_to((DATA/'depth').resolve())
        mask_path=VIEW/'seg_mask'/Path(row['rgb_path']).name
        assert sha256(rgb_path)==records[mask_path.name]['rgb_sha256']
        assert sha256(mask_path)==records[mask_path.name]['sha256']
        rgb=cv2.imread(str(rgb_path));depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
        rgb=cv2.cvtColor(cv2.resize(rgb,(512,384)),cv2.COLOR_BGR2RGB)
        image=torch.from_numpy(rgb[None]).permute(0,3,1,2)
        depth=torch.from_numpy(cv2.resize(depth,(512,384)).astype(np.float32)/5000.)
        labels=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        assert labels.shape==(480,640) and labels.dtype==np.uint8 and labels.max()<150
        mask=cv2.dilate((labels==12).astype(np.uint8),np.ones((8,8),np.uint8),iterations=1)
        mask=torch.from_numpy(cv2.resize(mask,(512,384)).astype(np.uint8))
        intrinsics=.8*torch.tensor([535.4,539.2,320.1,247.6])
        # Filler requires a six-field tuple but ignores this None slot.
        data.append((i,image,depth,intrinsics,None,mask))
    return data,rows,{'runtime_index_sha256':sha256(index),'mask_manifest_sha256':sha256(manifest)}
