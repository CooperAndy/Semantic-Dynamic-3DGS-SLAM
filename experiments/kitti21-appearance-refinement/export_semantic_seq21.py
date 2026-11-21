from pathlib import Path
import json,hashlib
import numpy as np
from export_s3_assets_reference import prepare_records,write_ply,CITYSCAPES_COLORS,UNKNOWN_COLOR,EXPECTED_CLASS_NAMES
r=Path('/root/autodl-tmp/seq21_chy_20260926')
param=r/'runs/F1_all200/params.npz'
semantic=r/'semantic_fusion_F1_all200/semantic_gaussian_labels.npz'
out=r/'viewer_assets/F1/KITTI21_F1_Semantic.ply'
with np.load(param,allow_pickle=False) as f: p={k:f[k] for k in ('means3D','rgb_colors','log_scales','unnorm_rotations','logit_opacities')}
with np.load(semantic,allow_pickle=False) as f: labels=f['label']; indices=f['source_gaussian_index']; names=tuple(map(str,f['class_names'].tolist()))
count=len(p['means3D'])
assert names==EXPECTED_CLASS_NAMES and len(labels)==count and np.array_equal(indices,np.arange(count))
assert labels.min()>=-1 and labels.max()<len(CITYSCAPES_COLORS)
colors=np.tile(UNKNOWN_COLOR,(count,1));known=labels>=0;colors[known]=CITYSCAPES_COLORS[labels[known]]
records=prepare_records(p['means3D'],colors,p['log_scales'],p['unnorm_rotations'],p['logit_opacities'])
out.parent.mkdir(parents=True,exist_ok=True);write_ply(out,records)
sha=lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
manifest={'source_params_sha256':sha(param),'source_semantic_sha256':sha(semantic),'semantic_ply_sha256':sha(out),'gaussian_count':count,'assigned_count':int(known.sum()),'pose_source':'ORB-SLAM3 estimated'}
(out.parent/'KITTI21_F1_semantic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
