"""Read-only bounded input audit. No torch/CUDA imports. Writes only --out."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def file_info(path):
    p=Path(path)
    return {'path':str(p),'exists':p.is_file(),**({'bytes':p.stat().st_size,'sha256':sha(p)} if p.is_file() else {})}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    root=Path('/root/autodl-tmp'); s=root/'s3_seq08_work'
    candidates=[s/'deployment_20260925_200f_v1/runs/P1_all200_recovery800_resume_v1/params.npz',s/'subset/subset_manifest.json',s/'subset/semantic/id2label.json',s/'subset/semantic/palette.json',s/'deployment_20260925_200f_v1/code/fuse_semantic_gaussians_kitti08.py']
    result={'cpu_only':True,'files':[file_info(p) for p in candidates],'semantic_arrays':[],'dataset_roots':[]}
    sem=s/'subset/semantic'
    for kind,pattern in [('labels','*_semantic.npy'),('confidence','*_confidence.npy')]:
        files=sorted((sem/kind).glob(pattern)); entries=[]
        for p in files:
            v=np.load(p,allow_pickle=False)
            entries.append({'file':str(p),'sha256':sha(p),'shape':list(v.shape),'dtype':str(v.dtype),'min':float(v.min()),'max':float(v.max()),'finite':bool(np.isfinite(v).all())})
        result['semantic_arrays'].append({'kind':kind,'count':len(files),'entries':entries})
    for p in candidates:
        if p.suffix=='.json' and p.exists(): result.setdefault('metadata',{})[str(p)]=json.loads(p.read_text())
    rgb=s/'subset/sequences/08/image_2'; files=sorted(rgb.glob('*.png'))
    result['rgb']={'path':str(rgb),'count':len(files),'first_last':[file_info(p) for p in ([files[0],files[-1]] if files else [])]}
    for d in [root/'datasets',root/'SLAM',s/'subset',root/'datasets/vkitti2_2.0.3/extracted']:
        result['dataset_roots'].append({'path':str(d),'children':[p.name for p in d.iterdir()] if d.is_dir() else []})
    # Bounded path-name audit, not reading GT pixels or consumed Test predictions.
    for d in [root/'datasets/vkitti2_2.0.3/extracted',root/'SLAM']:
        found=[]
        if d.exists():
            import os
            for base,dirs,names in os.walk(d):
                rel=Path(base).relative_to(d)
                if len(rel.parts)>=4: dirs[:]=[]
                for n in dirs+names:
                    if any(t in n.lower() for t in ['classseg','semantic','kitti','label','color','intrinsic','extrinsic']): found.append(str(Path(base)/n))
                if len(found)>250: break
        result.setdefault('gt_path_candidates',[]).append({'root':str(d),'paths':found[:250]})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'map_exists':result['files'][0]['exists'],'label_count':result['semantic_arrays'][0]['count'],'confidence_count':result['semantic_arrays'][1]['count'],'rgb_count':len(files)}))

if __name__=='__main__':main()
