"""Freeze real base parameters and initialize five empty semantic sidecars."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def array_hash(v):
    h=hashlib.sha256();h.update(str(v.dtype).encode());h.update(json.dumps(list(v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes());return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);source=Path(a.source);target=root/'inputs/base_map/params.npz'
    expected='3244884f5a22ebfc6cf94b4ec0e48ddb4f666f5803938bc5e53cc4c62ae6031b'
    if sha(source)!=expected:raise ValueError('historical base hash mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(source,target)
    if sha(target)!=expected:raise ValueError('base copy hash mismatch')
    with np.load(target,allow_pickle=False) as pz:base={k:pz[k] for k in pz.files}
    n=len(base['means3D']);ids=base['source_gaussian_index'].astype(np.int64)
    fields={k:{'shape':list(v.shape),'dtype':str(v.dtype),'array_sha256':array_hash(v)} for k,v in base.items()}
    manifest={'status':'FROZEN_ENGINEERING_DEV_BASE','source_local':str(source.resolve()),'base_local':str(target.resolve()),
        'base_remote':'/root/autodl-tmp/stage3_chy_20261006/inputs/base_map/params.npz','file_sha256':expected,
        'num_gaussians':n,'fields':fields,'stable_id':'source_gaussian_index; original full-map identity',
        'geometry_exposed_original_frames':list(range(3000,3200)),'rgb_opacity_exposed_original_frames':list(range(3000,3200)),
        'historical_semantic_frames':list(range(3000,3200)),
        'allowed_semantic_fusion_original_frames':[3000+i for i in range(200) if i%10!=9],
        'dev_evaluation_original_frames':[3000+i for i in range(200) if i%10==9],
        'split_status':'EXPOSED_DEV; semantic omission only; geometry and prior development have seen all frames',
        'pose_source':'KITTI GT; does not evaluate tracking','depth_source':'RAFT-Stereo confidence depth used historically',
        'independent_2d_semantic_gt':'NOT_FOUND_IN_AUDITED_KITTI08_INPUTS','prediction_version':'sha256 inventory per label/confidence file',
        'groups':['S0','S1','S-G','S2','S2-H'],'native_rgb_render_invariance':'PENDING_GPU'}
    (root/'inputs/base_map_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    checks=[]
    for group in manifest['groups']:
        # Explicit parameter copies in memory, verified field by field; shared
        # frozen base on disk prevents wasting five identical large files.
        params={k:v.copy() for k,v in base.items()}
        assert all(np.array_equal(params[k],v,equal_nan=True) for k,v in base.items())
        state=Evidence(ids,19);out=root/'01_语义表示_初始化与渲染审计'/'initializations'/group
        out.mkdir(parents=True,exist_ok=True);file=out/'semantic_state.npz'
        if file.exists():
            old=Evidence.load(file,ids)
            if old.evidence.any() or old.support.any() or old.total_weight.any() or old.processed:
                raise FileExistsError('refusing to overwrite accumulated semantic state')
        state.save(file)
        restored=Evidence.load(file,ids)
        assert restored.evidence.sum()==0 and restored.support.sum()==0 and restored.total_weight.sum()==0 and restored.processed=={}
        record={'group':group,'state_sha256':sha(file),'base_sha256':expected,'all_parameter_fields_equal':True,
            'zero_evidence_support_weight':True,'history_conflict':'disabled; no imported history',
            'old_class_labels':'not imported; all unknown','identity_roundtrip':True,'num_gaussians':n,
            'native_rgb_render_check':'PENDING_GPU','parameters_storage':'shared frozen base; separate in-memory copies verified'}
        (out/'initialization_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8');checks.append(record)
    report={'status':'PASS_CPU_INITIALIZATION','groups':checks,'base_file_hash_before_after_equal':sha(target)==expected,
            'gpu_used':False,'native_rgb_render_check':'PENDING_GPU'}
    (root/'01_语义表示_初始化与渲染审计'/'initialization_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'gaussians':n,'groups':len(checks),'status':report['status']}))

if __name__=='__main__':main()
