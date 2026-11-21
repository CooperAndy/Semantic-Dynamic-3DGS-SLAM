"""Full Step00 CPU audit and source-lock recovery; no inference or GT scoring."""
import argparse,csv,hashlib,json,os,shutil,platform
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/stage3_chy_20261006')
SOURCE=Path('/root/autodl-tmp/slam_project/data/kitti/dataset/sequences/08')
EXPECTED='3244884f5a22ebfc6cf94b4ec0e48ddb4f666f5803938bc5e53cc4c62ae6031b'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def ah(v):
    h=hashlib.sha256();h.update(str(v.dtype).encode());h.update(json.dumps(list(v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes());return h.hexdigest()

def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    out=ROOT/'00_输入与基图_GT曝光审计/reaudit_20261007_v1'
    out.mkdir(exist_ok=True)
    recovery=ROOT/'inputs/kitti08_source_locked_20261007_v1'
    if recovery.exists():raise FileExistsError('refusing existing recovered input version')
    recovery.mkdir()
    rows=[];issues=[]
    old=json.loads((ROOT/'00_输入与基图_GT曝光审计/remote_inputs.json').read_text())
    old_hashes={}
    for group in old['semantic_arrays']:
        for e in group['entries']:
            local_frame=int(Path(e['file']).name.split('_')[0]);old_hashes[(group['kind'],local_frame)]=e['sha256']
    sem=SOURCE/'semantic_mask2former_3000_3599'
    for kind in ['image_2','image_3','labels','confidence','depth_sgbm_candidate']:(recovery/kind).mkdir()
    for local in range(200):
        original=local+3000
        for kind,path,filename in [
            ('image_2',SOURCE/'image_2'/f'{original:06d}.png',f'{local:06d}.png'),
            ('image_3',SOURCE/'image_3'/f'{original:06d}.png',f'{local:06d}.png'),
            ('labels',sem/'labels'/f'{original:06d}_semantic.npy',f'{local:06d}_semantic.npy'),
            ('confidence',sem/'confidence'/f'{original:06d}_confidence.npy',f'{local:06d}_confidence.npy'),
            ('depth_sgbm_candidate',SOURCE/'depth_sgbm_3000_3599'/f'{original:06d}_depth.npy',f'{local:06d}_depth.npy')]:
            row={'kind':kind,'local_frame':local,'original_frame':original,'source':str(path),'exists':path.is_file()}
            if not path.is_file():issues.append('missing '+str(path));rows.append(row);continue
            row.update(sha256=sha(path),bytes=path.stat().st_size)
            if path.suffix=='.png':
                with Image.open(path) as im:row.update(shape=[im.height,im.width],mode=im.mode)
                row['shape_valid']=row['shape']==[370,1226]
            else:
                v=np.load(path,allow_pickle=False);row.update(shape=list(v.shape),dtype=str(v.dtype),finite=bool(np.isfinite(v).all()),min=float(v.min()),max=float(v.max()))
                row['shape_valid']=v.shape==(370,1226)
                if kind=='labels':row['range_valid']=bool(np.issubdtype(v.dtype,np.integer) and v.min()>=0 and v.max()<=18)
                if kind=='confidence':row['range_valid']=bool(v.min()>=0 and v.max()<=1)
                if kind=='depth_sgbm_candidate':
                    positive=np.isfinite(v)&(v>0);row['positive_fraction']=float(positive.mean())
                    row['positive_p50']=float(np.median(v[positive])) if positive.any() else None
                if kind in ['labels','confidence']:row['matches_20261006_cache']=row['sha256']==old_hashes.get((kind,local))
            if not row['shape_valid'] or row.get('finite') is False or row.get('range_valid') is False:issues.append('invalid '+str(path))
            dest=recovery/kind/filename;dest.symlink_to(path.resolve());row['stage3_reference']=str(dest);rows.append(row)
    metadata=[]
    for path in [SOURCE/'calib.txt',SOURCE/'times.txt',SOURCE.parents[1]/'poses/08.txt',
       sem/'generation_summary.json',sem/'verification_report.json',sem/'input_rgb_sha256.json',sem/'id2label.json',sem/'palette.json',sem/'model_config.json',
       Path('/root/autodl-tmp/slam_project/stage2_semantic/scripts/generate_mask2former_semantics_kitti.py')]:
        if path.is_file():
            target=out/path.name;shutil.copy2(path,target);metadata.append({'source':str(path),'snapshot':str(target),'sha256':sha(path)})
        else:issues.append('missing metadata '+str(path))
    generation=json.loads((sem/'generation_summary.json').read_text())
    checkpoint=Path(generation['checkpoint']['path'])
    model={'path':str(checkpoint),'exists':checkpoint.is_file(),'record':generation['checkpoint']}
    if checkpoint.is_file():model['sha256']=sha(checkpoint)
    dump(out/'model_provenance.json',{'generation':generation,'checkpoint_live':model,'producer':metadata[-1]})
    calib={}
    for line in (SOURCE/'calib.txt').read_text().splitlines():
        if ':' in line:
            name,data=line.split(':',1);calib[name]=np.fromstring(data,sep=' ').tolist()
    p2=np.array(calib['P2']).reshape(3,4);p3=np.array(calib['P3']).reshape(3,4)
    poses_path=SOURCE.parents[1]/'poses/08.txt';poses=np.loadtxt(poses_path);times=np.loadtxt(SOURCE/'times.txt')
    c2w=np.stack([np.vstack([v.reshape(3,4),[0,0,0,1]]) for v in poses[3000:3200]])
    with np.load(ROOT/'inputs/base_map/params.npz',allow_pickle=False) as base:
        base_fields={k:{'shape':list(base[k].shape),'dtype':str(base[k].dtype),'array_sha256':ah(base[k])} for k in base.files}
        stored=base['gt_w2c_all_frames'];relative=np.stack([np.linalg.inv(v)@c2w[0] for v in c2w])
        pose_checks={'stored_vs_absolute_w2c_max_abs':float(np.max(np.abs(stored-np.linalg.inv(c2w)))),
                     'stored_vs_first_relative_w2c_max_abs':float(np.max(np.abs(stored-relative)))}
        count=len(base['means3D']);id_unique=len(np.unique(base['source_gaussian_index']))==count
        intrinsics=base['intrinsics'].tolist()
    map_hash=sha(ROOT/'inputs/base_map/params.npz')
    dump(out/'base_validation.json',{'sha256':map_hash,'hash_match':map_hash==EXPECTED,'num_gaussians':count,'stable_ids_unique':id_unique,'fields':base_fields,'intrinsics':intrinsics})
    dump(out/'calibration_pose_validation.json',{'calib':calib,'stereo_baseline_m':float(abs(p3[0,3]/p3[0,0]-p2[0,3]/p2[0,0])),
        'P2_K':p2[:,:3].tolist(),'base_K_matches_P2':bool(np.allclose(intrinsics,p2[:,:3],atol=1e-5)),
        'pose_source':str(poses_path),'pose_rows':len(poses),'window_pose_shape':list(c2w.shape),'poses_finite':bool(np.isfinite(c2w).all()),
        'window_times':times[3000:3200].tolist(),'dt_min':float(np.diff(times[3000:3200]).min()),'dt_max':float(np.diff(times[3000:3200]).max()),'base_pose_checks':pose_checks})
    # GT discovery is based on inventories, not on consumed Test predictions.
    gt_dirs=[]
    for start in [Path('/root/autodl-tmp/slam_project/data/kitti/dataset'),Path('/root/autodl-tmp/datasets')]:
        for parent,dirs,files in os.walk(start):
            rel=Path(parent).relative_to(start)
            dirs[:]=[d for d in dirs if d not in ['Scene20','.git','outputs','prediction_root']]
            if len(rel.parts)>=7:dirs[:]=[]
            for d in dirs:
                low=d.lower()
                if low in ['classsegmentation','semanticsegmentation','velodyne','labels','semantic','semantic_labels']:
                    folder=Path(parent)/d
                    gt_dirs.append({'path':str(folder),'sample':[x.name for x in list(folder.iterdir())[:6]]})
    dump(out/'gt_discovery.json',{'searched_roots':['slam_project/data/kitti/dataset','datasets'],'excluded':['Scene20','outputs','prediction_root'],
        'candidate_dirs':gt_dirs,'independent_KITTI08_dense_semantic_gt':'NOT_FOUND_IN_AUDITED_INPUTS',
        'candidate_note':'folder names alone are not GT provenance; predicted labels remain predictions'})
    with open(out/'input_inventory.csv','w',encoding='utf-8-sig',newline='') as f:
        keys=sorted({k for row in rows for k in row});w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    semantic_rows=[e for e in rows if e['kind'] in ['labels','confidence']]
    summary={'audit_date':'2026-10-07','cpu_only':True,'host':platform.node(),'files_checked':len(rows),
       'counts':{k:sum(e['exists'] for e in rows if e['kind']==k) for k in ['image_2','image_3','labels','confidence','depth_sgbm_candidate']},
       'old_semantic_cache_hash_matches':sum(e.get('matches_20261006_cache',False) for e in semantic_rows),
       'base_hash_match':map_hash==EXPECTED,'base_gaussians':count,'metadata':metadata,'issues':issues,
       'readiness':'READY_STEP01_ENGINEERING_ONLY' if not issues and map_hash==EXPECTED else 'INPUT_ISSUES',
       'independent_semantic_gt':'NOT_FOUND_IN_AUDITED_INPUTS','historical_RAFT_depth':'OLD_PATH_REMOVED',
       'SGBM_depth':'AVAILABLE_CANDIDATE_NOT_SILENTLY_ADOPTED','science_ready':False,'recovered_root':str(recovery)}
    dump(out/'audit_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['metadata']},ensure_ascii=False))

if __name__=='__main__':main()
