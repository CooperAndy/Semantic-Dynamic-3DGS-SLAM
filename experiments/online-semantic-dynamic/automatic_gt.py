"""Evaluation-only official LiDAR labels. Sparse support is never densified."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

MOVING=np.array([252,253,254,255,256,257,258,259],np.uint16)
STATIC_MAPPED=np.array([10,11,13,15,16,18,30,31,32,40,48,50,51,60,70,71,72,80,81],np.uint16)
sys.path.insert(0,'/root/autodl-tmp/stage3_chy_20261006/scripts')
from semantickitti_projection import calibration,project


def official_gt(point_file,label_file,calib_file,shape):
    points=np.fromfile(point_file,dtype=np.float32).reshape(-1,4)
    packed=np.fromfile(label_file,dtype=np.uint32)
    if len(points)!=len(packed):raise ValueError('Point/label identity mismatch')
    P,T=calibration(Path(calib_file));result,stats,_=project(points,packed,P,T,*shape)
    raw=result['raw_semantic'];supported=result['point_index']>=0
    result['moving']=supported&np.isin(raw,MOVING)
    result['static']=supported&np.isin(raw,STATIC_MAPPED)
    result['motion_known']=result['moving']|result['static']
    return result,stats


def sparse_metrics(rgb,target,depth,silhouette,predicted_semantics,gt):
    static=gt['static'];count=int(static.sum())
    if not count:raise ValueError('No independent static GT support')
    mse=float(np.mean((rgb[:,static].astype(np.float64)-target[:,static].astype(np.float64))**2))
    # Missing depth contributes the GT depth error; never shrink the denominator.
    rendered=np.divide(depth,silhouette,out=np.zeros_like(depth),where=silhouette>1e-4)
    rendered=np.where(np.isfinite(rendered)&(silhouette>=.05),rendered,0)
    mae=float(np.abs(rendered[static]-gt['depth'][static]).mean())
    result=dict(static_GT_pixels=count,static_RGB_PSNR_dB=-10*float(np.log10(max(mse,1e-20))),
                static_depth_MAE_m=mae,static_depth_coverage=float((silhouette[static]>=.05).mean()),
                motion_GT_pixels=int(gt['motion_known'].sum()),moving_GT_pixels=int(gt['moving'].sum()))
    if predicted_semantics is not None:
        truth=gt['label'];known=truth!=255;pred=np.asarray(predicted_semantics)
        ious=[]
        for c in range(19):
            union=((truth==c)|(pred==c))&known
            if union.any():ious.append(float((((truth==c)&(pred==c))&known).sum()/union.sum()))
        result.update(sparse_mIoU=float(np.mean(ious)) if ious else None,semantic_GT_pixels=int(known.sum()),
                      semantic_label_coverage=float(((pred>=0)&(pred<19))[known].mean()))
    raw=gt.get('raw_semantic')
    if raw is not None:
        vehicles=static&np.isin(raw,[10,13,16,18])
        n=int(vehicles.sum());result['static_vehicle_GT_pixels']=n
        result['static_vehicle_depth_coverage']=float((silhouette[vehicles]>=.05).mean()) if n else None
        if predicted_semantics is not None:
            result['static_vehicle_label_coverage']=float(((pred>=0)&(pred<19))[vehicles].mean()) if n else None
            result['static_vehicle_semantic_accuracy']=float((pred[vehicles]==gt['label'][vehicles]).mean()) if n else None
    return result


def main():
    from experiment import REMOTE
    R=REMOTE/'evaluation_only/official_gt_v2';R.mkdir(parents=True,exist_ok=False)
    old=Path('/root/autodl-tmp/stage3_chy_20261006/inputs/semantickitti08_v1')
    calib=old/'calib.txt'
    assert hashlib.sha256(calib.read_bytes()).hexdigest()=='5e7b3136edf2799be47e472ca9c5362fcacbee5d21c6ab8038506b8691b8b9ec'
    rows=[]
    for i in range(200):
        frame=3000+i;point=old/'velodyne'/f'{frame:06}.bin';label=old/'labels'/f'{frame:06}.label'
        gt,stats=official_gt(point,label,calib,(370,1226))
        f=R/f'{frame:06}.npz';np.savez_compressed(f,**gt)
        rows.append(dict(frame=frame,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                         point_sha256=hashlib.sha256(point.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
                         static_pixels=int(gt['static'].sum()),moving_pixels=int(gt['moving'].sum())))
    receipt=dict(status='OFFICIAL_SPARSE_STATIC_MOVING_GT_READY',source='SemanticKITTI official raw point labels',
                 no_manual_annotation=True,dense_ROI=False,rows=rows)
    (R/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))

if __name__=='__main__':main()
