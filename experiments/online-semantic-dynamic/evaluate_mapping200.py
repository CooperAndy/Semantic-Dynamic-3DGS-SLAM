"""Frozen E1 maps, common ORB/GT cameras and official sparse independent support."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch,lpips
from PIL import Image
from experiment import REMOTE
from automatic_gt import sparse_metrics
from online_state import OnlineState
from semantic_mapping import render_channels

R=REMOTE/'E1_mapping200_v2';out=R/'evaluation';out.mkdir(exist_ok=False)
registration=json.loads((R/'registration.json').read_text());m=json.loads((R/'inputs/runtime/input_manifest.json').read_text())
sys.path[:0]=[str(R/'scripts'),str(R/'source')]
spec=importlib.util.spec_from_file_location('E1_frozen_eval_core',R/'source/scripts/splatam_external_tm.py')
core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
import external_tracking as ext
from utils.recon_helpers import setup_camera
torch.set_num_threads(4);perceptual=lpips.LPIPS(net='alex',version='0.1',verbose=False).cpu().eval()
reference=Path('/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181/evaluation_only/B03_MASS_S0_metrics.npz')
with np.load(reference,allow_pickle=False) as z:G=z['ground_truth_P2_c2w'].copy()
G=np.linalg.inv(G[0])@G
with np.load(R/'runs/MAP_BASE/params.npz',allow_pickle=False) as z:qbase=z['cam_unnorm_rots'].copy();tbase=z['cam_trans'].copy()
rows=[];proofs=[]
for variant in registration['variants']:
    source=R/'runs'/variant/'params.npz';before=hashlib.sha256(source.read_bytes()).hexdigest()
    receipt=json.loads((source.parent/'receipt.json').read_text());assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION'
    with np.load(source,allow_pickle=False) as z:
        qown=z['cam_unnorm_rots'].copy();town=z['cam_trans'].copy()
        camera_exact_equal=np.array_equal(qown,qbase) and np.array_equal(town,tbase)
        assert qown.shape==qbase.shape and town.shape==tbase.shape
        maps={k:torch.tensor(z[k],device='cuda') for k in ext.MAP_KEYS};K=z['intrinsics'].copy();W0=z['w2c'].copy()
    # Same sealed ORB source; tolerate only the already registered FP32 pose bound.
    check=dict(maps,cam_unnorm_rots=torch.tensor(qown,device='cuda'),cam_trans=torch.tensor(town,device='cuda'))
    baseline=dict(maps,cam_unnorm_rots=torch.tensor(qbase,device='cuda'),cam_trans=torch.tensor(tbase,device='cuda'))
    max_dt=0.;max_angle=0.
    from scipy.spatial.transform import Rotation
    for frame in range(200):
        A=ext.camera(check,frame,core).cpu().numpy().astype(np.float64);B=ext.camera(baseline,frame,core).cpu().numpy().astype(np.float64)
        delta=A@np.linalg.inv(B);max_dt=max(max_dt,float(np.linalg.norm(delta[:3,3])))
        max_angle=max(max_angle,float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude())))
    assert max_dt<=.0003 and max_angle<=.002,'Fixed ORB numerical pose bound failed'
    ids=np.load(source.parent/'source_gaussian_index.npy',allow_pickle=False)
    state=OnlineState.load(source.parent/'semantic_state_final.npz',ids)
    assert len(ids)==len(maps['means3D'])
    total=state.votes.sum(1);features=np.zeros((len(ids),20),np.float32)
    known=total>0;features[known,:19]=state.votes[known]/total[known,None];features[~known,19]=1
    feature=torch.as_tensor(features,device='cuda');base=dict(cam=setup_camera(m['W'],m['H'],K,W0),w2c=torch.tensor(W0,device='cuda'))
    for mode in ['COMMON_ORB','COMMON_GT']:
        p=dict(maps,cam_unnorm_rots=torch.tensor(qbase,device='cuda'),cam_trans=torch.tensor(tbase,device='cuda'))
        for frame in range(19,200,20):
            if mode=='COMMON_GT':ext.set_external(p,torch.tensor(np.linalg.inv(G[frame]),dtype=torch.float32,device='cuda'),frame,core)
            with Image.open(m['rows'][frame]['rgb']) as im:target=np.asarray(im.convert('RGB'),np.float32).transpose(2,0,1)/255.
            rgb,ds=ext.render(p,base,frame,core)
            with torch.no_grad():
                mass=render_channels(core,p,base,frame,feature,False).cpu().numpy()
            alpha=mass.sum(0)
            # Include unknown channel in opacity; do not render only labelled points.
            assert np.allclose(alpha,ds[1],atol=5e-5,rtol=1e-4)
            labels=np.full(alpha.shape,-2,np.int16);supported=alpha>=.05;labels[supported]=-1
            accepted=supported&(mass[:19].sum(0)>0)&(mass[19]<.5*alpha);labels[accepted]=mass[:19].argmax(0)[accepted]
            with np.load(REMOTE/'evaluation_only/official_gt_v2'/f'{3000+frame:06}.npz') as z:gt={k:z[k] for k in z.files}
            sparse=sparse_metrics(rgb,target,ds[0],ds[1],labels,gt)
            aa=torch.from_numpy(rgb)[None];bb=torch.from_numpy(target)[None]
            with torch.no_grad():ssim=float(core.calc_ssim(aa,bb));lp=float(perceptual(aa.clamp(0,1),bb,normalize=True))
            mse=float(np.mean((rgb.astype(np.float64)-target.astype(np.float64))**2))
            f=out/f'{variant}_{mode}_{frame:06}.npz'
            np.savez_compressed(f,rgb=rgb,target=target,depth_silhouette=ds,semantic=labels,intrinsics=K,w2c=ext.camera(p,frame,core).cpu().numpy())
            row=dict(variant=variant,mode=mode,frame=frame,source_frame=frame+3000,PSNR_dB=-10*float(np.log10(mse)),
                     SSIM=ssim,LPIPS=lp,coverage=float((ds[1]>.99).mean()),**sparse)
            rows.append(row)
            with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    proofs.append(dict(variant=variant,map_sha256=before,gaussians=len(ids),camera_exact_equal=camera_exact_equal,
                       max_camera_difference_m=max_dt,max_camera_difference_deg=max_angle,common_render_camera='MAP_BASE_ORB or shared GT',geometry_unchanged=True))
    del maps,p,feature,check,baseline;torch.cuda.empty_cache()
metrics=['PSNR_dB','SSIM','LPIPS','coverage','static_RGB_PSNR_dB','static_depth_MAE_m','static_depth_coverage','sparse_mIoU','semantic_label_coverage']
groups=[]
for mode in ['COMMON_ORB','COMMON_GT']:
    for variant in registration['variants']:
        rr=[x for x in rows if x['variant']==variant and x['mode']==mode]
        assert len(rr)==10
        groups.append(dict(variant=variant,mode=mode,views=10,means={k:float(np.mean([r[k] for r in rr])) for k in metrics}))
comparisons=[]
for mode in ['COMMON_ORB','COMMON_GT']:
    base=next(g['means'] for g in groups if g['mode']==mode and g['variant']=='MAP_BASE')
    for variant in ['MAP_SEMANTIC','MAP_SHUFFLED']:
        cur=next(g['means'] for g in groups if g['mode']==mode and g['variant']==variant)
        comparisons.append(dict(mode=mode,variant=variant,delta={k:cur[k]-base[k] for k in metrics}))
result=dict(status='COMPLETE_E1_MAPPING_60_VIEWS_OFFICIAL_SPARSE_GT',groups=groups,comparisons=comparisons,rows=rows,proofs=proofs,
            new_manual_labels=False,GT_in_prediction=False,quality_adoption=False,
            scope='Whole-image SSIM/LPIPS; sparse static RGB/depth and sparse semantic GT. No dense static ROI is claimed.')
(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(comparisons))
