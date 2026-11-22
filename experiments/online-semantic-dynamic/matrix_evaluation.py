"""Per-run sealed evaluation for the registered1000-frame cells, no manual labels."""
import argparse,hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
import numpy as np
from experiment import REMOTE

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def references(R):
    from automatic_gt import official_gt
    m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());reg=json.loads((R/'registration.json').read_text());seq=reg['cell']['sequence']
    E=REMOTE/'evaluation_only'/('K'+seq);E.mkdir(parents=True,exist_ok=True)
    old=REMOTE/'legacy-gt/semantickitti08_v1'
    with zipfile.ZipFile(old/'official_calibration.zip') as z:
        names=[n for n in z.namelist() if n.endswith('/'+seq+'/calib.txt')];assert len(names)==1
        calib=E/'calib.txt'
        if not calib.exists():calib.write_bytes(z.read(names[0]))
    c={}
    for line in calib.read_text().splitlines():
        key,val=line.split(':',1);c[key]=np.fromstring(val,sep=' ')
    K=c['P2'].reshape(3,4)[:,:3];assert np.allclose(K,m['K'],rtol=0,atol=5e-4)
    poses=REMOTE/'legacy-gt/poses'/(seq+'.txt')
    if not poses.exists():raise FileNotFoundError('Official trajectory GT is missing; retain predictions: '+str(poses))
    raw=np.loadtxt(poses,dtype=np.float64);ids=np.array([r['source_frame_id'] for r in m['rows']],int)
    G0=np.tile(np.eye(4),(len(ids),1,1));G0[:,:3]=raw[ids].reshape(-1,3,4)
    C0_from_C2=np.eye(4);C0_from_C2[:3,3]=-np.linalg.solve(K,c['P2'].reshape(3,4)[:,3]);G=G0@C0_from_C2
    gtpath=E/'trajectory_reference.npz'
    if not gtpath.exists():np.savez_compressed(gtpath,P2_c2w=G,source_frame_ids=ids)
    frames=[19+20*i for i in range(10)]+[819+20*i for i in range(10)]
    sources={}
    for kind,extension,url in [('velodyne','bin','https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip'),('labels','label','https://semantic-kitti.org/assets/data_odometry_labels.zip')]:
        folder=E/kind;folder.mkdir(exist_ok=True);archive=None;records=[]
        for frame in frames:
            source=m['rows'][frame]['source_frame_id'];p=folder/f'{source:06}.{extension}'
            if not p.exists():
                legacy=old/kind/p.name
                if seq=='08' and legacy.exists():p.hardlink_to(legacy)
                else:
                    sys.path.insert(0,str(REMOTE/'legacy-gt/scripts'))
                    from http_range_zip import RangeZip
                    if archive is None:archive=RangeZip(url,256*2**20)
                    names=[n for n in archive.info if n.endswith(f'/{seq}/{kind}/{source:06}.{extension}')];assert len(names)==1
                    p.write_bytes(archive.read(names[0]))
            records.append(dict(frame=source,path=str(p),sha256=sha(p)))
        sources[kind]=records
    for frame in frames:
        source=m['rows'][frame]['source_frame_id'];p=E/f'{source:06}_GT.npz'
        if not p.exists():
            gt,stats=official_gt(E/'velodyne'/f'{source:06}.bin',E/'labels'/f'{source:06}.label',calib,(m['H'],m['W']))
            np.savez_compressed(p,**gt)
    manifest=E/'reference_receipt.json'
    if not manifest.exists():manifest.write_text(json.dumps(dict(status='OFFICIAL20_VIEW_SPARSE_REFERENCES_READY',sequence=seq,GT_prediction=False,source_files=sources,pose_sha256=sha(poses),calib_sha256=sha(calib),dense_ROI=False),indent=2))
    return E,G


def evaluate(R,group,kitti_root=None,labels_root=None):
    import torch,lpips
    from PIL import Image,ImageDraw
    from online_state import OnlineState
    from semantic_mapping import render_channels
    from automatic_gt import sparse_metrics
    R=Path(R);d=R/'runs'/group;out=d/'evaluation';out.mkdir(exist_ok=False)
    from resource_audit import snapshot as resource_snapshot
    resources=resource_snapshot(REMOTE,R,group,out)
    from source_snapshot import capture_evaluation
    capture_evaluation(out)
    assert json.loads((d/'independent_numeric_audit.json').read_text())['status'].startswith('PASS_')
    if kitti_root is not None:
        from reproduction_references import prepare_references
        E,G=prepare_references(R,kitti_root,labels_root)
    else:
        E,G=references(R)
    m=json.loads((R/'inputs/runtime/input_manifest.json').read_text())
    from input_provenance import capture as capture_inputs
    input_identity=capture_inputs(R,group,out,REMOTE,E,portable=kitti_root is not None)
    sys.path[:0]=[str(R/'scripts'),str(R/'source')]
    spec=importlib.util.spec_from_file_location('matrix_frozen_eval_core',R/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    import external_tracking as ext
    from utils.recon_helpers import setup_camera
    from trajectory_metrics import evaluate as trajectory_metrics
    from semantickitti_projection import PALETTE
    from tracking_audit import audit_tracking
    torch.set_num_threads(4)
    f=d/'params.npz';before=sha(f)
    with np.load(f,allow_pickle=False) as z:
        maps={k:torch.tensor(z[k],device='cuda') for k in ext.MAP_KEYS};K=z['intrinsics'];W0=z['w2c'];q=z['cam_unnorm_rots'].copy();t=z['cam_trans'].copy()
    with np.load(d/'estimated_trajectory.npz') as z:estimate=np.linalg.inv(z['w2c'].astype(np.float64))
    track,arrays=trajectory_metrics(estimate,G);np.savez_compressed(out/'trajectory_arrays.npz',**arrays)
    tracking_diagnostics=audit_tracking(d)
    raw=np.loadtxt(m['pose_file'],dtype=np.float64);nominal=np.tile(np.eye(4),(1000,1,1));nominal[:,:3]=raw.reshape(-1,3,4)
    orb,_=trajectory_metrics(nominal,G);Grel=np.linalg.inv(G[0])@G
    ids=np.load(d/'source_gaussian_index.npy',allow_pickle=False);state=OnlineState.load(d/'semantic_state_final.npz',ids)
    total=state.votes.sum(1);known=total>0;features=np.zeros((len(ids),20),np.float32);features[known,:19]=state.votes[known]/total[known,None];features[~known,19]=1
    features=torch.as_tensor(features,device='cuda');data=dict(cam=setup_camera(m['W'],m['H'],K,W0),w2c=torch.tensor(W0,device='cuda'))
    from semantic_confusion import count_view,class_statistics,write_counts
    perceptual=lpips.LPIPS(net='alex',version='0.1',verbose=False).cpu().eval();rows=[];class_views=[]
    frames=list(range(19,200,20))+list(range(819,1000,20))
    for mode in ['COMMON_GT','COMMON_ORB','OWN_SELECTED']:
        params=dict(maps,cam_unnorm_rots=torch.tensor(q,device='cuda'),cam_trans=torch.tensor(t,device='cuda'))
        for i in frames:
            if mode!='OWN_SELECTED':
                W=np.linalg.inv(Grel[i] if mode=='COMMON_GT' else nominal[i]);ext.set_external(params,torch.tensor(W,dtype=torch.float32,device='cuda'),i,core)
            source=m['rows'][i]['source_frame_id']
            with Image.open(m['rows'][i]['rgb']) as im:target=np.asarray(im.convert('RGB'),np.float32).transpose(2,0,1)/255.
            rgb,ds=ext.render(params,data,i,core)
            with torch.no_grad():mass=render_channels(core,params,data,i,features,False).cpu().numpy()
            alpha=mass.sum(0);assert np.allclose(alpha,ds[1],atol=5e-5,rtol=1e-4)
            labels=np.full(alpha.shape,-2,np.int16);supported=alpha>=.05;labels[supported]=-1
            accepted=supported&(mass[:19].sum(0)>0)&(mass[19]<.5*alpha);labels[accepted]=mass[:19].argmax(0)[accepted]
            with np.load(E/f'{source:06}_GT.npz') as z:gt={k:z[k] for k in z.files}
            sparse=sparse_metrics(rgb,target,ds[0],ds[1],labels,gt)
            class_row=class_statistics(count_view(labels,gt['label']))
            assert abs(class_row['sparse_mIoU']-sparse['sparse_mIoU'])<1e-12
            class_views.append(dict(mode=mode,frame=i,source_frame=source,cohort='EARLY' if i<200 else 'RECENT',**class_row))
            with torch.no_grad():
                aa=torch.from_numpy(rgb)[None];bb=torch.from_numpy(target)[None];ssim=float(core.calc_ssim(aa,bb));lp=float(perceptual(aa.clamp(0,1),bb,normalize=True))
            mse=float(np.mean((rgb.astype(np.float64)-target.astype(np.float64))**2))
            row=dict(mode=mode,frame=i,source_frame=source,cohort='EARLY' if i<200 else 'RECENT',PSNR_dB=-10*float(np.log10(mse)),SSIM=ssim,LPIPS=lp,coverage=float((ds[1]>.99).mean()),**sparse)
            rows.append(row);np.savez_compressed(out/f'{mode}_{i:06}.npz',rgb=rgb,target=target,depth_silhouette=ds,semantic=labels,w2c=ext.camera(params,i,core).cpu().numpy(),intrinsics=K)
            if mode=='COMMON_GT' and i in [19,999]:
                panels=[np.uint8(np.rint(np.clip(x.transpose(1,2,0),0,1)*255)) for x in [target,rgb]]
                color=np.zeros((*labels.shape,3),np.uint8);color[labels==-1]=96
                positive=(labels>=0)&(labels<19);color[positive]=PALETTE[labels[positive]];panels.append(color)
                canvas=Image.new('RGB',(m['W']*3,m['H']+28),'white')
                for n,array in enumerate(panels):canvas.paste(Image.fromarray(array),(n*m['W'],28))
                draw=ImageDraw.Draw(canvas);draw.text((8,5),f'{group} / source {source} / shared GT camera / observation | RGB map | semantic map',fill='black')
                canvas.save(out/f'typical_{source:06}.png')
    assert len(rows)==60 and sha(f)==before
    write_counts(out/'semantic_confusion.json',class_views)
    names=['PSNR_dB','SSIM','LPIPS','coverage','static_RGB_PSNR_dB','static_depth_MAE_m','static_depth_coverage','sparse_mIoU','semantic_label_coverage','static_vehicle_depth_coverage','static_vehicle_label_coverage','static_vehicle_semantic_accuracy']
    summary=[]
    for mode in ['COMMON_GT','COMMON_ORB','OWN_SELECTED']:
        for cohort in ['EARLY','RECENT','ALL']:
            selected=[r for r in rows if r['mode']==mode and (cohort=='ALL' or r['cohort']==cohort)]
            summary.append(dict(mode=mode,cohort=cohort,views=len(selected),means={k:float(np.mean([r[k] for r in selected if r[k] is not None])) if any(r[k] is not None for r in selected) else None for k in names}))
    result=dict(status='COMPLETE60_FIXED_VIEWS_AND_FULL_TRAJECTORY',group=group,trajectory=track,tracking_diagnostics=tracking_diagnostics,ORB_reference=orb,summary=summary,rows=rows,map_sha256=before,geometry_unchanged=True,new_manual_annotation=False,GT_prediction=False,quality_adoption=False,resource_audit=resources)
    from report_views import preserve
    result['publication_views']=preserve(R,group,REMOTE)
    result['input_provenance']=input_identity
    result['semantic_confusion']=dict(path='semantic_confusion.json',sha256=sha(out/'semantic_confusion.json'),primary_metrics_changed=False)
    (out/'results.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--group',required=True)
    p.add_argument('--kitti-root',type=Path)
    p.add_argument('--labels-root',type=Path)
    a=p.parse_args()
    if bool(a.kitti_root)!=bool(a.labels_root):
        p.error('--kitti-root and --labels-root must be supplied together')
    print(json.dumps(evaluate(a.root,a.group,a.kitti_root,a.labels_root)['trajectory']))
