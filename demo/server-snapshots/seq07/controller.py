"""Independent seq07 display tuning; all products stay on the server."""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, traceback, zipfile
import numpy as np
from PIL import Image

R=Path('/root/autodl-tmp/seq07_000850_001049_tuning_20261115')
OLD=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
BATCH=OLD.parent; SPLAT=Path('/root/autodl-tmp/SplaTAM'); PY='/root/miniconda3/bin/python'; RAFT='/root/venvs/raft_stereo/bin/python'
ZIP=Path('/root/autodl-tmp/slam_project/data/kitti/data_odometry_color.zip'); S=R/'scripts'; SEQ=R/'sequences/07'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state(stage, **kw):
    p=R/'status.json'; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(dict(stage=stage,time=time.strftime('%FT%T%z'),**kw),indent=2)); tmp.replace(p); print(stage,kw,flush=True)
def run(name,cmd,gpu=False,py_path=None):
    if shutil.disk_usage(R).free < 8*2**30: raise RuntimeError('Free disk below 8 GiB; stopped without deleting existing files')
    state(name)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TASK_GPU_GIB='6',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=py_path or f'{S}:{SPLAT}')
    cmd=list(map(str,cmd)); cmd=([cmd[0],str(S/'resource_guard.py')]+cmd[1:]) if gpu else cmd
    with (R/f'{name}.log').open('x') as f: subprocess.run(cmd,cwd=SPLAT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    state('extract_inputs'); SEQ.mkdir(parents=True); S.mkdir()
    manifest=[]
    with zipfile.ZipFile(ZIP) as z:
        for cam in (2,3):
            folder=SEQ/f'image_{cam}';folder.mkdir()
            for i in range(200):
                src=f'dataset/sequences/07/image_{cam}/{850+i:06d}.png'; dest=folder/f'{i:06d}.png'
                with z.open(src) as a,dest.open('wb') as b:shutil.copyfileobj(a,b)
                with Image.open(dest) as im:
                    if i==0 and cam==2: width,height=im.size
                    assert im.size==(width,height);im.verify()
                manifest.append(dict(local=i,source=850+i,camera=cam,sha256=sha(dest)))
        for name in ('calib.txt','times.txt'):
            raw=z.read(f'dataset/sequences/07/{name}').decode()
            if name=='times.txt':raw='\n'.join(raw.splitlines()[850:1050])+'\n'
            (SEQ/name).write_text(raw)
    (R/'input_manifest.json').write_text(json.dumps(manifest,indent=2)); assert len(manifest)==400
    job=dict(sequence='07',start=850,end=1049,count=200,width=width,height=height,job='seq07_850_1049',remote_root=str(R))
    (R/'input_spec.json').write_text(json.dumps(job,indent=2));shutil.copytree(OLD/'vendor',R/'vendor')
    (R/'protocol.json').write_text(json.dumps(dict(job=job,thresholds=[.5,.6,.7],threshold_seed=0,extra_seeds=[42,2026],f1_steps=800,selection='mean full image PSNR across 13 fixed seen views, subject to visual review',GT_input=False),indent=2))
    for p in (OLD/'scripts').glob('*.py'):
        text=p.read_text().replace(str(OLD),str(R)).replace('sequences/09','sequences/07')
        text=re.sub(r'\b1226\b',str(width),text);text=re.sub(r'\b370\b',str(height),text)
        (S/p.name).write_text(text)
    spec=importlib.util.spec_from_file_location('old_prepare',BATCH/'server_prepare.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state('orb_localization');module.orb(job)
    # Verify the adapted stereo/depth/mapper chain on four frames before the full run.
    sample=R/'smoke_sequences/07';sample.mkdir(parents=True)
    for cam in (2,3):
        (sample/f'image_{cam}').mkdir()
        for i in range(4):(sample/f'image_{cam}'/f'{i:06d}.png').symlink_to(SEQ/f'image_{cam}'/f'{i:06d}.png')
    shutil.copy2(SEQ/'calib.txt',sample/'calib.txt');(sample/'times.txt').write_text('\n'.join((SEQ/'times.txt').read_text().splitlines()[:4])+'\n')
    small=['--raft-root','/root/autodl-tmp/slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo','--checkpoint','/root/raft_stereo_models/raftstereo-middlebury.pth','--sequence-dir',sample,'--num-frames','4','--max-depth','40']
    run('smoke_left',[RAFT,S/'generate_raft_depth_kitti.py',*small,'--output-dir',R/'smoke_left','--preview-indices','0','3'],True)
    run('smoke_right',[RAFT,S/'generate_raft_reverse_depth_kitti.py',*small,'--output-dir',R/'smoke_right'],True)
    np.savetxt(R/'smoke_poses.txt',np.loadtxt(R/'poses_estimated_camera2.txt')[:4])
    run('smoke_confidence',[PY,S/'build_depth_confidence_kitti.py','--left-image-dir',sample/'image_2','--right-image-dir',sample/'image_3','--depth-left-dir',R/'smoke_left','--depth-right-dir',R/'smoke_right','--pose-path',R/'smoke_poses.txt','--calib',sample/'calib.txt','--output-root',R/'smoke_confidence','--num-frames','4','--seed-threshold','.60','--loss-threshold','.20','--preview-indices','0','3'])
    sy=json.loads((R/'kitti.yaml').read_text());sy.update(depth_folder=str(R/'smoke_confidence/depth_confhard'),pose_path=str(R/'smoke_poses.txt'));(R/'smoke.yaml').write_text(json.dumps(sy))
    sc=ast.literal_eval(ast.parse((OLD/'scripts/mapping_config.py').read_text()).body[0].value);sc.update(workdir=str(R/'runs'),run_name='smoke',seed=0,save_checkpoints=False);sc['data'].update(basedir=str(R/'smoke_sequences'),sequence='07',num_frames=4,desired_image_width=width,desired_image_height=height,gradslam_data_cfg=str(R/'smoke.yaml'));sc['mapping']['num_iters']=5;(R/'smoke_config.py').write_text('config = '+repr(sc)+'\n')
    run('smoke_mapping',[PY,S/'splatam_isolated.py',R/'smoke_config.py'],True);assert (R/'runs/smoke/params.npz').is_file()
    common=['--raft-root','/root/autodl-tmp/slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo','--checkpoint','/root/raft_stereo_models/raftstereo-middlebury.pth','--sequence-dir',SEQ,'--num-frames','200','--max-depth','40']
    run('raft_left',[RAFT,S/'generate_raft_depth_kitti.py',*common,'--output-dir',R/'raft_left','--preview-indices','0','99','199'],True)
    run('raft_right',[RAFT,S/'generate_raft_reverse_depth_kitti.py',*common,'--output-dir',R/'raft_right'],True)
    run('confidence',[PY,S/'build_depth_confidence_kitti.py','--left-image-dir',SEQ/'image_2','--right-image-dir',SEQ/'image_3','--depth-left-dir',R/'raft_left','--depth-right-dir',R/'raft_right','--pose-path',R/'poses_estimated_camera2.txt','--calib',SEQ/'calib.txt','--output-root',R/'confidence','--num-frames','200','--seed-threshold','.60','--loss-threshold','.20','--preview-indices','0','99','199'])
    (R/'reliable_roi').mkdir(); coverage={str(t):[] for t in (50,60,70)}
    for t in (50,70):(R/f'depth_{t}').mkdir()
    for i in range(200):
        raw=np.load(R/f'raft_left/{i:06d}_depth.npy',allow_pickle=False); c=np.load(R/f'confidence/confidence/{i:06d}_confidence.npy',allow_pickle=False); base=np.load(R/f'confidence/depth_confhard/{i:06d}_depth.npy',allow_pickle=False)
        assert raw.shape==c.shape==base.shape==(height,width) and np.isfinite(raw).all() and np.isfinite(c).all() and np.isfinite(base).all()
        np.save(R/f'reliable_roi/{i:06d}_valid.npy',base>0);coverage['60'].append(float((base>0).mean()))
        for t in (50,70):
            d=np.where(c>=t/100,raw,0).astype('float32'); assert (d>0).any();np.save(R/f'depth_{t}/{i:06d}_depth.npy',d);coverage[str(t)].append(float((d>0).mean()))
    (R/'coverage.json').write_text(json.dumps(coverage,indent=2))
    # Same camera, full-resolution target and fixed baseline ROI for every candidate.
    ev=(BATCH.parent/'seq09_300_499_demo_tuning_20261114/f1_confidence_tuning/evaluate_f1_candidates.py').read_text().replace(str(OLD),str(R)).replace('sequences/09','sequences/07').replace('source_frame=300+i','source_frame=850+i')
    ev=re.sub(r'\b1226\b',str(width),ev);ev=re.sub(r'\b370\b',str(height),ev);(S/'evaluate.py').write_text(ev)
    evalpy='/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python'; raster='/root/autodl-tmp/s23_effects_20261105_p16247/raster_fp64_accum_v1'; evaluation=R/'evaluation';evaluation.mkdir()
    cfg=ast.literal_eval(ast.parse((OLD/'scripts/mapping_config.py').read_text()).body[0].value)
    results={}
    def candidate(t,seed):
        name=f'c{t}_s{seed}'; out=R/name;out.mkdir();y=json.loads((R/'kitti.yaml').read_text());y['depth_folder']=str(R/'confidence/depth_confhard' if t==60 else R/f'depth_{t}');(out/'kitti.json').write_text(json.dumps(y,indent=2))
        import copy
        c=copy.deepcopy(cfg);c.update(workdir=str(out),run_name='map',seed=seed,save_checkpoints=False);c['data'].update(basedir=str(R/'sequences'),sequence='07',desired_image_width=width,desired_image_height=height,num_frames=200,gradslam_data_cfg=str(out/'kitti.json'))
        (out/'mapping_config.py').write_text('config = '+repr(c)+'\n');run(name+'_mapping',[PY,S/'splatam_isolated.py',out/'mapping_config.py'],True)
        m=out/'map/params.npz';assert m.is_file()
        run(name+'_f1',[PY,S/'train_f1_seq04.py','--map',m,'--expected-map-sha256',sha(m),'--out',out/'F1','--train-frame-ids',*map(str,range(200)),'--steps','800','--protocol-root',R/'deployment','--run-kind','f1_all200'],True)
        f=out/'F1/params.npz'; assert f.is_file()
        with np.load(f,allow_pickle=False) as x:assert x['cam_trans'].shape[-1]==200
        run(name+'_evaluate',[evalpy,S/'evaluate.py','--name',name,'--map',f,'--out',evaluation],py_path=f'{raster}:{BATCH.parent}/seq09_300_499_demo_tuning_20261114/S0/source')
        result=json.loads((evaluation/f'{name}.json').read_text());result['sha256']=sha(f);results[name]=result
        (R/'results_partial.json').write_text(json.dumps(results,indent=2))
    for t in (60,50,70):candidate(t,0)
    best=max((50,60,70),key=lambda t:results[f'c{t}_s0']['mean_psnr']);(R/'threshold_selection.json').write_text(json.dumps(dict(threshold=best,criterion='mean PSNR on seen views',visual_review_pending=True),indent=2))
    for seed in (42,2026):candidate(best,seed)
    state('complete',selected_threshold=best,visual_review_pending=True,candidates=list(results),results=str(R/'results_partial.json'),maps_retained=True)
try:main()
except BaseException as e:state('failed',error=repr(e),traceback=traceback.format_exc());raise
