"""Scene-tuned F1 comparison, isolated from previous maps."""
from pathlib import Path
import ast, hashlib, json, os, shutil, subprocess, time, traceback
import numpy as np

D=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
R=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114/f1_confidence_tuning')
PY='/root/miniconda3/bin/python'; S=D/'scripts'
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TASK_GPU_GIB='6',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=f'{S}:/root/autodl-tmp/SplaTAM')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(step, **kw):
    (R/'status.json').write_text(json.dumps(dict(step=step,time=time.strftime('%FT%T%z'),**kw),indent=2))
def run(cmd,log):
    if shutil.disk_usage(R).free < 12*2**30: raise RuntimeError('Less than 12 GiB free; stopping safely')
    with (R/log).open('x') as f: subprocess.run(list(map(str,cmd)),cwd='/root/autodl-tmp/SplaTAM',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def depth(threshold):
    dest=R/f'depth_{int(threshold*100):02d}'; dest.mkdir(exist_ok=False); coverage=[]
    for i in range(200):
        n=f'{i:06}'; raw=np.load(D/'raft_left'/f'{n}_depth.npy',allow_pickle=False); c=np.load(D/'confidence/confidence'/f'{n}_confidence.npy',allow_pickle=False)
        assert raw.shape==c.shape==(370,1226) and np.isfinite(raw).all() and np.isfinite(c).all()
        z=np.where(c>=threshold,raw,0).astype(np.float32); assert np.any(z>0)
        np.save(dest/f'{n}_depth.npy',z,allow_pickle=False); coverage.append(float((z>0).mean()))
    (R/f'coverage_{int(threshold*100):02d}.json').write_text(json.dumps(dict(threshold=threshold,mean=float(np.mean(coverage)),per_frame=coverage),indent=2))
    return dest
def mapping(threshold,seed):
    name=f'c{int(round(threshold*100)):02d}_s{seed}'; out=R/name; out.mkdir(exist_ok=False)
    target=D/'confidence/depth_confhard' if threshold==.6 else R/f'depth_{int(threshold*100):02d}'
    y=json.loads((D/'kitti.yaml').read_text()); y['depth_folder']=str(target); (out/'kitti.json').write_text(json.dumps(y,indent=2))
    cfg=ast.literal_eval(ast.parse((S/'mapping_config.py').read_text()).body[0].value)
    cfg.update(workdir=str(out),run_name='map',seed=seed,save_checkpoints=False)
    cfg['data']['gradslam_data_cfg']=str(out/'kitti.json')
    (out/'mapping_config.py').write_text('config = '+repr(cfg)+'\n')
    state('mapping',name=name,seed=seed,threshold=threshold)
    run([PY,S/'resource_guard.py',S/'splatam_isolated.py',out/'mapping_config.py'],f'{name}_mapping.log')
    m=out/'map/params.npz'; assert m.is_file()
    state('f1',name=name,seed=seed,threshold=threshold)
    run([PY,S/'resource_guard.py',S/'train_f1_seq04.py','--map',m,'--expected-map-sha256',sha(m),'--out',out/'F1','--train-frame-ids',*map(str,range(200)),'--steps','800','--protocol-root',D/'deployment','--run-kind','f1_all200'],f'{name}_f1.log')
    assert (out/'F1/params.npz').is_file(); return out/'F1/params.npz'
def evaluate(name,path):
    state('evaluate',name=name)
    evalenv=dict(env,PYTHONPATH='/root/autodl-tmp/s23_effects_20261105_p16247/raster_fp64_accum_v1:/root/autodl-tmp/seq09_300_499_demo_tuning_20261114/S0/source')
    with (R/f'{name}_eval.log').open('x') as f:
        subprocess.run(['/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python',str(R/'evaluate_f1_candidates.py'),'--name',name,'--map',str(path),'--out',str(R/'evaluation')],cwd='/root/autodl-tmp/SplaTAM',env=evalenv,stdout=f,stderr=subprocess.STDOUT,check=True)
    return json.loads((R/'evaluation'/f'{name}.json').read_text())
try:
    R.mkdir(parents=True,exist_ok=False); shutil.copy2(R.parent/'evaluate_f1_candidates.py',R/'evaluate_f1_candidates.py'); state('initializing')
    (R/'protocol.json').write_text(json.dumps(dict(thresholds=[.5,.6,.7],selection_seed=0,seeds=[42,2026],criterion='mean full RGB PSNR over fixed 13 seen views',fixed_F1_roi=True),indent=2))
    depth(.5); depth(.7)
    results={}; results['c60_s0']=evaluate('c60_s0',D/'runs/F1_appearance800/params.npz')
    for t in [.5,.7]:
        n=f'c{int(t*100):02d}_s0'; results[n]=evaluate(n,mapping(t,0))
    selected=max(results,key=lambda k:results[k]['mean_psnr']); threshold=int(selected[1:3])/100
    (R/'threshold_selection.json').write_text(json.dumps(dict(selected=selected,threshold=threshold,results=results,criterion='mean PSNR 13 seen views',visual_review_pending=True),indent=2))
    for seed in [42,2026]:
        n=f'c{int(threshold*100):02d}_s{seed}'; results[n]=evaluate(n,mapping(threshold,seed))
    (R/'results.json').write_text(json.dumps(dict(threshold=threshold,selection=selected,results=results,scope='scene-tuned seen views; visual review pending'),indent=2))
    state('complete',threshold=threshold,visual_review_pending=True)
except BaseException as e:
    R.mkdir(parents=True,exist_ok=True); state('failed',error=repr(e),traceback=traceback.format_exc()); raise
