"""Independent seq13 display tuning; all products stay on the server."""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, traceback, zipfile
import numpy as np
from PIL import Image

R=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115')
OLD=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
BATCH=OLD.parent; SPLAT=Path('/root/autodl-tmp/SplaTAM'); PY='/root/miniconda3/bin/python'; RAFT='/root/venvs/raft_stereo/bin/python'
ZIP=Path('/root/autodl-tmp/slam_project/data/kitti/data_odometry_color.zip'); S=R/'scripts'; SEQ=R/'sequences/13'
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
            for i in range(260):
                src=f'dataset/sequences/13/image_{cam}/{780+i:06d}.png'; dest=folder/f'{i:06d}.png'
                with z.open(src) as a,dest.open('wb') as b:shutil.copyfileobj(a,b)
                with Image.open(dest) as im:
                    if i==0 and cam==2: width,height=im.size
                    assert im.size==(width,height);im.verify()
                manifest.append(dict(local=i,source=780+i,camera=cam,sha256=sha(dest)))
        for name in ('calib.txt','times.txt'):
            raw=z.read(f'dataset/sequences/13/{name}').decode()
            if name=='times.txt':raw='\n'.join(raw.splitlines()[780:1040])+'\n'
            (SEQ/name).write_text(raw)
    (R/'input_manifest.json').write_text(json.dumps(manifest,indent=2)); assert len(manifest)==520
    job=dict(sequence='13',start=780,end=1039,count=260,width=width,height=height,job='seq13_780_1039_O3F',remote_root=str(R))
    (R/'input_spec.json').write_text(json.dumps(job,indent=2));shutil.copytree(OLD/'vendor',R/'vendor')
    (R/'protocol.json').write_text(json.dumps(dict(job=job,thresholds=[.5,.6,.7],threshold_seed=0,extra_seeds=[42,2026],f1_steps=800,selection='mean full image PSNR across 13 fixed seen views, subject to visual review',GT_input=False),indent=2))
    for p in (OLD/'scripts').glob('*.py'):
        text=p.read_text().replace(str(OLD),str(R)).replace('sequences/09','sequences/13')
        text=re.sub(r'\b1226\b',str(width),text);text=re.sub(r'\b370\b',str(height),text)
        (S/p.name).write_text(text)
    spec=importlib.util.spec_from_file_location('old_prepare',BATCH/'server_prepare.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state('orb_localization')
    source=(BATCH/'server_prepare.py').read_text();a=source.index('def orb(job):');b=source.index("\nif __name__=='__main__':",a);fn=source[a:b].replace('200','260');exec(compile(fn,'orb_260_adapter.py','exec'),module.__dict__);module.orb(job)
    common=['--raft-root','/root/autodl-tmp/slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo','--checkpoint','/root/raft_stereo_models/raftstereo-middlebury.pth','--sequence-dir',SEQ,'--num-frames','260','--max-depth','40']
    run('raft_left',[RAFT,S/'generate_raft_depth_kitti.py',*common,'--output-dir',R/'raft_left','--preview-indices','0','129','259'],True)
    run('raft_right',[RAFT,S/'generate_raft_reverse_depth_kitti.py',*common,'--output-dir',R/'raft_right'],True)
    bundle=R/'bundle';bundle.mkdir();(bundle/'raw_left').symlink_to(R/'raft_left',target_is_directory=True);(bundle/'raw_right').symlink_to(R/'raft_right',target_is_directory=True)
    mats={}
    for line in (SEQ/'calib.txt').read_text().splitlines():
        k,v=line.split(':',1);mats[k]=np.fromstring(v,sep=' ').reshape(3,4)
    p2,p3=mats['P2'],mats['P3'];K=p2[:,:3];beta=p2[0,3]-p3[0,3];(bundle/'projection_beta.txt').write_text(str(beta));pose=R/'poses_estimated_camera2.txt'
    context={'window':{'primary_pose_path':str(pose),'primary_pose_sha256':sha(pose),'K':K.tolist(),'projection_beta':float(beta),'shape':[height,width],'input_view':str(SEQ),'source_start':780}}
    (bundle/'context.json').write_text(json.dumps(context,indent=2));code=Path('/root/autodl-tmp/s23_effects_20261105_p16247/confidence_bundle.py').read_text().replace('1000','260');(R/'confidence_260.py').write_text(code)
    run('g19c_confidence',['/root/venvs/semantic_seg/bin/python',R/'confidence_260.py',bundle,bundle/'context.json'])
    runtime=json.loads((bundle/'runtime.json').read_text());assert runtime['frame_count']==260 and [x['source_frame_id'] for x in runtime['rows']]==list(range(780,1040));state('inputs_complete',frames=260,source_frames=[780,1039],width=width,height=height,GT_input=False,confidence='G19C current and previous frame')
try:main()
except BaseException as e:state('failed',error=repr(e),traceback=traceback.format_exc());raise
