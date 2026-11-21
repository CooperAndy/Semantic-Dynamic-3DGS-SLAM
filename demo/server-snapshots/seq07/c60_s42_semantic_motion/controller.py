"""Build seq07 semantic and Pi3MOS assets for the selected F1 map in isolation."""
from pathlib import Path
import csv, hashlib, importlib.util, json, os, shutil, subprocess, time, traceback
import numpy as np
P=Path('/root/autodl-tmp/seq07_000850_001049_tuning_20261115'); B=P/'c60_s42_semantic_motion'; OLD=Path('/root/autodl-tmp/demo_batch09_17_20261107'); SPLAT=Path('/root/autodl-tmp/SplaTAM'); PY='/root/miniconda3/bin/python'; SEM='/root/venvs/semantic_seg/bin/python'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state(s,**kw):
    x=B/'status.json';tmp=x.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=s,time=time.strftime('%FT%T%z'),**kw),indent=2));tmp.replace(x);print(s,kw,flush=True)
def run(name,args,gpu=False,limit=6):
    if shutil.disk_usage(B).free<8*2**30:raise RuntimeError('Less than 8 GiB free; stopped with outputs preserved')
    state(name); S=B/'scripts'; A=B/'annotations_20261107/scripts'; paths=[S,A,SPLAT,Path('/root/autodl-tmp/stage2_new_model_pi3mos/vendor'),Path('/root/autodl-tmp/stage2_new_model_pi3mos/Pi3MOS-SLAM')]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TASK_GPU_GIB=str(limit),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=':'.join(map(str,paths)))
    cmd=list(map(str,args));cmd=[cmd[0],str(S/'resource_guard.py'),*cmd[1:]] if gpu else cmd
    with (B/f'{name}.log').open('x') as f:subprocess.run(cmd,cwd=SPLAT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    state('prepare'); src=P/'c60_s42/F1/params.npz'; map0=P/'c60_s42/map/params.npz'; expected='4ae010a3d6f08ab50e5ca544021de40bdce2f5c19a75fb80de51b2a5929ed6d8';assert sha(src)==expected
    with np.load(src,allow_pickle=False) as z:count=len(z['means3D']);assert z['cam_trans'].shape[-1]==200
    job=dict(sequence='07',start=850,end=1049,count=200,width=1226,height=370,job='seq07_850_1049_c60_s42',remote_root=str(B),confidence_seed=.6,seed=42)
    (B/'input_spec.json').write_text(json.dumps(job,indent=2)); (B/'sequences').mkdir();(B/'sequences/07').symlink_to(P/'sequences/07',target_is_directory=True)
    for name in ('raft_left','poses_estimated_camera2.txt','reliable_roi','deployment'):(B/name).symlink_to(P/name,target_is_directory=(P/name).is_dir())
    records=json.loads((P/'input_manifest.json').read_text()); lookup={(a['local'],a['camera']):a['sha256'] for a in records};times=(P/'sequences/07/times.txt').read_text().splitlines()
    with (B/'input_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['local_index','source_frame','timestamp','image_2_sha256','image_3_sha256']);w.writerows((i,850+i,times[i],lookup[i,2],lookup[i,3]) for i in range(200))
    sp=importlib.util.spec_from_file_location('prepare',OLD/'server_prepare.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.prepare(job)
    for ver,target in [('F200_orb_raft_conf060',map0),('F1_appearance800',src)]:
        folder=B/'runs'/ver;folder.mkdir(parents=True);(folder/'params.npz').symlink_to(target)
    (B/'field_verification.json').write_text(json.dumps(dict(F1_sha256=expected,F200_sha256=sha(map0),gaussian_count=count,scope='selected seq07 c60 seed42'),indent=2))
    S=B/'scripts'; A=B/'annotations_20261107'; Q=A/'scripts'; img=B/'sequences/07/image_2'
    run('map_export',[PY,S/'export_assets.py'])
    run('semantic',[SEM,Q/'generate_semantic.py','--image-dir',img,'--model-id','/root/autodl-tmp/slam_project/stage2_semantic/models/mask2former-swin-small-cityscapes-semantic-91281d04','--local-files-only','--device','cuda','--start','0','--num-frames','200','--preview-every','25','--output-dir',A/'semantic'],True)
    run('flow',[PY,Q/'generate_flow.py'],True)
    run('instances',[SEM,Q/'run_instances.py','--image-dir',img,'--model-dir','/root/autodl-tmp/slam_project/stage2_semantic/models/mask2former-swin-small-cityscapes-instance-4c0fb9b2-hf457','--model-source-id','facebook/mask2former-swin-small-cityscapes-instance','--model-revision','4c0fb9b22484a4dfbac0e27ee74d3f155f29ae93+hf457-key-remap','--dense-semantic-dir',A/'semantic','--flow-dir',A/'flow','--output-dir',A/'instances','--device','cuda:0','--start','0','--end','199','--score-threshold','.30','--min-area','32','--dedupe-mask-iou','.30','--dedupe-containment','.70','--cross-category-mask-iou','.80','--cross-category-containment','.95','--min-dense-agreement','.10','--track-iou','.08','--track-max-age','2','--flow-warp-dilation','3','--preview-every','25','--warmup-frames','20','--seed','42'],True)
    run('verify_instances',[SEM,Q/'verify_instances.py','--image-dir',img,'--instance-dir',A/'instances','--dense-semantic-dir',A/'semantic','--start','0','--end','199','--max-same-category-mask-iou','.30','--max-same-category-containment','.70','--max-cross-category-mask-iou','.80','--max-cross-category-containment','.95','--min-dense-agreement','.10','--report',A/'instance_verification.json'])
    run('motion',[PY,Q/'run_motion.py'],True,7.5)
    run('semantic_fusion',[PY,Q/'fuse_semantic.py','--params',src,'--semantic-dir',A/'semantic','--rgb-dir',img,'--output-dir',A/'fusion','--splatam-root',SPLAT,'--expected-gaussians',str(count),'--expected-params-sha256',expected,'--expected-width','1226','--expected-height','370','--deep-check'],True)
    run('annotation_export',[PY,Q/'export_annotations.py'])
    state('complete',gaussians=count,map_sha256=expected,semantic_dir=str(A/'delivery/KITTI07_F1_Semantic'),motion_dir=str(A/'delivery/KITTI07_MotionDemo'),source_frames=[850,1049],note='frame export uses local numbering pending demo adaptation')
try:main()
except BaseException as e:state('failed',error=repr(e),traceback=traceback.format_exc());raise
