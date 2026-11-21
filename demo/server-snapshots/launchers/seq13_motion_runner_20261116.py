"""Isolated seq13 flow, instances and Pi3MOS inference from the same 260 RGB frames."""
from pathlib import Path
import json, os, shutil, subprocess, time, traceback
R=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115')
B=R/'motion_demo_20261116'; A=B/'annotations_20261107'; S=A/'scripts'
OLD=Path('/root/autodl-tmp/seq07_000850_001049_tuning_20261115/c60_s42_semantic_motion/annotations_20261107/scripts')
PY='/root/miniconda3/bin/python'; SEM='/root/venvs/semantic_seg/bin/python'; IMG=B/'sequences/13/image_2'
def status(stage,**kw):
    (B/'status.json').write_text(json.dumps(dict(stage=stage,time=time.strftime('%FT%T%z'),**kw),indent=2));print(stage,kw,flush=True)
def run(stage,cmd,env):
    if shutil.disk_usage(B).free<8*2**30:raise RuntimeError('Free space below 8 GiB')
    status(stage)
    with (B/(stage+'.log')).open('x') as f:subprocess.run(list(map(str,cmd)),cwd='/root/autodl-tmp/SplaTAM',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    B.mkdir(exist_ok=False);A.mkdir();shutil.copytree(OLD,S)
    (B/'sequences').mkdir();(B/'sequences/13').symlink_to(R/'sequences/13',target_is_directory=True)
    (B/'raft_left').symlink_to(R/'raft_left',target_is_directory=True)
    (A/'semantic').symlink_to(R/'semantic_demo_20261116/predictions',target_is_directory=True)
    flow=(S/'generate_flow.py').read_text();pairs=[('sequences/07','sequences/13'),('range(1,200)','range(1,260)'),('(0,6,0,6)','(0,7,0,0)'),(':370,:1226',':376,:1241'),('(370,1226,2)','(376,1241,2)')]
    for a,b in pairs:
        assert a in flow,a;flow=flow.replace(a,b)
    (S/'generate_flow.py').write_text(flow)
    mot=(S/'run_motion.py').read_text();pairs=[('sequences/07','sequences/13'),('range(200)','range(260)'),('t==199','t==259'),('(1226,370)','(1241,376)'),('size=(370,1226)','size=(376,1241)'),('[280,910]','[280,924]'),('image_size=[1226,370]','image_size=[1241,376]'),('frames=200,inferred_frames=193','frames=260,inferred_frames=253'),('input_size=[910,280]','input_size=[924,280]')]
    for a,b in pairs:
        assert a in mot,a;mot=mot.replace(a,b)
    (S/'run_motion.py').write_text(mot)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TASK_GPU_GIB='8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=':'.join(map(str,[S,Path('/root/autodl-tmp/SplaTAM'),Path('/root/autodl-tmp/stage2_new_model_pi3mos/vendor'),Path('/root/autodl-tmp/stage2_new_model_pi3mos/Pi3MOS-SLAM')])))
    assert len(list((A/'semantic/labels').glob('*.npy')))==260
    run('flow',[PY,S/'generate_flow.py'],env)
    run('instances',[SEM,S/'run_instances.py','--image-dir',IMG,'--model-dir','/root/autodl-tmp/slam_project/stage2_semantic/models/mask2former-swin-small-cityscapes-instance-4c0fb9b2-hf457','--model-source-id','facebook/mask2former-swin-small-cityscapes-instance','--model-revision','4c0fb9b22484a4dfbac0e27ee74d3f155f29ae93+hf457-key-remap','--dense-semantic-dir',A/'semantic','--flow-dir',A/'flow','--output-dir',A/'instances','--device','cuda:0','--start','0','--end','259','--score-threshold','.30','--min-area','32','--dedupe-mask-iou','.30','--dedupe-containment','.70','--cross-category-mask-iou','.80','--cross-category-containment','.95','--min-dense-agreement','.10','--track-iou','.08','--track-max-age','2','--flow-warp-dilation','3','--preview-every','25','--warmup-frames','20','--seed','42'],env)
    run('verify_instances',[SEM,S/'verify_instances.py','--image-dir',IMG,'--instance-dir',A/'instances','--dense-semantic-dir',A/'semantic','--start','0','--end','259','--max-same-category-mask-iou','.30','--max-same-category-containment','.70','--max-cross-category-mask-iou','.80','--max-cross-category-containment','.95','--min-dense-agreement','.10','--report',A/'instance_verification.json'],env)
    run('motion',[PY,S/'run_motion.py'],env)
    status('complete',frames=260,source_frames=[780,1039])
try: main()
except BaseException as e:
    if B.exists():status('failed',error=repr(e),traceback=traceback.format_exc())
    raise
