"""Prepare 801 staged seq03 ORB, RAFT and exact Demo confidence inputs."""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, traceback, zipfile
import numpy as np
from PIL import Image

R=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116')
OLD=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
BATCH=OLD.parent; SPLAT=Path('/root/autodl-tmp/SplaTAM'); PY='/root/miniconda3/bin/python'; RAFT='/root/venvs/raft_stereo/bin/python'
ZIP=Path('/root/autodl-tmp/slam_project/data/kitti/data_odometry_color.zip'); S=R/'scripts'; SEQ=R/'sequences/03'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state(stage, **kw):
    p=R/'input_status.json'; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(dict(stage=stage,time=time.strftime('%FT%T%z'),**kw),indent=2)); tmp.replace(p); print(stage,kw,flush=True)
def run(name,cmd,gpu=False,py_path=None):
    if shutil.disk_usage(R).free < 15*2**30: raise RuntimeError('Free disk below 8 GiB; stopped without deleting existing files')
    state(name)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TASK_GPU_GIB='6',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=py_path or f'{S}:{SPLAT}')
    cmd=list(map(str,cmd)); cmd=([cmd[0],str(S/'resource_guard.py')]+cmd[1:]) if gpu else cmd
    with (R/f'{name}.log').open('x') as f: subprocess.run(cmd,cwd=SPLAT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    state('validate_staged_inputs')
    staged=json.loads((R/'staging_receipt.json').read_text());job=json.loads((R/'input_spec.json').read_text());width,height=staged['image_dimensions'];assert staged['frames']==801 and staged['stereo_images']==1602
    for row in json.loads((R/'input_manifest.json').read_text()):
        assert sha(SEQ/f"image_{row['camera']}"/f"{row['source']:06}.png")==row['sha256']
    spec=importlib.util.spec_from_file_location('old_prepare',BATCH/'server_prepare.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state('orb_localization')
    source=(BATCH/'server_prepare.py').read_text();a=source.index('def orb(job):');b=source.index("\nif __name__=='__main__':",a);fn=source[a:b].replace('200','801');exec(compile(fn,'orb_801_adapter.py','exec'),module.__dict__);module.orb(job)
    common=['--raft-root','/root/autodl-tmp/slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo','--checkpoint','/root/raft_stereo_models/raftstereo-middlebury.pth','--sequence-dir',SEQ,'--num-frames','801','--max-depth','40']
    run('raft_left',[RAFT,S/'generate_raft_depth_kitti.py',*common,'--output-dir',R/'raft_left','--preview-indices','0','400','800'],True)
    run('raft_right',[RAFT,S/'generate_raft_reverse_depth_kitti.py',*common,'--output-dir',R/'raft_right'],True)
    state('demo_confidence_060')
    conf=R/'confidence_demo060'
    run('demo_confidence_060',[PY,S/'build_depth_confidence_kitti.py','--left-image-dir',SEQ/'image_2','--right-image-dir',SEQ/'image_3','--depth-left-dir',R/'raft_left','--depth-right-dir',R/'raft_right','--pose-path',R/'poses_estimated_camera2.txt','--calib',SEQ/'calib.txt','--output-root',conf,'--num-frames','801','--seed-threshold','.60','--loss-threshold','.20','--min-depth','.5','--max-depth','40','--lr-cycle-sigma-px','1.0','--photo-sigma','.1','--temporal-sigma-rel','.1','--preview-indices','0','400','800'])
    bundle=R/'bundle';bundle.mkdir();mats={}
    for line in (SEQ/'calib.txt').read_text().splitlines():
        k,v=line.split(':',1);mats[k]=np.fromstring(v,sep=' ').reshape(3,4)
    p2,p3=mats['P2'],mats['P3'];(bundle/'projection_beta.txt').write_text(str(p2[0,3]-p3[0,3]));rows=[]
    for i in range(801):
        n=f'{i:06d}';left=SEQ/'image_2'/f'{n}.png';right=SEQ/'image_3'/f'{n}.png';depth=conf/'depth_confhard'/f'{n}_depth.npy';confidence=conf/'confidence'/f'{n}_confidence.npy';raw=R/'raft_left'/f'{n}_depth.npy'
        d=np.load(depth,allow_pickle=False);c=np.load(confidence,allow_pickle=False);z=np.load(raw,allow_pickle=False);assert d.shape==c.shape==z.shape==(height,width) and np.isfinite(c).all() and np.array_equal(d,np.where(c>=.60,z,0)),i
        rows.append(dict(frame_id=i,source_frame_id=i,rgb=str(left),right=str(right),depth=str(depth),confidence=str(confidence),hashes={k:sha(v) for k,v in [('rgb',left),('right',right),('depth',depth),('confidence',confidence)]}))
    pose=R/'poses_estimated_camera2.txt';runtime=dict(task_id='SEQ03_DEMO_CONF060_EXACT',pose_source='ORB_estimate',pose_file=str(pose),pose_sha256=sha(pose),frame_convention='P2',frame_count=801,K=p2[:,:3].tolist(),H=height,W=width,depth_units='meters',valid_depth_range_m=[.5,40],confidence_threshold=.6,GT_runtime=False,temporal_scope='offline: current and previous/next adjacent frames',rows=rows)
    (bundle/'runtime.json').write_text(json.dumps(runtime,indent=2));(bundle/'confidence_audit.json').write_text(json.dumps(dict(status='PASS',frames=801,seed_threshold=.6,formula='original Demo confidence code',depth_mask='verified exact raw_depth where confidence >= .60',temporal_scope='previous and next, offline',loss_mask='not separately used in mapper; mapper consumes hard depth',mass_proposal_gate=.8,confidence_script_sha256=sha(S/'build_depth_confidence_kitti.py')),indent=2));state('inputs_complete',frames=801,source_frames=[0,800],width=width,height=height,GT_input=False,confidence='Demo .60 exact, offline adjacent frames',confidence_audit=str(bundle/'confidence_audit.json'))
try:main()
except BaseException as e:state('failed',error=repr(e),traceback=traceback.format_exc());raise
