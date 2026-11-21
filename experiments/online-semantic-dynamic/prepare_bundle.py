"""Prepare registered 1000-frame predicted inputs; P1 uses raw stereo, mapping uses G19C.

The distinction is an input-interface correction, not a lowered TSDD threshold.
Existing E0/E1 and the old sparse-P1 observations are retained unchanged.
"""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from experiment import REMOTE,LIMITS
from export_observations import export

S=Path('/root/autodl-tmp/slam_project');PY='/root/venvs/semantic_seg/bin/python'
G21=Path('/root/autodl-tmp/stage3_g21_stereo_20261029_p47149')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(cmd,log):
    begin=time.perf_counter();status='FAILED_PRESERVED'
    try:
        with log.open('x') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=9000,
                           env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1'))
        status='COMPLETE'
    finally:
        with (log.parent/'producer_timings.jsonl').open('a') as f:
            f.write(json.dumps(dict(log=log.name,status=status,seconds=time.perf_counter()-begin,argv=cmd))+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sequence',choices=['08','05','10'],required=True);a=ap.parse_args()
    R=REMOTE/'bundles'/a.sequence;R.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(R).free>=23*2**30
    assert json.loads((REMOTE/'E0_receipt.json').read_text())['status']=='PASS_E0_REPEATABILITY'
    assert (REMOTE/'E1_mapping200_v2/evaluation/results.json').is_file()
    if (REMOTE/'stage2_frozen_sources.json').exists():
        subprocess.run([PY,str(REMOTE/'freeze_stage2.py')],check=True)
    (R/'P1_input_correction_registration.json').write_text(json.dumps(dict(
        status='REGISTERED_BEFORE_DENSE_P1_PREDICTION',reason='Birth-gated depth is not raw motion-evidence depth',
        min_valid_ratio=.35,thresholds_changed=False,mapping_depth_unchanged=True,GT_prediction=False,
        evidence_sha256=sha(REMOTE/'motion_depth_interface_diagnostic.json')),indent=2))
    if a.sequence=='08':
        old=Path('/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181/inputs/runtime/input_manifest.json')
        m=json.loads(old.read_text());inputdir=Path(m['rows'][0]['rgb']).parent.parent
        rawdir=Path('/root/autodl-tmp/stage3_g14_20261027_p30915/data_pipeline/raw_left')
        calib=Path('/root/autodl-tmp/stage3_g19_past_conf_20261029_p47149/calib.txt')
        beta=380.34753
    else:
        # A new isolated, single-window producer, using the already frozen original formula.
        # Do not invoke or weaken the historical two-window 36GiB controller.
        if not (REMOTE/'formal_method_freeze.json').is_file():raise RuntimeError('Freeze methods before formal input generation')
        wid=f'K{a.sequence}_F000000_000999'
        context=json.loads((G21/'windows'/wid/'cpu_context.json').read_text())['window']
        inputdir=Path(context['input_view']);calib=inputdir/'calib.txt';beta=context['projection_beta']
        commands=json.loads((G21/'execution_cpu_registration.json').read_text())['planned_stereo_commands']
        for item in commands:
            if item['window']!=wid:continue
            argv=item['argv'];start=next(i for i,x in enumerate(argv) if x.endswith('/bin/python'))
            cmd=list(argv[start:]);outindex=cmd.index('--output-dir')+1
            cmd[outindex]=str(R/('raw_left' if item['step']=='left' else 'raw_right'))
            run(cmd,R/f"stereo_{item['step']}.log")
        run([PY,str(REMOTE/'confidence_bundle.py'),str(R),str(G21/'windows'/wid/'cpu_context.json')],R/'confidence.log')
        m=json.loads((R/'runtime.json').read_text());rawdir=R/'raw_left'
    assert m['frame_count']==1000 and m['GT_runtime'] is False and m['pose_source']=='ORB_estimate'
    (R/'runtime.json').write_text(json.dumps(m,indent=2));shutil.copy2(calib,R/'calib.txt')
    (R/'projection_beta.txt').write_text(str(beta)+'\n')
    (R/'motion_depth').mkdir();raw_locks=[]
    for row in m['rows']:
        i=row['frame_id'];p=rawdir/f'{i:06}_depth.npy';d=np.load(p,allow_pickle=False)
        assert d.shape==(m['H'],m['W']) and d.dtype==np.float32
        good=np.isfinite(d)&(d>=.5)&(d<=80);target=R/'motion_depth'/p.name
        if ((d==0)|good).all():target.symlink_to(p)
        else:np.save(target,np.where(good,d,0).astype(np.float32),allow_pickle=False)
        raw_locks.append(dict(frame=i,path=str(p),sha256=sha(p),motion_depth_sha256=sha(target)))
    (R/'raw_motion_depth_locks.json').write_text(json.dumps(raw_locks,indent=2))
    image=str(inputdir/'image_2');models=S/'stage2_semantic/models'
    commands=[
      [PY,str(S/'stage2_semantic/scripts/generate_mask2former_semantics_kitti.py'),'--image-dir',image,'--output-dir',str(R/'semantic'),'--model-id',str(models/'mask2former-swin-small-cityscapes-semantic-91281d04'),'--local-files-only','--start','0','--num-frames','1000','--device','cuda','--preview-every','100'],
      [PY,str(S/'stage2_tsdd/scripts/run_mask2former_instance_tracking.py'),'--image-dir',image,'--model-dir',str(models/'mask2former-swin-small-cityscapes-instance-4c0fb9b2-hf457'),'--model-source-id','facebook/mask2former-swin-small-cityscapes-instance','--model-revision','4c0fb9b2','--dense-semantic-dir',str(R/'semantic'),'--output-dir',str(R/'instances'),'--start','0','--end','999','--device','cuda:0'],
      [PY,str(S/'stage2_tsdd/scripts/compute_motion_evidence.py'),'--image-dir',image,'--depth-dir',str(R/'motion_depth'),'--pose-file',m['pose_file'],'--pose-coordinate','p2','--calib-file',str(R/'calib.txt'),'--instance-dir',str(R/'instances'),'--output-dir',str(R/'motion'),'--start','0','--end','999','--device','cuda:0'],
      [PY,str(S/'stage2_tsdd/scripts/run_tsdd.py'),'--image-dir',image,'--instance-dir',str(R/'instances'),'--evidence-dir',str(R/'motion'),'--config',str(S/'stage2_tsdd/configs/tsdd.yaml'),'--output-dir',str(R/'states'),'--start','0','--end','998','--run-status','ENGINEERING_SMOKE']]
    (R/'commands.json').write_text(json.dumps(commands,indent=2))
    for i,cmd in enumerate(commands):
        assert shutil.disk_usage(R).free>8*2**30
        (R/'progress.json').write_text(json.dumps(dict(status='RUNNING',step=i,sequence=a.sequence)))
        run(cmd,R/f'step{i}.log')
    export(R,a.sequence)
    shutil.copy2(R/'semantic/generation_summary.json',R/'semantic_inference_timing.json')
    (R/'prediction_complete.json').write_text(json.dumps(dict(status='COMPLETE_1000_TRUE_PREDICTED_OBSERVATIONS',sequence=a.sequence,
          frames=1000,mapping_depth_source='unchanged G19C',P1_depth_source='raw metric stereo range .5..80',GT_prediction=False,
          observation_manifest_sha256=sha(R/'observations/manifest.json')),indent=2))

if __name__=='__main__':main()
