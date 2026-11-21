"""Registered E2/E3 execution. Fails closed on unprepared inputs or unmet gates.

Each input bundle contains runtime.json, observations/manifest.json and calib.txt.
No GT, missing-frame substitution, implicit resumption, or parameter search.
"""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from experiment import REMOTE,PARENT,PYTHON,LIMITS,design


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare_cell(cell,bundle,R,candidate=None):
    m=json.loads((bundle/'runtime.json').read_text())
    assert m['frame_count']==1000 and m['GT_runtime'] is False and m['pose_source']=='ORB_estimate'
    assert [r['frame_id'] for r in m['rows']]==list(range(1000))
    assert [r['source_frame_id'] for r in m['rows']]==list(range(cell['source_start'],cell['source_start']+1000))
    obs=json.loads((bundle/'observations/manifest.json').read_text())
    assert obs['sequence']==cell['sequence'] and obs['camera']=='P2' and obs['GT_runtime'] is False
    R.mkdir(exist_ok=False)
    candidate=Path(candidate) if candidate is not None else REMOTE/'candidate_v1'
    for section in ['source','scripts']:shutil.copytree(candidate/section,R/section)
    for section in ['runs','receipts','inputs/runtime']:(R/section).mkdir(parents=True,exist_ok=True)
    original='/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181'
    p=R/'scripts/common.py';p.write_text(p.read_text().replace(original,str(R)))
    shutil.copy2(bundle/'runtime.json',R/'inputs/runtime/input_manifest.json')
    K=np.asarray(m['K']);yaml=R/'source/configs/data/g20_manifest.yaml'
    yaml.write_text('dataset_name: kitti\ninput_manifest: '+str(R/'inputs/runtime/input_manifest.json')+'\ncamera_params:\n'+
                    '\n'.join(f'  {k}: {v}' for k,v in dict(image_height=m['H'],image_width=m['W'],fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],png_depth_scale=1.0,crop_edge=0).items())+'\n')
    base={};exec((REMOTE/'technical25/source/configs/kitti/TECH_TMD25.py').read_text(),base)
    config=json.loads(json.dumps(base['config']))
    groups=['B','T','M','TM','D','TD','MD','TMD'] if cell['stage']=='E2' else ['B','TM','D','TMD']
    configs={}
    from experiment import FLAGS
    for group in groups:
        cfg=json.loads(json.dumps(config));cfg.update(workdir=str(R/'runs'),run_name=group,variant_id=group,task_id='S23_'+cell['stage'],seed=cell['seed'],feedback_policy='MASS',donor_variant='B',checkpoint_interval=199,eval_only_frames=list(range(19,1000,20)))
        cfg['data'].update(num_frames=1000,desired_image_height=m['H'],desired_image_width=m['W'],sequence=cell['sequence'],gradslam_data_cfg=str(yaml))
        cfg['tracking']['camera_policy']='external_init_refine';cfg['factorial'].update(history_budget=3,seed=cell['seed'])
        cfg['semantic']=dict(zip(['T','M','D'],FLAGS[group]));cfg['semantic'].update(labels=True,shuffle=False,manifest=str(bundle/'observations/manifest.json'))
        cfg['mapping']['residual_fb']=float(bundle.joinpath('projection_beta.txt').read_text())
        p=R/f'source/configs/kitti/{group}.py';p.write_text('config = '+repr(cfg)+'\n');configs[group]=digest(p)
    registration=dict(variants=groups,cell=cell,config_sha256=configs,limits=LIMITS,quality_adoption=False,candidate=candidate.name)
    numerical=candidate/'numerical_accumulation_revision.json'
    if numerical.exists():registration['native_numerical_revision']=json.loads(numerical.read_text())
    (R/'registration.json').write_text(json.dumps(registration,indent=2))
    locks=[dict(path=p.relative_to(R).as_posix(),sha256=digest(p)) for p in R.rglob('*.py')]
    (R/'source_locks.json').write_text(json.dumps(locks,indent=2));return groups


def check_prerequisites(stage,inputs):
    missing=[]
    e0=REMOTE/'E0_receipt.json'
    if not e0.exists() or json.loads(e0.read_text())['status']!='PASS_E0_REPEATABILITY':missing.append('E0_REPEATABILITY')
    e1=REMOTE/'E1_mapping200_v2/evaluation/results.json'
    if not e1.exists() or json.loads(e1.read_text())['status']!='COMPLETE_E1_MAPPING_60_VIEWS_OFFICIAL_SPARSE_GT':missing.append('E1_FIXED_POSE_MAPPING_EVALUATION')
    if not (REMOTE/'evaluation_only/dynamic_evaluation.json').exists():missing.append('E1_AUTOMATIC_MOTION_DIAGNOSTIC')
    if stage=='E3':
        receipt=REMOTE/'E2_matrix_receipt.json'
        if not receipt.exists() or json.loads(receipt.read_text())['status']!='COMPLETE_REGISTERED_MATRIX':missing.append('E2_COMPLETE_AND_FREEZE')
    sequences=['08'] if stage=='E2' else ['05','10']
    for sequence in sequences:
        for name in ['runtime.json','observations/manifest.json','calib.txt','projection_beta.txt']:
            if not (inputs/sequence/name).is_file():missing.append(f'K{sequence}:{name}')
    if shutil.disk_usage(REMOTE).free<LIMITS['startup_free_GiB']*2**30:missing.append('STARTUP_23GIB')
    return missing


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['E2','E3'],required=True);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--check-only',action='store_true');ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    missing=check_prerequisites(a.stage,a.inputs)
    readiness=dict(stage=a.stage,status='READY' if not missing else 'WAITING_PREREQUISITES_NOT_EXECUTED',missing=missing,quality_adoption=False)
    (REMOTE/f'{a.stage}_readiness.json').write_text(json.dumps(readiness,indent=2));print(json.dumps(readiness),flush=True)
    if a.check_only or not a.execute:return
    if (REMOTE/'STOP_AT_SAFE_BOUNDARY').exists():raise RuntimeError('Design-only stop marker is present; no experiment started')
    if missing:raise RuntimeError('Cannot execute incomplete matrix: '+','.join(missing))
    D=design();cells=D['development'] if a.stage=='E2' else D['formal']
    windows=[]
    for c in cells:
        key=(c['sequence'],c['seed'])
        if key not in windows:windows.append(key)
    if a.stage=='E3':
        freeze=REMOTE/'formal_method_freeze.json'
        if freeze.exists():raise FileExistsError('Preserve prior formal freeze')
        freeze.write_text(json.dumps(dict(status='FROZEN_BEFORE_FORMAL_PREDICTION_AND_NEW_WINDOW_GT',design=D,source_sha256={p.name:digest(p) for p in (REMOTE/'candidate_v1/scripts').glob('*.py')},E2_receipt_sha256=digest(REMOTE/'E2_matrix_receipt.json'),new_window_GT_consulted=False),indent=2))
    env=dict(os.environ,PYTHONPATH=str(PARENT/'raster_sm86'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled')
    completed=[]
    for sequence,seed in windows:
        cell=next(c for c in cells if c['sequence']==sequence and c['seed']==seed)
        R=REMOTE/f'{a.stage}_K{sequence}_S{seed}';groups=prepare_cell(cell,a.inputs/sequence,R)
        for group in groups:
            if shutil.disk_usage(R).free<LIMITS['startup_free_GiB']*2**30:raise RuntimeError('Insufficient registered capacity; do not delete parent science')
            if group!='B':
                assert (R/'receipts/donor.json').is_file(),'Baseline donor must be sealed'
            with (R/f'{group}.log').open('x') as log:
                subprocess.run([PYTHON,str(R/'scripts/worker.py'),group],cwd=R/'scripts',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=LIMITS['run_seconds'],check=True)
            receipt=json.loads((R/'runs'/group/'receipt.json').read_text())
            assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION' and receipt['frames']==1000 and receipt['mapping_backward']==57000
            if group=='B':
                donor=R/'runs/B';files=[donor/'target_schedule.npz']+sorted((donor/'addition_masks').glob('*.npz'))
                assert len(files)==950
                (R/'receipts/donor.json').write_text(json.dumps(dict(status='PASS_SEALED_DONOR_INPUTS',files=[dict(path=p.relative_to(R).as_posix(),sha256=digest(p)) for p in files]),indent=2))
            completed.append(dict(sequence=sequence,seed=seed,group=group,receipt_sha256=digest(R/'runs'/group/'receipt.json')))
            (REMOTE/f'{a.stage}_progress.json').write_text(json.dumps(dict(status='RUNNING',completed=completed,quality_adoption=False),indent=2))
    assert len(completed)==len(cells)
    (REMOTE/f'{a.stage}_matrix_receipt.json').write_text(json.dumps(dict(status='COMPLETE_REGISTERED_MATRIX',completed=completed,quality_adoption=False,evaluation_status='PENDING_FULL_JOINT_EVALUATION'),indent=2))

if __name__=='__main__':main()
