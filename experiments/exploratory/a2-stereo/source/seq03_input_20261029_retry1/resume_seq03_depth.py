from pathlib import Path
import hashlib, json, os, subprocess, time, datetime, shutil, zipfile
import numpy as np

R = Path('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_input_20261029_retry1')
OLD = Path('/root/autodl-tmp/stage3_g05C_20261025')
PY = '/root/autodl-tmp/stage3_g05_runtime_20261025/venv/bin/python'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def write(name, obj): (R/name).write_text(json.dumps(obj, indent=2), encoding='utf8')

started = time.monotonic()
assert not (R/'result.json').exists()
commands = []
try:
    manifest=json.loads((R/'seq03_input_200_manifest_20261029.json').read_text())
    assert sha(R/'seq03_input_200_20261029.zip')==manifest['archive_sha256']
    for item in manifest['rows']:
        p=R/item['path']; assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
    assert shutil.disk_usage(R).free>8*2**30
    assert not any((R/x).exists() for x in ('raw_left','raw_right','cache','result.json'))
    p2,p3=[np.array([float(v) for v in line.split()[1:]]).reshape(3,4) for line in (R/'input_view/calib.txt').read_text().splitlines() if line.split()[0] in ('P2:','P3:')]
    registration=json.loads((R/'registration.json').read_text())
    assert registration['seed_confidence_threshold']==0.8 and sha(R/'input_view/calib.txt')==registration['calib_sha256']
    prior=json.loads((OLD/'input_registration.json').read_text())
    assert sha(prior['checkpoint'])==prior['checkpoint_sha256'] and sha(OLD/'pipeline/stereo_kitti_headless')==prior['binary_sha256']
    for row in prior['pipeline_sources']: assert sha(row['copied_path'])==row['copied_sha256']
    for row in prior.get('raft_core_file_locks',[]): assert sha(row['path'])==row['sha256']
    pose=np.loadtxt(R/'orb/external_P2_c2w.txt'); assert pose.shape==(200,12) and np.isfinite(pose).all()
    rot=pose.reshape(200,3,4)[:,:,:3]; assert np.max(np.abs(rot.transpose(0,2,1)@rot-np.eye(3)))<2e-5
    assert not any(s in (R/'orb/file_open_audit.log').read_text(errors='replace') for s in ['/poses/','groundtruth.txt','/evaluation_only/'])
    orig_commands=json.loads((R/'commands.json').read_text()); assert orig_commands[-1]['label']=='ORB' and orig_commands[-1]['exit_code']==0
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled')
    def run(label,cmd,limit,cwd=None):
        record={'label':label,'command':cmd,'started_at':datetime.datetime.now().astimezone().isoformat()};commands.append(record);write('resume_commands.json',commands)
        with (R/'logs'/f'{label}.log').open('xb') as f: p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd,timeout=limit)
        record['exit_code']=p.returncode;record['elapsed_seconds']=time.monotonic()-started;write('resume_commands.json',commands)
        if p.returncode: raise RuntimeError(f'{label} failed {p.returncode}')
        print(label,'PASS',flush=True)
    common=['--raft-root',prior['raft_root'],'--checkpoint',prior['checkpoint'],'--sequence-dir',str(R/'input_view'),'--num-frames','200','--valid-iters','32','--min-depth','0.5','--max-depth','80']
    print('WAITING_FOR_GPU_IDLE',flush=True)
    gpu_deadline=time.monotonic()+3600
    while True:
        gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not gpu: break
        if time.monotonic()>=gpu_deadline: raise TimeoutError('GPU still busy after 1 hour')
        time.sleep(15)
    print('GPU_IDLE_STARTING_RAFT',flush=True)
    run('RAFT_LEFT',[PY,str(OLD/'pipeline/left.py'),*common,'--output-dir',str(R/'raw_left'),'--preview-indices','0','99','199'],1200)
    run('RAFT_RIGHT',[PY,str(OLD/'pipeline/right.py'),*common,'--output-dir',str(R/'raw_right')],1200)
    run('CONFIDENCE',[PY,str(OLD/'pipeline/confidence.py'),'--left-image-dir',str(R/'input_view/image_2'),'--right-image-dir',str(R/'input_view/image_3'),'--depth-left-dir',str(R/'raw_left'),'--depth-right-dir',str(R/'raw_right'),'--pose-path',str(R/'orb/external_P2_c2w.txt'),'--calib',str(R/'input_view/calib.txt'),'--output-root',str(R/'cache'),'--num-frames','200','--min-depth','.5','--max-depth','40','--seed-threshold','.80','--loss-threshold','.20'],900)
    rows=[]
    for i in range(200):
        paths={'rgb':R/f'input_view/image_2/{i:06d}.png','right':R/f'input_view/image_3/{i:06d}.png','depth':R/f'cache/depth_confhard/{i:06d}_depth.npy','confidence':R/f'cache/confidence/{i:06d}_confidence.npy'}
        d=np.load(paths['depth']);c=np.load(paths['confidence'])
        assert d.dtype==c.dtype==np.float32 and d.shape==c.shape==(375,1242) and np.isfinite(d).all() and np.isfinite(c).all()
        assert np.all((d==0)|((d>=.5)&(d<=40))) and np.all(c[d>0]>=.8) and np.all((c>=0)&(c<=1))
        rows.append({'frame_id':i,'source_frame_id':i,**{k:str(v) for k,v in paths.items()},'hashes':{k:sha(v) for k,v in paths.items()},'valid_depth_pixels':int((d>0).sum())})
    m={'task_id':'A2_SEQ03_INPUT','pose_source':'ORB_estimate','pose_file':str(R/'orb/external_P2_c2w.txt'),'pose_sha256':sha(R/'orb/external_P2_c2w.txt'),'frame_convention':'P2 c2w, relative to window first P2, metres','frame_count':200,'K':p2[:,:3].tolist(),'H':375,'W':1242,'depth_units':'metres','confidence_threshold':0.8,'GT_runtime':False,'temporal_scope':registration['temporal_scope'],'rows':rows}
    write('input_manifest.json',m)
    write('result.json',{'status':'PASS_INPUT_PREPARATION','frames':200,'pose_sha256':m['pose_sha256'],'manifest_sha256':sha(R/'input_manifest.json'),'elapsed_seconds':time.monotonic()-started,'independent_test_status':'not established'})
except BaseException as e:
    write('result.json',{'status':'FAILED_INPUT_PREPARATION_PRESERVED','error':repr(e),'commands':commands,'elapsed_seconds':time.monotonic()-started})
    raise
