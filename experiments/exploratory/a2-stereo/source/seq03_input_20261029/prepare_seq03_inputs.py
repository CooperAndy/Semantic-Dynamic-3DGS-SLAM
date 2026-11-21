from pathlib import Path
import hashlib, json, os, subprocess, time, datetime, shutil, zipfile
import numpy as np

R = Path('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_input_20261029')
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
    manifest = json.loads((R/'seq03_input_200_manifest_20261029.json').read_text())
    archive = R/'seq03_input_200_20261029.zip'
    assert sha(archive) == manifest['archive_sha256']
    assert not (R/'input_view').exists()
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {x['path'] for x in manifest['rows']}
        assert z.testzip() is None
        z.extractall(R)
    for item in manifest['rows']:
        p = R/item['path']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], item['path']
    assert shutil.disk_usage(R).free > 8*2**30
    p2, p3 = [np.array([float(v) for v in line.split()[1:]]).reshape(3,4) for line in (R/'input_view/calib.txt').read_text().splitlines() if line.split()[0] in ('P2:', 'P3:')]
    baseline = abs(-p3[0,3]/p3[0,0] + p2[0,3]/p2[0,0])
    assert 0.1 < baseline < 1 and p2[0,0] > 0
    prior = json.loads((OLD/'input_registration.json').read_text())
    assert sha(prior['checkpoint']) == prior['checkpoint_sha256']
    assert sha(OLD/'pipeline/stereo_kitti_headless') == prior['binary_sha256']
    for row in prior['pipeline_sources']:
        assert sha(row['copied_path']) == row['copied_sha256'], row['copied_path']
    for row in prior.get('raft_core_file_locks', []):
        assert sha(row['path']) == row['sha256'], row['path']
    for folder in ('orb','logs'):(R/folder).mkdir(exist_ok=False)
    settings=(OLD/'orb/settings.yaml').read_text()
    replace={'Camera1.fx':p2[0,0],'Camera1.fy':p2[1,1],'Camera1.cx':p2[0,2],'Camera1.cy':p2[1,2], 'Camera.width':1242,'Camera.height':375,'Stereo.b':baseline}
    settings='\n'.join((line.split(':',1)[0]+': '+str(replace[line.split(':',1)[0]]) if ':' in line and line.split(':',1)[0] in replace else line) for line in settings.splitlines())+'\n'
    (R/'orb/settings.yaml').write_text(settings)
    registration={'sequence':'KITTI03','frames':[0,199],'seed_confidence_threshold':0.8,'loss_threshold':0.2,'depth_raw_range_m':[0.5,80], 'confidence_valid_range_m':[0.5,40], 'K':p2[:,:3].tolist(),'baseline_m':baseline,'original_manifest_sha256':sha(R/'seq03_input_200_manifest_20261029.json'),'calib_sha256':sha(R/'input_view/calib.txt'),'checkpoint_sha256':sha(prior['checkpoint']),'binary_sha256':sha(OLD/'pipeline/stereo_kitti_headless'),'source_registration_sha256':sha(OLD/'input_registration.json'),'temporal_scope':'Same G05 as prior A2: adjacent t-1/t+1 including mapper held-out frames; ORB window-optimized; not strictly online or strict held-out input isolation','lidar_used':False}
    write('registration.json',registration)
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled')
    orb_env=env.copy();orb_env['LD_LIBRARY_PATH']=':'.join(prior['library_paths'])
    def run(label,cmd,limit,cwd=None):
        record={'label':label,'command':cmd,'started_at':datetime.datetime.now().astimezone().isoformat()}; commands.append(record);write('commands.json',commands)
        with (R/'logs'/f'{label}.log').open('xb') as f:
            p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=orb_env if label=='ORB' else env,cwd=cwd,timeout=limit)
        record['exit_code']=p.returncode;record['elapsed_seconds']=time.monotonic()-started;write('commands.json',commands)
        if p.returncode: raise RuntimeError(f'{label} failed {p.returncode}')
        print(label,'PASS',flush=True)
    run('ORB',['strace','-f','-e','trace=openat','-o',str(R/'orb/file_open_audit.log'),str(OLD/'pipeline/stereo_kitti_headless'),prior['vocabulary'],str(R/'orb/settings.yaml'),str(R/'input_view'),'O0','-','-','03',str(R/'orb/frame_features.csv')],900,R/'orb')
    opens=(R/'orb/file_open_audit.log').read_text(errors='replace')
    assert not any(s in opens for s in ['/poses/','groundtruth.txt','/evaluation_only/'])
    raw=np.loadtxt(R/'orb/CameraTrajectory.txt',dtype=np.float64);assert raw.shape==(200,12) and np.isfinite(raw).all(),raw.shape
    T=np.tile(np.eye(4),(200,1,1));T[:,:3]=raw.reshape(200,3,4)
    assert np.max(np.abs(T[:,:3,:3].transpose(0,2,1)@T[:,:3,:3]-np.eye(3)))<2e-5
    normalized=np.linalg.inv(T[0])@T
    np.savetxt(R/'orb/external_P2_c2w.txt',normalized[:,:3].reshape(200,12),fmt='%.9f')
    common=['--raft-root',prior['raft_root'],'--checkpoint',prior['checkpoint'],'--sequence-dir',str(R/'input_view'),'--num-frames','200','--valid-iters','32','--min-depth','0.5','--max-depth','80']
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
