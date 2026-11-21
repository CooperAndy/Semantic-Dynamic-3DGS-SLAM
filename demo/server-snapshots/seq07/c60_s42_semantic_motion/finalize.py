"""Wait for selected-map predictions; prepare validated source-frame object assets."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile,time,traceback
P=Path('/root/autodl-tmp/seq07_000850_001049_tuning_20261115');B=P/'c60_s42_semantic_motion'; T=B/'object_tools'; O=B/'delivery_seq07'; SC=O/'KITTI07_C60S42'; A=B/'annotations_20261107'; PY='/root/miniconda3/bin/python'; SCENE='kitti07_conf060_seed42'; H='4ae010a3d6f08ab50e5ca544021de40bdce2f5c19a75fb80de51b2a5929ed6d8'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(stage,**kw):save(B/'finalize_status.json',dict(stage=stage,time=time.strftime('%FT%T%z'),**kw));print(stage,kw,flush=True)
def run(name,cmd):
    status(name)
    if shutil.disk_usage(B).free<8*2**30:raise RuntimeError('Disk below 8 GiB; stopped without deleting files')
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (B/f'finalize_{name}.log').open('x') as f:subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    status('waiting_for_predictions')
    for _ in range(720):
        st=read(B/'status.json')
        if st['stage']=='failed':raise RuntimeError('Prediction pipeline failed; inspect status.json and logs')
        if st['stage']=='complete':break
        time.sleep(30)
    else:raise RuntimeError('Prediction pipeline did not finish within six hours')
    assert sha(P/'c60_s42/F1/params.npz')==H
    status('prepare_delivery'); O.mkdir(exist_ok=False);SC.mkdir()
    for name in ('KITTI07_F200','KITTI07_F1'):shutil.copytree(B/'viewer_assets'/name,SC/name)
    for name in ('KITTI07_F1_Semantic','KITTI07_MotionDemo'):shutil.copytree(A/'delivery'/name,SC/name)
    for name in ('F200','F1'):
        folder=SC/f'KITTI07_{name}'; pp=folder/f'KITTI07_{name}_camera_poses.json'; poses=read(pp);assert len(poses)==200
        for i,p in enumerate(poses):p.update(source_frame=850+i,img_name=f'{850+i:06d}.png')
        save(pp,poses);mp=folder/f'KITTI07_{name}_manifest.json';man=read(mp);man['source_frames']=[850,1049];man['files']={x.name:dict(bytes=x.stat().st_size,sha256=sha(x)) for x in folder.iterdir() if x.is_file() and x!=mp};save(mp,man)
    sem=SC/'KITTI07_F1_Semantic/semantic_manifest.json';sm=read(sem);assert sm['source_params_sha256']==H;sm['source_frames']=[850,1049];save(sem,sm)
    motion=SC/'KITTI07_MotionDemo';index=read(motion/'index.json');assert len(index['frames'])==200
    for entry in index['frames']:
        local=entry['frame'];source=850+local;old=motion/f'frames/{local:06d}.json';snap=read(old);snap.update(reference_frame=source,scene_id=SCENE,source_frames=[850,1049],reference_image=f'scene-data/KITTI07_C60S42/KITTI07_MotionDemo/reference/{source:06d}.jpg');snap['window']=[850+x for x in snap['window']]
        for obj in snap['objects']:
            for ob in obj.get('observations',[]):ob['frame']+=850
        save(motion/f'frames/{source:06d}.json',snap);old.unlink();(motion/f'reference/{local:06d}.jpg').rename(motion/f'reference/{source:06d}.jpg')
        entry.update(frame=source,snapshot=f'scene-data/KITTI07_C60S42/KITTI07_MotionDemo/frames/{source:06d}.json')
    index['default_frame']+=850;save(motion/'index.json',index);shutil.copy2(motion/f"frames/{index['default_frame']:06d}.json",motion/'snapshot.json')
    T.mkdir();
    with tarfile.open(B/'object_tools.tar.gz') as tar:tar.extractall(T,filter='data')
    ins=O/'object_inputs';obs=O/'observations';objects=O/'object_results'
    run('prepare_object_inputs',[PY,T/'reuse/prepare_inputs.py','--instances',A/'instances','--depth',P/'raft_left','--motion',A/'motion','--params',P/'c60_s42/F1/params.npz','--sequence','07','--start','850','--end','1049','--local-start','0','--output',ins])
    run('aggregate_observations',[PY,T/'aggregate_vehicle_states.py','run','--frames',motion/'frames','--output',obs,'--fixed-config','short4_z0.5-40_far20'])
    profile=read(T/'reuse/seq09_c50s42_profile.json');profile.update(name='seq07_c60s42_object_regions',workspace_root='/',sequence='07',scene_id=SCENE,frame_range=[850,1049],enable_static_hold=False,scope='New map and new observations; static hold disabled pending review; no accuracy ground truth')
    profile['paths'].update(semantic_dir=str(SC/'KITTI07_F1_Semantic'),semantic_ply='KITTI07_F1_Semantic.ply',raw_frames=str(motion/'frames'),camera_poses=str(SC/'KITTI07_F1/KITTI07_F1_camera_poses.json'),instance_inputs=str(ins),observation_run=str(obs))
    profile_path=O/'object_profile.json';save(profile_path,profile)
    run('associate_vehicle_objects',[PY,T/'reuse/run_vehicle_objects.py','--profile',profile_path,'--output',objects])
    result=read(objects/'summary.json');assert result['source_map_sha256']==H
    status('complete',delivery=str(O),objects=str(objects),summary=result,static_hold=False,map_sha256=H,note='No dynamic Gaussian removal; object labels are offline predictions, not evaluated accuracy')
try:main()
except BaseException as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise
