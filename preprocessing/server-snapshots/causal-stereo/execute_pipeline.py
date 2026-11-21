"""Serial full-window stereo/confidence execution after G20 and a concrete method freeze."""
from common import *
from resource_guard import run_guarded,check_resources
import argparse,shutil,subprocess,signal,time,re,traceback

LIMITS=dict(free_bytes=8*2**30,raw_bytes=18*2**30,combined_bytes=27*2**30)
START_FREE=36*2**30

def verify_cpu():
    initial=json.loads((R/'registration_cpu.json').read_text())
    for name,digest in initial['source_files'].items():assert sha(R/name)==digest,name
    stage=json.loads((R/'execution_stage_registration.json').read_text())
    for name,digest in stage['sources'].items():assert sha(R/name)==digest,name
    contexts=json.loads((R/'execution_cpu_registration.json').read_text())
    for context in contexts['contexts']:
        for name,digest in context['files'].items():assert sha(Path(context['path'])/name)==digest,name
    assert json.loads((R/'trace_numpy_contract_receipt.json').read_text())['status']=='PASS_REAL_NUMPY_TOFILE_WRITE_CLOSE_TRACE_AND_ARRAY_BYTES'
    assert json.loads((R/'resource_guard_contract_receipt.json').read_text())['status']=='PASS_REAL_CHILD_SUCCESS_OWNED_TIMEOUT_DISK_GATE_AND_UNRELATED_PROCESS_PRESERVATION'
    return initial,contexts

def readiness(initial):
    parent=R.parent/'stage3_g20_evidence_20261029_p47149'
    analysis=parent/'analysis_controller.json';quality=R.parent/'stage3_g20_20261029_p47149/evaluation_only_slam_eval/quality_checks.json';archive=parent/'final_delta_status.json'
    freeze=R/'formal_method_freeze.json';missing=[]
    if not analysis.exists() or not json.loads(analysis.read_text()).get('status','').startswith('COMPLETE'):missing.append('G20_FULL_EVALUATION')
    if not quality.exists():missing.append('G20_QUALITY_RESULTS')
    if not archive.exists() or not json.loads(archive.read_text()).get('status','').startswith('COMPLETE_STRICT_POST_EVALUATION'):missing.append('G20_SERVER_SCIENCE_ARCHIVE_CLOSED')
    if not freeze.exists():missing.append('CONCRETE_FORMAL_METHOD_FREEZE')
    if shutil.disk_usage(R).free<START_FREE:missing.append('36GIB_INITIAL_SPACE')
    if missing:return dict(ready=False,reasons=missing,free_bytes=shutil.disk_usage(R).free)
    frozen=json.loads(freeze.read_text())
    assert frozen['status']=='FROZEN_AFTER_G20_DEV_BEFORE_NEW_WINDOW_GT'
    assert frozen['new_window_GT_consulted'] is False
    assert frozen['windows']==[x['id'] for x in initial['windows']] and frozen['backend_seeds']==[0,1,2]
    assert frozen['g20_analysis_sha256']==sha(analysis) and frozen['g20_quality_sha256']==sha(quality)
    assert frozen['methods'] and len({x['id'] for x in frozen['methods']})==len(frozen['methods'])
    for method in frozen['methods']:
        assert re.fullmatch(r'[A-Za-z0-9_-]+',method['id'])
        assert Path(method['config_path']).is_file() and sha(method['config_path'])==method['config_sha256']
    return dict(ready=True,freeze_sha256=sha(freeze),g20_analysis_sha256=sha(analysis),g20_quality_sha256=sha(quality))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    initial,contexts=verify_cpu();ready=readiness(initial)
    if args.check_only:print(json.dumps(dict(status='READY' if ready['ready'] else 'WAITING_REGISTERED_DEPENDENCIES',GPU_calls=0,**ready)));return
    assert ready['ready'],ready
    assert not (R/'execution_registration.json').exists() and not (E/'controller.json').exists()
    frozen=json.loads((R/'formal_method_freeze.json').read_text());(R/'frozen_method_configs/methods').mkdir(parents=True)
    for method in frozen['methods']:
        target=R/'frozen_method_configs/methods'/(method['id']+Path(method['config_path']).suffix)
        shutil.copy2(method['config_path'],target);assert sha(target)==method['config_sha256']
    shutil.copy2(R.parent/'stage3_g20_evidence_20261029_p47149/analysis_controller.json',R/'frozen_method_configs/G20_analysis.json')
    shutil.copy2(R.parent/'stage3_g20_20261029_p47149/evaluation_only_slam_eval/quality_checks.json',R/'frozen_method_configs/G20_quality.json')
    check_resources(LIMITS)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    assert int(gpu.stdout.strip().splitlines()[0])>=8192,'Need8GiB GPU headroom;preserve other jobs'
    for win in initial['windows']:
        W=R/'windows'/win['id']
        assert not any((W/name).exists() for name in ['raw_left','raw_right','cache','registration.json'])
    files={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    dump(R/'execution_registration.json',dict(status='REGISTERED_FULL_TWO_WINDOW_STEREO_CONFIDENCE_BEFORE_GPU',at=now(),source_files=files,limits=LIMITS,start_free_bytes=START_FREE,readiness=ready,windows=[w['id'] for w in initial['windows']],phase_GPU_seconds=7200,confidence_seconds=1200,total_seconds=90000,quality_adoption=False))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',PYTHONPATH=str(R))
    gpu_env=dict(env);gpu_env.pop('CUDA_VISIBLE_DEVICES')
    started=time.monotonic();steps=[]
    def run(name,argv,timeout,gpu=False):
        assert time.monotonic()-started<90000
        result=run_guarded(name,argv,timeout,gpu_env if gpu else env,LIMITS);steps.append(result)
        dump(E/'completed_steps.json',dict(at=now(),steps=steps))
    run('archive_source',['python3',str(R/'archive_stereo.py'),'--label','source'],600)
    for win in initial['windows']:
        wid=win['id'];W=R/'windows'/wid
        for item in contexts['planned_stereo_commands']:
            if item['window']==wid:
                side=item['step'];run(wid+'_'+side,item['argv'],7200,gpu=True)
                run(wid+'_'+side+'_audit',[str(CPU),str(R/'audit_stereo_trace.py'),'--window',wid,'--side',side],900)
        run(wid+'_register_confidence',[str(CPU),str(R/'register_confidence.py'),'--window',wid],900)
        run(wid+'_confidence',['strace','-f','-ttt','-yy','-s','256','-e','trace=openat,close,write','-o',str(W/'file_activity.log'),str(CPU),str(W/'scripts/producer.py')],1200)
        run(wid+'_output_audit',[str(CPU),str(W/'scripts/audit.py')],900)
        run(wid+'_access_audit',[str(CPU),str(R/'audit_confidence_access.py'),'--window',wid],900)
        # The next window's inference overlaps the previous transfer. Before
        # compression, release the previous verified archive to keep one copy.
        if wid!=initial['windows'][0]['id']:
            previous=initial['windows'][0]['id'];wait=time.monotonic()
            while not (E/'archives'/(previous+'_remote_compressed_cleanup.json')).exists():
                check_resources(LIMITS);assert time.monotonic()-wait<86400
                dump(E/'backup_wait.json',dict(status='WAITING_PREVIOUS_WINDOW_FULL_LOCAL_BACKUP_AND_COMPRESSED_RELEASE',at=now(),window=previous));time.sleep(15)
        run(wid+'_archive',['python3',str(R/'archive_stereo.py'),'--label',wid],3600)
    dump(E/'prediction_complete.json',dict(status='COMPLETE_TWO_WINDOWS_1000_STEREO_AND_PAST_CONFIDENCE_AUDITS_BEFORE_UNION',at=now(),windows=[w['id'] for w in initial['windows']],GT_scoring=False,quality_adoption=False))
    run('archive_union',['python3',str(R/'archive_stereo.py'),'--label','union'],3600)
    dump(E/'controller.json',dict(status='COMPLETE_TWO_WINDOWS_FULL_STEREO_CONFIDENCE_AUDITS_AND_SERVER_SCIENCE_UNION',at=now(),seconds=time.monotonic()-started,windows=[w['id'] for w in initial['windows']],steps=steps,GT_scoring=False,quality_adoption=False))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(RuntimeError('OWN_CONTROLLER_STOP')))
    try:main()
    except BaseException as err:
        # A failed precondition must not fabricate a running or completed GPU stage.
        dump(E/'execution_failure.json',dict(status='FAILED_PRESERVED',at=now(),error=repr(err),traceback=traceback.format_exc()));raise
