"""Record factual completion, provenance, and unchanged historical source hashes."""
from pathlib import Path
import json
import hashlib
import importlib.metadata
import datetime
import shutil
import subprocess
from tum_window import ROOT, DATA, MODEL, ONEFORMER_SHA256, sha256

def main():
    reports=ROOT/'05_reports/cpu_prepare'
    checks={s:json.loads((reports/f'cpu_check_{s}.json').read_text()) for s in ['inputs','models','droid_weights','scatter']}
    assert all(v['status']=='PASS' and not v['cuda_initialized_after'] for v in checks.values())
    assert not list(Path('/dev').glob('nvidia*')) and not (DATA/'seg_mask').exists()
    assert sha256(MODEL/'pytorch_model.bin')==ONEFORMER_SHA256
    assert sha256(ROOT/'01_source_env/weights/droid.pth')=='46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526'
    original=json.loads((ROOT/'01_source_env/official_source_locks.json').read_text())
    diffs=[]
    for file in original['files']:
        p=ROOT/'01_source_env/DG-SLAM'/file['path']
        current=sha256(p)
        if current!=file['sha256']:diffs.append({'path':file['path'],'original_sha256':file['sha256'],'current_sha256':current})
    (reports/'dg_original_source_diff_hashes.json').write_text(json.dumps(diffs,indent=2))
    files=[]
    for folder in [ROOT/'scripts',ROOT/'01_source_env/patches']:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)})
    history=json.loads((reports/'historical_source_locks.json').read_text())
    unchanged=[]
    for item in history:
        if 'sha256' not in item:
            assert 'No such file' in item.get('error','') and not Path(item['server_path']).exists()
            unchanged.append({'path':item['server_path'],'matches_before':True,'state':'ABSENT_BEFORE_AND_AFTER'})
            continue
        current=sha256(item['server_path'])
        unchanged.append({'path':item['server_path'],'sha256':current,'matches_before':current==item['sha256']})
    assert all(x['matches_before'] for x in unchanged), 'Historical source changed: inspect before signing isolation'
    packages=sorted([{'name':d.metadata['Name'],'version':d.version,'location':str(d.locate_file(''))} for d in importlib.metadata.distributions()],key=lambda x:(x['name'].lower(),x['location']))
    (reports/'runtime_packages.json').write_text(json.dumps(packages,indent=2))
    # Requirement-style resolved lock without copying VCS URLs or credentials.
    chosen={}
    for d in importlib.metadata.distributions():chosen.setdefault(d.metadata['Name'].lower(),(d.metadata['Name'],d.version))
    (reports/'requirements.resolved.txt').write_text('\n'.join(f'{name}=={version}' for name,version in sorted(chosen.values()))+'\n')
    trial=ROOT/'01_source_env/dependencies/DROID-SLAM'
    trial_sources=[]
    for p in sorted(trial.rglob('*')):
        if p.is_file() and '.git' not in p.parts and 'build' not in p.parts and '__pycache__' not in p.parts and p.suffix in ['.cpp','.cu','.h','.py']:
            trial_sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'bytes':p.stat().st_size})
    (reports/'preserved_trial_source_hashes.json').write_text(json.dumps(trial_sources,indent=2))
    logs=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(reports.glob('*.log')) if p.name!='final_audit.log']
    (reports/'preserved_log_hashes.json').write_text(json.dumps(logs,indent=2))
    for label,folder in [('droid',trial),('lietorch',trial/'thirdparty/lietorch')]:
        (reports/f'preserved_{label}_git_status.txt').write_text(subprocess.check_output(['git','-C',str(folder),'status','--short'],text=True))
        (reports/f'preserved_{label}_tracked.diff').write_text(subprocess.check_output(['git','-C',str(folder),'diff'],text=True))
    metadata=[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for p in sorted(MODEL.iterdir()) if p.is_file() and p.suffix in ['.json','.txt','.md']]
    result={'status':'CPU_ASSETS_READY_NATIVE_BUILD_DEFERRED_RESOURCE_LIMIT','timestamp':datetime.datetime.now().astimezone().isoformat(),'gpu_started':False,'gpu_nodes':[],'masks_generated':0,'slam_executed':False,'checks':{s:v['status'] for s,v in checks.items()},'native_extensions_status':'NOT_COMPILED_NOT_ACCEPTED','build_deferred_by_user':True,'build_trial_preserved':str(trial),'clean_source_candidate':'01_source_env/dependencies/DROID-SLAM_runtime (not compiled)','model_metadata_hashes':metadata,'disk_free_gib':round(shutil.disk_usage(ROOT).free/1024**3,2),'cgroup_memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'source_original_commit':original['commit'],'changed_original_files':diffs,'historical_sources':unchanged,'prepared_scripts_and_patches':files,'environment_isolation':'venv local overrides; shared base torch/cu128 read-only; base not modified','remaining':['Compile and accept native extensions in a larger-memory environment; prepared build entry is untested','GPU forward/backward and scatter PTX / FAISS verification','Real OneFormer semantic inference','30-frame R0 engineering smoke','200-frame DEV after smoke assessment','Full 828-frame and GT-free variants are future work']}
    result['cgroup_memory_events']=Path('/sys/fs/cgroup/memory.events').read_text()
    result['failure_cause_limits']='Compiler processes were killed under the fixed 2GB / 0.5 CPU quota; memory.events reports no kernel OOM kill, so the exact kill source is not proven. The later experimental CUDA-header separation also has a compile error.'
    (reports/'final_cpu_preparation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['status','timestamp','checks','disk_free_gib','gpu_started','masks_generated','slam_executed']},indent=2))

if __name__=='__main__':main()
