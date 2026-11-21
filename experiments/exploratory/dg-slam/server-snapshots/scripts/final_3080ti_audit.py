"""Accept preparation for this GPU without claiming a SLAM reproduction."""
from pathlib import Path
import datetime
import difflib
import importlib.metadata
import json
import shutil
import subprocess
import torch
import importlib
from tum_window import ROOT,DATA,MODEL,sha256,ONEFORMER_SHA256

reports=ROOT/'05_reports/3080ti_prepare'
assert (reports/'pip_check.txt').read_text().strip()=='No broken requirements found.'
assert torch.cuda.get_device_capability()==(8,6)
checks={}
for stage in ['inputs','models','droid_weights','scatter','native']:
    result=json.loads((reports/f'cpu_check_{stage}.json').read_text())
    assert result['status']=='PASS' and not result['cuda_initialized_after']
    assert result['torch_version']==torch.__version__ and result['target_arch']=='8.6'
    checks[stage]=result['status']
gpu_path=Path((reports/'gpu_sanity_receipt_path.txt').read_text().strip())
gpu=json.loads(gpu_path.read_text())
assert gpu['status']=='PASS' and gpu['gpu']==torch.cuda.get_device_name()
oneformer=json.loads((reports/'oneformer_gpu_check.json').read_text())
assert oneformer['status']=='PASS' and oneformer['actual_network_forward']
model_init=json.loads((reports/'model_init_check.json').read_text())
assert model_init['status']=='PASS' and not model_init['tracking_mapping_executed']
assert gpu['droid_legacy_altcorr_forward_backward']=='PASS' and gpu['droid_BA_and_frame_distance']=='PASS'
assert sha256(MODEL/'pytorch_model.bin')==ONEFORMER_SHA256
assert not (DATA/'seg_mask').exists(), 'Experiment masks were outside this preparation run'
history=json.loads((ROOT/'05_reports/cpu_prepare/historical_source_locks.json').read_text())
preserved=[]
for item in history:
    if 'sha256' in item:
        assert sha256(item['server_path'])==item['sha256']
        preserved.append({'path':item['server_path'],'sha256':item['sha256']})
    else: assert not Path(item['server_path']).exists()
trial=json.loads((ROOT/'05_reports/cpu_prepare/preserved_trial_source_hashes.json').read_text())
for item in trial: assert sha256(ROOT/item['path'])==item['sha256']
changes=[]
patch=[]
trial_source=ROOT/'01_source_env/dependencies/DROID-SLAM'
runtime=trial_source.with_name('DROID-SLAM_runtime')
for original,current in [(trial_source,runtime),(trial_source/'thirdparty/lietorch',runtime/'thirdparty/lietorch')]:
    tracked=subprocess.check_output(['git','-C',str(original),'ls-tree','-r','--name-only','HEAD'],text=True).splitlines()
    for relative in tracked:
        file=current/relative
        if not file.is_file(): continue
        before=subprocess.check_output(['git','-C',str(original),'show','HEAD:'+relative])
        after=file.read_bytes()
        if before!=after:
            changes.append({'path':str(file.relative_to(ROOT)),'sha256':sha256(file)})
            patch.extend(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='pinned/'+relative,tofile=str(file.relative_to(ROOT))))
(reports/'runtime_source_changes.diff').write_text(''.join(patch))
backend=json.loads((ROOT/'01_source_env/patches/dg_backend_compatibility.json').read_text())
backend_diff=[]
for item in backend['files']:
    file=Path(backend['source'])/'src'/item['file']
    assert sha256(file)==item['patched_sha256']
    before=subprocess.check_output(['git','-C',str(trial_source),'show',f"{backend['backend_commit']}:src/{item['file']}"])
    assert __import__('hashlib').sha256(before).hexdigest()==item['original_sha256']
    if file.read_bytes()!=before:
        backend_diff.extend(difflib.unified_diff(before.decode().splitlines(True),file.read_text().splitlines(True),fromfile='pinned_backend/src/'+item['file'],tofile=str(file.relative_to(ROOT))))
(reports/'backend_torch_dispatch_compat.diff').write_text(''.join(backend_diff))
native=[]
for name in ['lietorch_backends','lietorch_extras','droid_backends','diff_gaussian_rasterization_depth_pose._C']:
    module=importlib.import_module(name)
    file=Path(module.__file__)
    listing=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--list-elf',str(file)],text=True)
    assert '.sm_86.' in listing, name
    native.append({'module':name,'path':str(file),'sha256':sha256(file),'native_sm_86':True})
source=json.loads((ROOT/'01_source_env/official_source_locks.json').read_text())
for item in source['files']:
    file=ROOT/'01_source_env/DG-SLAM'/item['path']
    current=sha256(file)
    if current!=item['sha256']:changes.append({'path':str(file.relative_to(ROOT)),'sha256':current,'original_sha256':item['sha256']})
files=[]
for folder in [ROOT/'scripts',ROOT/'01_source_env/patches']:
    for file in sorted(folder.rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts:
            files.append({'path':str(file.relative_to(ROOT)),'sha256':sha256(file)})
packages=sorted([{'name':d.metadata['Name'],'version':d.version,'location':str(d.locate_file(''))} for d in importlib.metadata.distributions()],key=lambda x:(x['name'].lower(),x['location']))
(reports/'runtime_packages.json').write_text(json.dumps(packages,indent=2))
selected={}
for d in importlib.metadata.distributions():selected.setdefault(d.metadata['Name'].lower(),(d.metadata['Name'],d.version))
(reports/'requirements.resolved.txt').write_text('\n'.join(f'{n}=={v}' for n,v in sorted(selected.values()))+'\n')
result={'status':'PREPARATION_ACCEPTED_FOR_3080TI','timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'gpu':torch.cuda.get_device_name(),'compute_capability':[8,6],'torch':torch.__version__,
        'cpu_checks':checks,'gpu_operators':gpu,'oneformer_actual_firstframe':oneformer,'dg_model_initialization':model_init,
        'native_binaries':native,'droid_backend_provenance':backend,'pip_check':'PASS',
        'native_source_changes':changes,'historical_unchanged':preserved,
        'preserved_trial_source_files':len(trial),'scripts_and_patches':files,
        'disk_free_gib':round(shutil.disk_usage(ROOT).free/1024**3,2),
        'environment':'Layered venv; base torch/CUDA Python packages reused read-only; no base installs',
        'experiment_masks_generated':0,'slam_executed':False,
        'remaining':['30-frame official-reference smoke','200-frame DEV after smoke assessment','Full 828-frame and GT-free variants']}
(reports/'final_preparation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['status','timestamp','gpu','cpu_checks','preserved_trial_source_files','disk_free_gib','slam_executed']},indent=2))
