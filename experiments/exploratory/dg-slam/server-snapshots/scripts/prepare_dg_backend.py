"""Export a fixed DROID backend matching DG's four-argument AltCorr interface."""
from pathlib import Path
import subprocess
import json
import hashlib
import re

R=Path('/root/autodl-tmp/dg_slam_chy_20261011')
repo=R/'01_source_env/dependencies/DROID-SLAM'
out=repo.with_name('DROID-backend_dg_api')
commit='c2f604a4def91513c7bdfabfb7f8aede5194ab79'
assert not out.exists(), 'Do not overwrite an existing source candidate'
(out/'src').mkdir(parents=True)
records=[]
for name in ['droid.cpp','droid_kernels.cu','correlation_kernels.cu','altcorr_kernel.cu']:
    data=subprocess.check_output(['git','-C',str(repo),'show',f'{commit}:src/{name}'])
    text=data.decode()
    # PyTorch 2.x dispatch macros require ScalarType, not DeprecatedTypeProperties.
    text,count=re.subn(r'(AT_DISPATCH_FLOATING_TYPES_AND_HALF\(\w+)\.type\(\)',r'\1.scalar_type()',text)
    (out/'src'/name).write_text(text)
    records.append({'file':name,'original_sha256':hashlib.sha256(data).hexdigest(),
                    'patched_sha256':hashlib.sha256(text.encode()).hexdigest(),'dispatch_api_replacements':count})
setup=subprocess.check_output(['git','-C',str(repo),'show','2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99:setup.py'])
(out/'setup.py').write_bytes(setup)
(out/'thirdparty').mkdir()
(out/'thirdparty/lietorch').symlink_to(repo.with_name('DROID-SLAM_runtime')/'thirdparty/lietorch',target_is_directory=True)
v={'backend_commit':commit,'setup_commit':'2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99',
   'files':records,'source':str(out),'reason':'DG AltCorrLayer expects four-argument forward / five-argument backward',
   'algorithm_bodies_changed':False,'dispatch_api_replacements':sum(x['dispatch_api_replacements'] for x in records)}
(R/'01_source_env/patches/dg_backend_compatibility.json').write_text(json.dumps(v,indent=2))
print(json.dumps(v,indent=2))
