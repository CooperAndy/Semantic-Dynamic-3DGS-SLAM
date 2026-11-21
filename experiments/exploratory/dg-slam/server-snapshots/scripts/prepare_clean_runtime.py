"""Preserve the build trial and create a separate conservative source candidate.

No compilation. Pinned git archives exclude all experimental/untracked files.
"""
from pathlib import Path
import subprocess
import tarfile
import io
import json
import hashlib
import re

R=Path('/root/autodl-tmp/dg_slam_chy_20261011')
D=R/'01_source_env/dependencies/DROID-SLAM'
OUT=D.with_name('DROID-SLAM_runtime')
assert not OUT.exists(), 'Never overwrite a prior candidate'
entries=[]
for source,target,commit in [
    (D,OUT,'2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99'),
    (D/'thirdparty/lietorch',OUT/'thirdparty/lietorch','7f687644fcea81ab337831749445224503c1290d'),
    (D/'thirdparty/lietorch/eigen',OUT/'thirdparty/lietorch/eigen','824272cde8ca2541e8b67b0887f5ded92b128d1f')]:
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert actual==commit
    data=subprocess.check_output(['git','-C',str(source),'archive',commit])
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(target,filter='data')
    entries.append({'source':str(source),'target':str(target),'commit':commit})
setup=OUT/'thirdparty/lietorch/setup.py'
before=setup.read_text()
after=re.sub(r"^.*'-gencode=arch=compute_.*\n",'',before,flags=re.M)
assert after!=before and '-gencode' not in after
setup.write_text(after)
manifest={'status':'SOURCE_CANDIDATE_NOT_COMPILED','entries':entries,
          'changes':['Remove old lietorch hardcoded gencode; target architecture is selected by build entry'],
          'trial_preserved':str(D),'source_splitting_applied':False,
          'setup_sha256':hashlib.sha256(setup.read_bytes()).hexdigest()}
(R/'01_source_env/patches/clean_runtime_source.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
