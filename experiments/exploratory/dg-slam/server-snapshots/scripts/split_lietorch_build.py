"""Split translation units without changing any operation (2GB no-card build)."""
from pathlib import Path
import re
import json
import hashlib
import difflib

R=Path('/root/autodl-tmp/dg_slam_chy_20261011')
root=R/'01_source_env/dependencies/DROID-SLAM/thirdparty/lietorch'
records=[]
for kind,suffix in [('cpu','.cpp'),('gpu','.cu')]:
    path=root/f'lietorch/src/lietorch_{kind}{suffix}'
    text=path.read_text()
    matches=list(re.finditer(r'^(?:torch::Tensor|std::vector<torch::Tensor>)\s+(\w+)\(',text,re.MULTILINE))
    assert len(matches)==19
    prefix=text[:matches[0].start()]
    folder=root/f'lietorch/src/split_{kind}'
    folder.mkdir(exist_ok=True)
    header=root/f'lietorch/src/{kind}_kernels_impl.h'
    header.write_text('#pragma once\n'+prefix)
    blocks=[]
    for i,m in enumerate(matches):
        block=text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)]
        blocks.append(block)
        destination=folder/f'{m.group(1)}{suffix}'
        destination.write_text(f'#include "../{header.name}"\n'+block)
        records.append({'file':str(destination.relative_to(root)),'operation':m.group(1),'unchanged_body_sha256':hashlib.sha256(block.encode()).hexdigest()})
    assert prefix+''.join(blocks)==text
setup=root/'setup.py'
before=setup.read_text()
after=before.replace('from setuptools import setup','from setuptools import setup\nimport glob')
for kind,suffix in [('cpu','.cpp'),('gpu','.cu')]:
    after=after.replace(f"'lietorch/src/lietorch_{kind}{suffix}'",f"*sorted(glob.glob('lietorch/src/split_{kind}/*{suffix}'))")
after=after.replace("'cxx': ['-O0', '-g0']","'cxx': ['-O2', '-g0']")
setup.write_text(after)
(R/'01_source_env/patches/lietorch_split_manifest.json').write_text(json.dumps(records,indent=2))
(R/'01_source_env/patches/lietorch_split_setup.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='lietorch/setup.py.before_split',tofile='lietorch/setup.py')))
print('Split 38 wrappers; original operation bodies unchanged; all 4 Lie groups retained')
