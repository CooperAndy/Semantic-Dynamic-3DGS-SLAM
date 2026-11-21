"""Server side: build the refine-objective-fix source tree from source_v4_depthselection.

Copies the frozen Stage A+B tree (so the depth guard and the observed-depth selector
are inherited unchanged) and replaces exactly one file, the mapper carrying the
refinement objective. Refuses to overwrite an existing tree, and verifies the parent
file hashes before and the patched hashes after the copy.

Counterpart protocol:
    00_任务边界与协议/精修目标函数修复_C_20261013.md
"""
from pathlib import Path
import hashlib
import json
import py_compile
import shutil

R = Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
O = R / 'stage_refinefix'
P = json.loads((O / 'preparation.json').read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()

target = R / P['source_tree']
assert not target.exists(), 'refusing to overwrite existing source tree: %s' % target

for name, digest in P['parent_sha256'].items():
    assert sha(R / 'source_v4_depthselection' / name) == digest, 'parent mismatch: %s' % name

shutil.copytree(R / 'source_v4_depthselection', target, symlinks=True,
                ignore=shutil.ignore_patterns('__pycache__'))

for name, digest in P['patched_sha256'].items():
    uploaded = O / 'upload' / name
    assert sha(uploaded) == digest, 'uploaded patch mismatch: %s' % name
    shutil.copy2(uploaded, target / name)
    py_compile.compile(str(target / name), doraise=True)

py_compile.compile(str(R / 'scripts/run_stage_c.py'), doraise=True)

receipt = {
    'status': 'DEPLOYED',
    'source_tree': str(target),
    'preparation_sha256': sha(O / 'preparation.json'),
    'parent_verified': sorted(P['parent_sha256']),
    'patched_applied': sorted(P['patched_sha256']),
    'worker_sha256': sha(R / 'scripts/run_stage_c.py'),
}
(O / 'deploy_receipt.json').write_text(json.dumps(receipt, indent=2))
print('REFINE_FIX_DEPLOYED_PARENT_HASHES_PASS')
