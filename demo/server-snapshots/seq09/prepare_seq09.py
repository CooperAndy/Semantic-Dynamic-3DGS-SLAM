"""Isolated seq09 S23 input preparation. No GT, no mutation of prior runs."""
from pathlib import Path
import json, hashlib, shutil, subprocess, sys

ROOT = Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114')
S = Path('/root/autodl-tmp/s23_effects_20261105_p16247')
D = Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
B = ROOT / 'bundle'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
B.mkdir(exist_ok=False)
image = D / 'sequences/09'
pose = D / 'poses_estimated_camera2.txt'
for name, target in [('raw_left', D/'raft_left'), ('raw_right', D/'raft_right'), ('semantic', D/'annotations_20261107/semantic'), ('instances', D/'annotations_20261107/instances')]:
    (B/name).symlink_to(target)
shutil.copy2(image/'calib.txt', B/'calib.txt')
lines = (image/'calib.txt').read_text().splitlines()
P2 = list(map(float, next(x for x in lines if x.startswith('P2:')).split()[1:]))
P3 = list(map(float, next(x for x in lines if x.startswith('P3:')).split()[1:]))
beta = P2[3]-P3[3]
(B/'projection_beta.txt').write_text(str(beta)+'\n')
context = {'window': {'primary_pose_path':str(pose), 'primary_pose_sha256':sha(pose), 'K':[[P2[0],P2[1],P2[2]],[P2[4],P2[5],P2[6]],[P2[8],P2[9],P2[10]]], 'projection_beta':beta, 'shape':[370,1226], 'input_view':str(image), 'source_start':300}}
(B/'context.json').write_text(json.dumps(context,indent=2))
code=(S/'confidence_bundle.py').read_text().replace('(1000,12)','(200,12)').replace('(1000,1,1)','(200,1,1)').replace('range(1000)','range(200)').replace('frame_count=1000','frame_count=200')
(ROOT/'confidence_200.py').write_text(code)
py='/root/venvs/semantic_seg/bin/python'
def run(args, log):
    with (ROOT/log).open('x') as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)
run([py,str(ROOT/'confidence_200.py'),str(B),str(B/'context.json')], 'confidence_200.log')
print('Confidence and runtime ready', flush=True)
