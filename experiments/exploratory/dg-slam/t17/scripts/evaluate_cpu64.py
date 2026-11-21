"""Preserve frozen evaluator; promote inputs before its unchanged metric formulas."""
from pathlib import Path
import hashlib,importlib.util,json,numpy as np
F=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
L=json.loads((F/'execution_lock.json').read_text());repair=json.loads((F/'receipts/CPU_eval_precision_repair.json').read_text())
assert sha(Path(__file__))==repair['wrapper_sha256']
assert sha(F/'scripts/evaluate.py')==L['evaluator_sha256']==repair['frozen_evaluator_sha256']
spec=importlib.util.spec_from_file_location('frozen_t17_evaluator',F/'scripts/evaluate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
raw=m.metrics
def metrics64(gt,est):return raw(np.asarray(gt,dtype=np.float64),np.asarray(est,dtype=np.float64))
m.metrics=metrics64
launch=json.loads((F/'receipts/registered_launch.json').read_text());assert launch['status']=='ALL_SIX_TERMINAL'
with np.load(F/'receipts/evaluation_reference.npz') as z:truth=m.mats(z['reference'])
checks=[]
for entry in launch['runs']:
 if entry['exit_code']!=0:continue
 with np.load(Path(entry['folder'])/'keyframe_poses.npz') as z:
  k=z['indices'].astype(int)
  for name in ['coarse','tracked','mapped']:
   item={'seed':entry['seed'],'arm':entry['arm'],'kind':name,'original_dtype':str(z[name].dtype)}
   try:raw(truth[k],z[name]);item['original_Horn_check']='PASS'
   except AssertionError:item['original_Horn_check']='FLOAT32_CENTERING_CHECK_FAILED'
   metrics64(truth[k],z[name]);item['float64_Horn_check']='PASS';checks.append(item)
(F/'receipts/CPU_eval_precision_diagnosis.json').write_text(json.dumps({'status':'PASS_FLOAT64_ALL_SPARSE','checks':checks,'frozen_evaluator_unchanged':True,'Horn_MSE_tolerance_unchanged':1e-12,'no_GPU_reruns':True},indent=2))
m.main()
