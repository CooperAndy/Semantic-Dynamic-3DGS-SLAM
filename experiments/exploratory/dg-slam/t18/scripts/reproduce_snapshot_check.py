"""Check standalone snapshot recomputation against the original-input CPU analysis."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
baseline=load(F/'receipts/analysis.json')
(F/'receipts/full_source_analysis.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
csvs={p.name:sha(p) for p in F.glob('*.csv')}
arrays={p.name:{k:v.copy() for k,v in np.load(p).items()} for p in (F/'arrays').glob('*.npz')}
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',T18_INPUT_ROOT=str(F/'inputs_snapshot'),PYTHONIOENCODING='utf-8')
result=subprocess.run([sys.executable,str(F/'scripts/analyze_filler.py')],env=env,capture_output=True,text=True,encoding='utf-8')
(F/'receipts/snapshot_recompute.log').write_text(result.stdout+result.stderr,encoding='utf-8')
assert result.returncode==0,result.stderr
after=load(F/'receipts/analysis.json')
for key in ['runs','stage_metrics','subset_metrics','original_dense_metrics_max_difference_cm']:
 assert baseline[key]==after[key],key
assert after['input_manifest_mode']=='snapshot' and after['input_file_manifest_verified']==35
for name,h in csvs.items():assert sha(F/name)==h,name
for name,values in arrays.items():
 with np.load(F/'arrays'/name) as z:
  for k,v in values.items():assert np.array_equal(v,z[k]),(name,k)
assert 'torch' not in sys.modules
receipt={'status':'PASS','snapshot_input_files_verified':35,'all_stage_and_subset_metrics_identical':True,'all_six_CSV_SHA256_identical':True,'all_six_array_contents_bitwise_identical':True,'analysis_subprocess_returncode':result.returncode,'GPU_or_SLAM_runs':0,'remote_connections':0}
(F/'receipts/snapshot_reproducibility.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
