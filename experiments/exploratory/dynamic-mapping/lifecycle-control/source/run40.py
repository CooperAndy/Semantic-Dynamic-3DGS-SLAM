import sys,time,runpy,json,os
from pathlib import Path
import torch
r=Path('/root/autodl-tmp/stage2_round12_20261031_v1');sys.path.insert(0,str(r/'p5_snapshot'));sys.path.insert(0,str(r/'source'));sys.argv=[str(r/'source/splatam_audited.py'),str(r/'source/config40.py')];start=time.perf_counter();ok=False
try:runpy.run_path(sys.argv[0],run_name='__main__');ok=True
finally:
 d={'success':ok,'elapsed_seconds':time.perf_counter()-start,'last_interval_peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'last_interval_peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576};(Path(os.environ['P5_AUDIT_OUT'])/'run_metrics.json').write_text(json.dumps(d,indent=2));print('METRICS',json.dumps(d),flush=True)
