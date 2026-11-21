"""Freeze A2 code, calibration and DEV input manifest before mapper runs."""
import hashlib,json
from pathlib import Path
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');B=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200')
paths=[O/'A2_DEV_protocol_20261028.md',O/'a2_losses.py',O/'test_a2_losses.py',O/'a2_gradient_calibration.py',O/'gradient_calibration_20261028.json',O/'a2_mapper_worker.py',O/'a2_score_mapper.py',O/'mapper_v1/source/scripts/splatam_external_tm.py',O/'mapper_v1/source/c04n_addition.py',O/'mapper_v1/source/manifest_dataset.py',B/'inputs/runtime/input_manifest.json',B/'source/configs/kitti/G00_S0.py']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
out=O/'A2_run_registration_20261028.json';assert not out.exists()
data={'state':'REGISTERED_EXPLORATORY_DEV','file_locks':[{'path':str(p),'sha256':sha(p)} for p in paths],'quality_adoption':False,'notes':'gamma is gradient matching, not noise calibration; inputs individually verified by worker'}
out.write_text(json.dumps(data,indent=2));print('registered',len(paths),'files',sha(out))
