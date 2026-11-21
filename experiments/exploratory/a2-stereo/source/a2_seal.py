"""Seal A2 scripts, protocols, audits, maps, receipts, histories and scores."""
import hashlib,json
from pathlib import Path
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');R=O/'mapper_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
names=['A2_DEV_protocol_20261028.md','pose_audit_revision_20261028.md','A2_distance_scoring_protocol_20261028.md','A2_DEV_RESULTS_zh.md','a2_losses.py','test_a2_losses.py','cpu_tests_20261028.log','a2_gradient_calibration.py','gradient_calibration_20261028.json','A2_run_registration_20261028.json','a2_mapper_worker.py','a2_prepare_mapper.py','a2_audit_mapper.py','a2_audit_mapper_v2.py','a2_score_mapper.py','a2_score_depth_bins.py','a2_run_200.py']
files=[O/n for n in names]+[R/'source/scripts/splatam_external_tm.py',R/'batch_200.json',R/'audit_25.log',R/'independent_audit_v2_25.json',R/'independent_audit_v2_200.json',R/'evaluation_200.json',R/'evaluation_depth_bins_200.json']
for n in (25,200):
 for arm in ('B0','B1','B2','B3','B4'):
  d=R/'runs'/f'{arm}_{n}';files.extend(d/x for x in ('params.npz','a2_receipt.json','addition_history.jsonl'))
out=R/'A2_DEV_evidence_seal_20261028.json';assert not out.exists()
items=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
out.write_text(json.dumps({'status':'SEALED','files':items,'quality_adoption':False},indent=2));print('sealed',len(items),'files',sha(out))
