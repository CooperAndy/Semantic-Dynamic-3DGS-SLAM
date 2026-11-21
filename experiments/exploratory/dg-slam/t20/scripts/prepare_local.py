"""Additive preparation from frozen T17/T19 sources; no historical writes."""
import ast
import datetime
import hashlib
import json
import shutil
from pathlib import Path

F=Path(__file__).resolve().parents[1];B=F.parent
old=B/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012'
candidate=B/'CHY_T19_补全接口修复与CPU验证_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (F/'protocol.json').exists()
for root in [old,candidate]:
    for entry in json.loads((root/'delivery_manifest.json').read_text(encoding='utf-8'))['files']:
        file=root/entry['path']
        assert file.stat().st_size==entry['bytes'] and sha(file)==entry['sha256'],str(file)
shutil.copytree(old/'source',F/'source')
registration=json.loads((candidate/'diagnostic_registration.json').read_text(encoding='utf-8'))
for entry in registration['candidate_source']:
    src=candidate/'candidate'/entry['path']
    assert sha(src)==entry['sha256']
    dst=F/'source'/entry['path']
    if entry['base_sha256']:assert sha(dst)==entry['base_sha256']
    shutil.copy2(src,dst)
for part in ['inputs','receipts','runs']:(F/part).mkdir()
shutil.copy2(old/'inputs/DEV200_runtime.csv',F/'inputs/DEV200_runtime.csv')
for name in ['runtime_boot.py','live_refinement.py','pose_math.py','bounded_step_cpu.py']:
    shutil.copy2(old/'scripts'/name,F/'scripts'/name)
shutil.copy2(candidate/'scripts/analyze_trace.py',F/'scripts/analyze_trace.py')
shutil.copy2(B/'CHY_正式Tracking采用验收_20261012/archive/scripts/native_gpu_check.py',F/'scripts/native_gpu_check.py')
shutil.copy2(candidate/'diagnostic_registration.json',F/'T19_diagnostic_registration.json')
p=dict(registration)
p.update(task='T20_T19_REGISTERED_GPU_FILLER_DIAGNOSIS',owner='CHY',
         remote_root='/root/autodl-tmp/dg_slam_chy_20261011/11_T20_filler_GPU_20261012',
         prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         scope='Execute unchanged T19 exposed-DEV mechanism registration; no new formal quality acceptance',
         seeds=[2718],arms=['NO_GS_REFINE','LOCAL_ID'],
         registration_sha256=sha(candidate/'diagnostic_registration.json'),
         inherited_T17_manifest_sha256=sha(old/'delivery_manifest.json'),
         inherited_T19_manifest_sha256=sha(candidate/'delivery_manifest.json'),
         budgets={'minimum_free_bytes':5*1024**3,'per_worker_timeout_seconds':1200,'trace_reserved_max_per_case':256*1024**2},
         scope_addition='Native GPU/real-network engineering smoke before both registered full-state builds; no GT scoring in smoke')
dump(F/'protocol.json',p)
runner=(old/'scripts/run_tracking.py').read_text(encoding='utf-8')
start=runner.index('  seeds=torch.from_numpy(')
end=runner.index(' except Exception as e:',start)
tail='''  np.savez(output/'keyframe_poses.npz',indices=ids,coarse=np.asarray([v['coarse_c2w'] for v in frame_rows]),tracked=np.asarray([v['tracked_c2w'] for v in frame_rows]),mapped=matrices,droid_before_injection=model.video.poses[:n].cpu().numpy())
  from run_filler_cases import run_cases
  case_results=run_cases(model,matrices,data,output,args0.arm,audit)
  assert audit['blocked_GT_opens']==0 and counts['gs_mapping_calls']>0
  assert (counts['camera_backward']==0 if args0.arm=='NO_GS_REFINE' else counts['camera_backward']>0)
  if args0.arm=='NO_GS_REFINE':assert np.allclose(np.asarray([v['coarse_c2w'] for v in frame_rows]),np.asarray([v['tracked_c2w'] for v in frame_rows]),atol=1e-6)
  size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file());assert size<=2*1024**3
  meta.update(status='COMPLETED',droid_keyframes=n,mapped_keyframes=model.mapping_counter,gaussian_count=int(arrays['xyz'].shape[0]),mapping_identity_audit='PASS',first_pose_identity=True,
   output_bytes=size,diagnostic_cases=[v['case'] for v in case_results],native_video_not_injected_before_snapshot=True,
   artifacts={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='run_manifest.json'})
'''
runner=runner[:start]+tail+runner[end:]
runner=runner.replace("exp_name='T17_'+args0.arm","exp_name='T20_'+args0.arm")
(F/'scripts/run_tracking.py').write_text(runner,encoding='utf-8')
for file in F.rglob('*.py'):ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
dump(F/'receipts/local_preparation.json',{'status':'PASS','T17_files_verified':281,'T19_files_verified':80,
     'candidate_sha256_matches_registration':True,'syntax':'PASS','runtime_GT_files_copied':False,
     'historical_sources_read_only':True,'GPU_runs_before_preparation':0})
print(json.dumps({'status':'PASS','source_files':sum(1 for p in (F/'source').rglob('*') if p.is_file()),'protocol_sha256':sha(F/'protocol.json')}))
