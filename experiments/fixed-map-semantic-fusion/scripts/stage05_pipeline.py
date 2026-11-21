import subprocess,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'05_条件轻量候选与历史诊断/dev_diagnostics_20261007_v2'
sources=['stage05_dev_diagnostics.py','stage05_gpu_run.py','stage05_evaluate.py','stage05_pipeline.py','semantic_cpu.py','native_semantic_render.py','evaluate_vkitti_gt_dev.py']
locks={s:hashlib.sha256((R/'scripts'/s).read_bytes()).hexdigest() for s in sources}
subprocess.run(['/root/miniconda3/bin/python',str(R/'scripts/stage05_dev_diagnostics.py')],check=True)
(O/'quality_policy.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_05_NEW_GT_SCORES',baseline='S2',candidates=['S2-V','S2-Z'],minimum_equal_scene_mean_mIoU_gain=.01,maximum_single_scene_mIoU_loss=.02,maximum_single_scene_coverage_loss=.02,maximum_subgroup_loss=.02,required_subgroups=['small_vehicle','non_small_vehicle','boundary','interior'],minimum_subgroup_GT_pixels=100,role='exposed DEV diagnostic only; no new isolated final acceptance; default S0 remains'),ensure_ascii=False,indent=2),encoding='utf-8')
(O/'source_snapshots').mkdir()
for s in sources:shutil.copy2(R/'scripts'/s,O/'source_snapshots'/s)
(O/'source_sha256.json').write_text(json.dumps(locks,indent=2)+'\n')
subprocess.run(['/root/miniconda3/bin/python',str(R/'scripts/stage05_gpu_run.py')],check=True)
subprocess.run(['/root/miniconda3/bin/python',str(R/'scripts/stage05_evaluate.py')],check=True)
assert all(hashlib.sha256((R/'scripts'/s).read_bytes()).hexdigest()==v for s,v in locks.items())
for filename in ['input_locks.json','GPU_input_locks.json']:
 for r in json.loads((O/filename).read_text()):assert hashlib.sha256((R/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
(O/'pipeline_result.json').write_text(json.dumps({'status':'COMPLETE_05_DEV_DIAGNOSTICS_AND_CONDITIONAL_CANDIDATES','sources_and_inputs_post_verified':True,'Stage2_or_KITTI360_final_accessed':False,'default_S0_unchanged':True},indent=2)+'\n')
print('COMPLETE_05_PIPELINE',flush=True)
