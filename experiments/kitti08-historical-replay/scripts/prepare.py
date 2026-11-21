from common import *
import ast,copy,shutil,numpy as np
assert not (R/'plan.json').exists()
I=R/'data_pipeline';result=json.loads((I/'input_result.json').read_text());assert result['status']=='PASS_FRESH_GT_FREE_CROSS_WINDOW_INPUTS' and result['frames']==N
(R/'inputs/runtime').mkdir()
m=json.loads((I/'input_manifest.json').read_text());assert m['frame_count']==N and (m['H'],m['W'])==(370,1226)
shutil.copy2(m['pose_file'],R/'inputs/runtime/external_orb_P2_c2w.txt');m.update(pose_file=str(R/'inputs/runtime/external_orb_P2_c2w.txt'),mapping_holdout_frames=HOLD);assert sha(m['pose_file'])==m['pose_sha256']
dump(R/'inputs/runtime/input_manifest.json',m)
for row in m['rows']:
 for k in ['rgb','right','depth','confidence']:assert sha(row[k])==row['hashes'][k]
base=R.parent/'stage3_g13_20261027_p30915'
parent=ast.literal_eval(ast.parse((base/'source/configs/kitti/B00C0_S0.py').read_text()).body[0].value)
data_cfg=R/'source/configs/data/g00_manifest.yaml'
data_cfg.write_text('dataset_name: kitti\ninput_manifest: '+str(R/'inputs/runtime/input_manifest.json')+'\ncamera_params:\n  image_height: 370\n  image_width: 1226\n  fx: 707.0912\n  fy: 707.0912\n  cx: 601.8873\n  cy: 183.1104\n  png_depth_scale: 1.0\n  crop_edge: 0\n')
for v in VARIANTS:
 c=copy.deepcopy(parent);seed=definition(v)['seed']
 c.update(workdir=str(R/'runs'),run_name=v,task_id='T3-G14',variant_id=v,seed=seed,eval_only_frames=HOLD,factorial=definition(v),scope='KITTI08 1000-frame exposed DEV,3 backend seeds,within-pair frozen H0 births/targets')
 c['data'].update(num_frames=N,sequence='08',gradslam_data_cfg=str(data_cfg),desired_image_height=m['H'],desired_image_width=m['W'])
 c['mapping'].update(sampling_seed=seed,residual_fb=380.34753)
 (R/'source/configs/kitti'/f'{v}.py').write_text('config = '+repr(c)+'\n')
assert sha(R/'source/scripts/splatam_external_tm.py')==CORE_SHA
dump(R/'plan.json',dict(task_id='T3-G14',root=str(R),variants=VARIANTS,frames=N,mapping_holdout=HOLD,backend_seeds=[0,1,2],shared_frontend_runs=1,
 budget=dict(per_run_seconds=5400,batch_seconds=33000,new_output_bytes_max=16*2**30,archive_output_bytes_max=12*2**30,disk_remaining_min_bytes=5*2**30),
 scope='Cross-window paired DEV,donor H0 inputs frozen before same-seed H3,not independent frontend or tracking gain',quality_adoption=False))
dump(R/'receipts/input_preparation.json',dict(status='PASS4000_INPUT_HASHES_AND1000_POSES',at=now(),input_hashes=4000,steps_per_run=57000,history_slots_per_H3=2280,shared_frontend_runs=1))
print('PREPARE_PASS',flush=True)
