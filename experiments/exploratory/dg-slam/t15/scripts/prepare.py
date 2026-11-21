"""Freeze numerical gates and eight conditional DEV replays before GPU execution."""
from pathlib import Path
import ast, hashlib, json, shutil
from datetime import datetime, timezone
F=Path(__file__).resolve().parents[1];BASE=F.parent
OLD=BASE/'CHY_接手DGSLAM_T11起_20261012'
PKG=BASE/'DGSLAM_队友交接_截至T10_20261012/DGSLAM_队友交接_截至T10_20261012/阶段3/DGSLAM_方案核查_20261011'
T14=BASE/'CHY_正式Tracking采用验收_20261012'
REMOTE='/root/autodl-tmp/dg_slam_chy_20261011/08_T15_DEV_replay_20261012'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
 for d in ['inputs','receipts','runs','scripts']: (F/d).mkdir(parents=True,exist_ok=True)
 assert not (F/'protocol.json').exists()
 old=json.loads((OLD/'T12_GPU执行_20261012/execution_lock.json').read_text(encoding='utf-8'))
 inputs={};cases=[]
 for spec in old['map_cases']:
  frame=spec['frame'];case={k:spec[k] for k in ['frame','keyframe','map_path','case_path']}
  for key in ['map_path','case_path']:
   src=PKG/'t02_receipts'/Path(spec[key].split('/dg_slam_chy_20261011/')[1])
   assert sha(src)==old['immutable_inputs'][spec[key]]
   dst=F/'inputs'/src.name;shutil.copyfile(src,dst);inputs[str(dst.relative_to(F)).replace('\\','/')]=sha(dst)
   case[key]=REMOTE+'/inputs/'+src.name
  cases.append(case)
 source_manifest=PKG/'t02_receipts/04_reproduction/T02_200_DIAG_20261011_195357/run_manifest.json'
 assert sha(source_manifest)==old['immutable_inputs'][str(old['original_manifest'])]
 dst=F/'inputs/original_run_manifest.json';shutil.copyfile(source_manifest,dst);inputs['inputs/'+dst.name]=sha(dst)
 for name in ['pose_math.py','bounded_step_cpu.py']:
  src=T14/'archive/scripts'/name;shutil.copyfile(src,F/'scripts'/name)
 observer=PKG/'t02_scripts/t02_observer.py'
 dump(F/'receipts/snapshot_provenance.json',{'status':'PASS','cases':cases,'observer_source_sha256':sha(observer),
  'evidence':'Observer.finish asserts live_map_unchanged_during_tracking then saves live_map_before_frame, before mapping dispatch resumes.',
  'scope':'Previously exposed old DEV frames 47 and 95; no future-map substitution; original run immutable hashes verified.'})
 p={'task':'T15_CHY_DEV_MECHANISM_REPLAY','created_at':datetime.now(timezone.utc).isoformat(),
    'remote_root':REMOTE,'t14_root':'/root/autodl-tmp/dg_slam_chy_20261011/07_formal_tracking_T14_20261012',
    'cases':cases,'input_hashes':inputs,'owner':'CHY','scope':'DEV ONLY; no formal adoption or new Test',
    'scale':[.02,.02,.02,0.017453292519943295,0.017453292519943295,0.017453292519943295],
    'support':'At canonicalized coarse T0: finite observed depth in (0,8), finite rendered depth, static seg, 10px border exclusion, initial opacity >.95, >=1000 pixels; freeze within snapshot and across arms.',
    'loss':'0.9*sum(abs(RGB-render))+0.1*sum(clamp(abs(depth-render_depth),max=1000)); identical fixed support',
    'P':'Central residual Jacobian at T0, physical eps .001m/.01deg; H=(.9 JcT Jc+.1 JdT Jd)/N; trace-normalized eigen inverse sqrt with damping .01; Frobenius norm sqrt(6)',
    'numeric_gate':{'L1_FD_normalized_h':[.001,.002,.004],'primary_h':.002,
      'gradient_relative_L2_max':.15,'gradient_cosine_min':.98,'FD_gradient_scale_relative_max':.20,
      'ID_FULL_directional_relative_max':.15,'matrix_eps_multipliers':[.5,1.,2.],
      'P_relative_Frobenius_max':.20,'Pg_direction_cosine_min':.98,
      'repeat_forward_exact':True,'decision':'Both snapshots must pass every gate before any of eight optimization replays. On failure stop replays and diagnose numerical component; do not relax gates.'},
    'arms':['LOCAL_ID','LOCAL_FULL','LOCAL_DIAG_P','LOCAL_FULL_CAP'],
    'seed':1219,'repeats':1,'iteration_max':50,'step_radius':.05,'backtracks':6,'backtrack_factor':.5,'Armijo_c1':1e-4,
    'total_anchor_radius':.25,'cap_definition':'||S^-1 log(inv(T0) T_candidate)||_2 <= .25; outside proposals rejected before loss render and continue same six backtracks; never project or consult GT.',
    'cap_rationale':'One prior DEV choice: five maximum single steps; bounds pure translation by5mm or pure rotation by.25deg; no radius search or per-case tuning.',
    'diag_definition':'diag(diag(P_FULL)), no re-estimation from diag(H), same normalized direction rule',
    'identity_binding':['initial pose','map tensor bytes','observations','support','H/P','seed/RNG state','initial rendered outputs'],
    'runtime_GT':'Forbidden; evaluate only after numerical output/eight replay raw outputs sealed; fixed SE3 fit on original full coarse keyframes, never per-candidate alignment.',
    'candidate_progression':{'non_ID_arms':'All numeric gates pass; for BOTH frames final position error <=.95*ID; rotation <=1.05*ID+1e-6rad; both RGB/depth loss<=1.05*ID and coverage drop<=.02; choose smallest mean position error, fixed tie order DIAG_P,FULL_CAP,FULL. Otherwise no full-window GPU experiment.',
       'interpretation':'DEV screening only; n=2 exposed snapshots does not establish formal quality or generalization.',
       'full_window':'If one candidate passes, separately register old DEV 0:200, seed1219, ID and candidate; include NO_GS_REFINE to assess GS benefit. All formal T14 windows remain exposed.'},
    'limits':{'max_replays':8,'new_snapshots':0,'new_data_downloads':0,'formal_reruns':0,'minimum_free_bytes':5368709120}}
 dump(F/'protocol.json',p)
 scripts={}
 for x in sorted((F/'scripts').glob('*.py')):
  ast.parse(x.read_text(encoding='utf-8'));scripts[x.name]=sha(x)
 dump(F/'execution_lock.json',{'protocol_sha256':sha(F/'protocol.json'),'scripts':scripts,'inputs':inputs,
       'T14_delivery_manifest_sha256':sha(T14/'delivery_manifest.json'),'T12_execution_lock_sha256':sha(OLD/'T12_GPU执行_20261012/execution_lock.json')})
 print(json.dumps({'status':'REGISTERED_BEFORE_GPU','protocol_sha256':sha(F/'protocol.json'),'cases':[c['frame'] for c in cases],'R':p['total_anchor_radius']},ensure_ascii=False))
if __name__=='__main__':main()
