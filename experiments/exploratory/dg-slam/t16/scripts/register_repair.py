"""Register branch-aware numerical validation before fresh GPU checks/eight replays."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
F=Path(__file__).resolve().parents[1];OLD=F.parent/'CHY_T15_DEV同图回放_20261012'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cause=json.loads((F/'receipts/branch_comparison.json').read_text());tiles=sorted({r['tile'] for a in cause['axes'] for r in a['selected']})
assert all(r['CPU_reconstruction_pass'] for a in cause['axes'] for r in a['selected'])
old=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
p={'task':'T16_BRANCH_AWARE_DEV_VALIDATION','registered_at':datetime.now(timezone.utc).isoformat(),
 'T15_root':'/root/autodl-tmp/dg_slam_chy_20261011/08_T15_DEV_replay_20261012',
 'remote_root':'/root/autodl-tmp/dg_slam_chy_20261011/09_T16_branch_repair_20261012','frames':[47,95],
 'diagnostic_tiles':tiles,'diagnostic_axes':[3,4],'conditional_FD_h':[.001,.002,.004],'primary_h':.002,
 'gate':{'conditional_gradient_relative_L2_max':.15,'conditional_gradient_cosine_min':.98,'conditional_FD_scale_relative_max':.20,
  'global_actual_gradient_relative_L2_max':.15,'global_actual_gradient_cosine_min':.98,'global_actual_ID_FULL_direction_relative_max':.15,
  'P_scale_relative_max':.20,'Pg_scale_cosine_min':.98,'backward_repeat_relative_max':1e-5,'native_CPU_RGB_max':1e-4,'native_CPU_depth_max':1e-3},
 'repair':'Validate local derivatives with initial per-pixel order/membership, alpha/termination/clamp decisions and L1 residual signs fixed ONLY in independent CPU derivative audit. Cross-branch actual-loss FD remains recorded separately, no global all-scales smoothness assumption.',
 'scope':'New DEV validation after observed native sort switch; original T15 FAIL retained. This is not a reclassification of T15 nor evidence of FULL adoption.',
 'certificate_sha256':sha(F/'receipts/branch_comparison.json'),'trace_result_sha256':sha(F/'trace_result.json'),
 'optimizer_recipe':{k:old[k] for k in ['arms','seed','iteration_max','step_radius','backtracks','backtrack_factor','Armijo_c1','total_anchor_radius','scale','candidate_progression']},
 'actual_proposal_acceptance':'Unchanged real full native render on common frozen support, actual L1 Armijo; never use frozen-order/branch counterfactual scores or GT.',
 'entry':'Both frames must pass branch-aware regional derivative audit AND global actual objective primary checks AND original P-stability; otherwise eight replays stay stopped.',
 'freshness':'Post-hoc DEV validation repair, thresholds and optimizer recipe locked before any of the eight optimization results; no Test claims.',
 'backward_binding':'Forward and inputs exact. Repeated CUDA backward may differ by atomic roundoff; compare relative gradient norm <=1e-5 instead of bitwise equality; record difference.',
 'runtime_GT':'Forbidden; original coarse-keyframe SE3 alignment used by separate CPU evaluator only after all eight raw outputs sealed.'}
assert not (F/'repair_protocol.json').exists();(F/'repair_protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False),encoding='utf-8')
print('REPAIR_PROTOCOL_REGISTERED',sha(F/'repair_protocol.json'),'TILES',tiles)
