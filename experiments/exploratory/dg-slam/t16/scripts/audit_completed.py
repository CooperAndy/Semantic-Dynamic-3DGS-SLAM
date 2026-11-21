"""Independent CPU audit of branch-aware gates, camera updates, seals and old archives."""
from pathlib import Path
import ast,hashlib,json,numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
F=Path(__file__).resolve().parents[1];B=F.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(2**20),b''):h.update(a)
 return h.hexdigest()
def verify(root,manifest):
 m=json.loads((root/manifest).read_text(encoding='utf-8'))
 for r in m['files']:
  p=root/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
 return len(m['files'])
P=json.loads((F/'repair_protocol.json').read_text(encoding='utf-8'));S=np.asarray(P['optimizer_recipe']['scale']);lock=json.loads((F/'repair_execution_lock.json').read_text())
assert sha(F/'repair_protocol.json')==lock['protocol_sha256']
for name,h in lock['scripts'].items():assert sha(F/'scripts'/name)==h,name
for name,h in lock['receipts'].items():assert sha(F/'receipts'/name)==h,name
raw=json.loads((F/'receipts/replay_raw_seal.json').read_text())
for path,h in raw['files'].items():assert sha(F/path)==h,path
gate=json.loads((F/'receipts/repaired_numerical_gate.json').read_text());assert gate['status']=='PASS' and gate['original_T15']=='FAIL_UNCHANGED'
conditional=json.loads((F/'receipts/conditional_derivatives.json').read_text())
cos=lambda x,y:float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
for row in gate['cases']:
 fds=np.asarray(next(r['FD_gradients'] for r in conditional['cases'] if r['frame']==row['frame']));g=np.asarray(row['conditional_gradient'])
 relative=np.linalg.norm(g-fds[1])/max(np.linalg.norm(g),np.linalg.norm(fds[1]),1.)
 scale=max(np.linalg.norm(v-fds[1])/max(np.linalg.norm(fds[1]),1.) for v in fds)
 assert abs(relative-row['conditional_relative_L2'])<1e-12 and abs(scale-row['conditional_FD_scale_relative'])<1e-12
 assert abs(cos(g,fds[1])-row['conditional_cosine'])<1e-12 and all(row['tests'].values())
maximum=0.;accepted_total=0;iterations_total=0;maxcap=0.;runchecks=[]
for frame in P['frames']:
 for arm in P['optimizer_recipe']['arms']:
  path=F/f'replays/f{frame}_{arm}';m=json.loads((path/'run_manifest.json').read_text());props=json.loads((path/'proposals.json').read_text())
  with np.load(path/'trace.npz') as z:a={k:v.copy() for k,v in z.items()}
  assert len(a['pre_c2w'])==len(a['post_c2w'])==len(a['gradients'])==len(a['accepted'])==m['iterations']
  assert int(a['accepted'].sum())==m['accepted']
  expected_C=np.eye(6) if arm=='LOCAL_ID' else np.diag(np.diag(a['P_FULL'])) if arm=='LOCAL_DIAG_P' else a['P_FULL']
  assert np.array_equal(expected_C,a['C']) and np.linalg.eigvalsh(a['C'])[0]>0
  assert np.array_equal(a['initial_c2w'],a['pre_c2w'][0]) and np.array_equal(a['candidate_c2w'],a['post_c2w'][-1])
  for k,(pre,post,g,step,ok) in enumerate(zip(a['pre_c2w'],a['post_c2w'],a['gradients'],a['step'],a['accepted'])):
   xi=S*step;w=xi[3:];W=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
   L=np.zeros((4,4));L[:3,:3]=W;L[:3,3]=xi[:3];expected=(pre.astype(float)@expm(L)).astype(np.float32)
   maximum=max(maximum,float(np.max(np.abs(expected-post))));assert np.max(np.abs(expected-post))<1e-6
   trials=[r for r in props if r['iteration']==k];d=-(a['C']@g);d/=np.linalg.norm(d)
   assert np.linalg.norm(step)<=.05+1e-12
   for r in trials:
    u=np.asarray(r['step']);assert np.allclose(u,.05*.5**r['backtrack']*d,rtol=1e-12,atol=1e-14)
    assert abs(r['armijo_bound']-(a['loss'][k]+1e-4*g@u))<=1e-8
   if ok:
    assert trials[-1]['loss'] is not None and trials[-1]['loss']<=trials[-1]['armijo_bound']
    assert np.array_equal(step,np.asarray(trials[-1]['step']))
   else:assert not step.any() and np.array_equal(pre,post)
   if arm=='LOCAL_FULL_CAP' and ok:maxcap=max(maxcap,trials[-1]['anchor_distance']);assert trials[-1]['anchor_distance']<=.25+1e-12
  assert np.all(np.diff(a['loss'])<=0) and m['final_loss']<=m['initial_loss']
  assert m['GT_audit']['blocked_GT_opens']==0 and m['map_unchanged'] and m['rng_unchanged']
  accepted_total+=m['accepted'];iterations_total+=m['iterations'];runchecks.append({'frame':frame,'arm':arm,'status':'PASS'})
old={}
for name,manifest in [('CHY_T15_DEV同图回放_20261012','delivery_manifest.json'),('CHY_正式Tracking采用验收_20261012','delivery_manifest.json'),('DGSLAM_队友交接_截至T10_20261012/DGSLAM_队友交接_截至T10_20261012','文件清单_SHA256.json')]:old[name]=verify(B/name,manifest)
for p in (F/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
evaluation=json.loads((F/'receipts/replay_evaluation.json').read_text());assert evaluation['eligible_full_window_candidate'] is None
csv_hash=sha(F/'T16_八次同图回放结果.csv');assert csv_hash=='f225fd67137259368ed6578033c2a8a23e5580f814b8b964494577b5c07ffd4c'
out={'status':'PASS','run_checks':runchecks,'camera_iterations':iterations_total,'accepted_updates':accepted_total,'expm_max_abs_error':maximum,
 'maximum_accepted_anchor_distance_CAP':maxcap,'branch_gate_recomputed':'PASS','old_files_unchanged':old,'full_window_runs':0,'formal_runs':0,
 'scoring_serialization_repair':{'scope':'numpy.bool_ converted to Python bool for JSON only; formulas, thresholds and CSV unchanged','CSV_sha256_before_and_after':csv_hash},
 'raw_seal_sha256':sha(F/'receipts/replay_raw_seal.json')}
(F/'receipts/completed_independent_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
