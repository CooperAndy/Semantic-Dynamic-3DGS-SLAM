from pathlib import Path
import csv,hashlib,json,numpy as np
r=Path('/root/autodl-tmp/seq21_chy_20260926');p=r/'protocol'
f0=r/'runs/train160_orb_raft_t060/params.npz';f1=r/'runs/F1_train160/params.npz'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
with np.load(f0,allow_pickle=False) as a,np.load(f1,allow_pickle=False) as b:
 assert set(a.files)==set(b.files)
 changed={'rgb_colors','logit_opacities'}
 fixed={k:bool(np.array_equal(a[k],b[k])) for k in a.files if k not in changed}
 assert all(fixed.values()),[k for k,v in fixed.items() if not v]
 assert a['means3D'].shape==b['means3D'].shape

def read_csv(path):
 with path.open(newline='') as h:return {int(v['original_frame_id']):v for v in csv.DictReader(h)}
a=read_csv(r/'eval/Dev_F0/per_frame_metrics.csv');b=read_csv(r/'eval/Dev_F1/per_frame_metrics.csv')
assert set(a)==set(b) and len(a)==20
rows=[]
for i in sorted(a):
 rows.append({'frame':i,'delta_psnr':float(b[i]['psnr_db'])-float(a[i]['psnr_db']),'delta_lpips':float(b[i]['lpips'])-float(a[i]['lpips']),'delta_hole':float(b[i]['hole_rate'])-float(a[i]['hole_rate'])})
mean=lambda k:float(np.mean([v[k] for v in rows]))
dev={'count':20,'delta_psnr_db':mean('delta_psnr'),'delta_lpips':mean('delta_lpips'),'delta_hole':mean('delta_hole'),'positive_psnr_frames':sum(v['delta_psnr']>0 for v in rows),'fixed_field_checks':fixed,'per_frame':rows}
(r/'eval/dev_pair_summary.json').write_text(json.dumps(dev,indent=2)+'\n')
m=json.loads((p/'protocol_manifest.json').read_text())
e=p/'evaluate_seq21_batch.py'
decision={'status':'FROZEN_BEFORE_TEST_RGB_READ','split_sha256':m['split_sha256'],'roi_mask_ledger_sha256':m['roi_mask_ledger_sha256'],'evaluator_sha256':sha(e),'pre_registered_success':m['pre_registered_success'],'candidate_order':[{'name':'F0','map_path':str(f0),'map_sha256':sha(f0)},{'name':'F1','map_path':str(f1),'map_sha256':sha(f1)}],'dev_pair_summary_sha256':sha(r/'eval/dev_pair_summary.json')}
with (p/'frozen_test_decision.json').open('x') as h:json.dump(decision,h,indent=2)
print('TEST_FROZEN',json.dumps({k:dev[k] for k in ('delta_psnr_db','delta_lpips','delta_hole','positive_psnr_frames')}))
