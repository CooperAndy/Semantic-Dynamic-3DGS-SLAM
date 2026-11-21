import json,sys,csv,glob,collections,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

S=('Scene01','Scene02','Scene06','Scene18');R=Path('/root/autodl-tmp/stage2_new_model_pi3mos');O=R/'run_all_clone_20261105';V=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted');v=Path(glob.glob('/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_*/vehicle_only_v2')[0]);P={s:v/'g2_online_dev_full_v2/prediction_root'/s/'clone/instances' for s in S};P['Scene20']=v/'scene20_formal_recovery_one_shot_20260929_v3/test_output_scene20_vehicle_20260929_recovery_03/prediction_root/Scene20/clone/instances'

for sc,n in [('Scene01',439),('Scene02',225),('Scene06',262),('Scene18',331)]:
 if len(list((O/sc).glob('[0-9]*.npz'))) != n: raise RuntimeError('incomplete '+sc)
allsummary={}
for sc in S:
 pred=O/sc;gtfile=v/'g1_v2_candidate_recheck/sealed_gt'/f'{sc}_vehicle_gt.json';gt=json.loads(gtfile.read_text())['observations'];groups=collections.defaultdict(list)
 available={int(x.stem) for x in pred.glob('[0-9]*.npz')}
 for r in gt:
  if r['sequence']==sc+'/clone' and r['state'] in ('Static','Dynamic') and int(r['frame_t']) in available and int(r['frame_t1'])==int(r['frame_t'])+1:groups[int(r['frame_t'])].append(r)
 @lru_cache(maxsize=2)
 def frame(t):
  meta=json.loads((P[sc]/f'{t:06d}.json').read_text())['instances']; z=np.load(P[sc]/f'{t:06d}_masks.npz');m=z['masks'].astype(bool);lab=np.asarray(Image.open(V/sc/'clone/frames/instanceSegmentation/Camera_0'/f'instancegt_{t:05d}.png'));assert m.shape[1:]==lab.shape==(375,1242);return meta,m,lab
 def assign(ids,meta,masks,lab):
  scores=np.zeros((len(ids),len(meta)));areas=np.bincount(lab.ravel().astype(np.int32));predareas=masks.reshape(len(masks),-1).sum(1) if len(masks) else []
  for j,item in enumerate(meta):
   if item.get('category','').lower() not in ('car','truck','bus','van'):continue
   k=int(item['mask_index']);inter=np.bincount(lab[masks[k]].astype(np.int32),minlength=len(areas))
   for i,g in enumerate(ids):
    g+=1;a=int(inter[g]) if g<len(inter) else 0;ga=int(areas[g]) if g<len(areas) else 0;u=ga+int(predareas[k])-a;scores[i,j]=a/u if u else 0
  if not scores.size:return {}
  rr,cc=linear_sum_assignment(-scores);return {ids[i]:int(j) for i,j in zip(rr,cc) if scores[i,j]>=.5}
 rows=[];extras=[];ignored=0
 for t,obs in sorted(groups.items()):
  cur,m0,lab0=frame(t);nxt,m1,lab1=frame(t+1);ids=sorted({int(a['track_id']) for a in obs});a0=assign(ids,cur,m0,lab0);a1=assign(ids,nxt,m1,lab1);dep=np.asarray(Image.open(V/sc/'clone/frames/depth/Camera_0'/f'depth_{t:05d}.png'));z=np.load(pred/f'{t:06d}.npz');score={str(i):float(v) for i,v in zip(z['track_ids'],z['scores']) if np.isfinite(v)};used=set();gt_ranges={}
  for r in obs:
   gid=int(r['track_id']);d=dep[lab0==gid+1];d=d[(d>0)&(d<65535)];n=int(((d>=50)&(d<=4000)).sum());rg='unknown' if not len(d) else 'inside' if n else 'outside';gt_ranges[gid]=rg
   i=a0.get(gid);j=a1.get(gid);pid=str(cur[i]['track_id']) if i is not None else None;match=i is not None and j is not None and pid==str(nxt[j]['track_id']);val=score.get(pid) if match else None
   if val is not None:used.add(pid)
   rows.append({'frame':t,'gt_track':gid,'state':r['state'],'gt_range':rg,'inside_pixels':n,'pred_track':pid,'paired_match':match,'score':val})
  reverse={str(cur[i]['track_id']):gid for gid,i in a0.items()}
  for item in cur:
   if item.get('category','').lower() not in ('car','truck','bus','van'):continue
   pid=str(item['track_id'])
   if pid in used:continue
   gid=reverse.get(pid);extras.append({'frame':t,'pred_track':pid,'score':score.get(pid),'one_frame_gt':gid,'gt_range':gt_ranges.get(gid,'none')})
 def metrics(rng,conditional):
  c=collections.Counter()
  for r in rows:
   if rng=='40m' and r['gt_range']!='inside':continue
   dyn=r['state']=='Dynamic';c['gt_dynamic' if dyn else 'gt_static']+=1;s=r['score']
   if s is None:
    c['missing']+=1
    if not conditional:c['fn' if dyn else 'fp']+=1
   else:
    c['scored']+=1;c['tp' if dyn and s>=.5 else 'fn' if dyn else 'fp' if s>=.5 else 'tn']+=1
  if not conditional:
   for r in extras:
    if r['score'] is None:c['unmatched_unknown_score']+=1;continue
    if r['score']<.5:continue
    if rng=='40m' and r['gt_range'] in ('outside','unknown'):continue
    if rng=='40m' and r['gt_range']=='none':c['unmatched_no_gt_range_counted']+=1
    c['unmatched_dynamic_fp']+=1;c['fp']+=1
  tp,fp,fn=c['tp'],c['fp'],c['fn'];c['precision']=tp/(tp+fp) if tp+fp else None;c['recall']=tp/(tp+fn) if tp+fn else None;c['f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None;return dict(c)
 summary={'scene':sc,'gt_file':str(gtfile),'instance_source':str(P[sc]),'predicted_frame_count':len(available),'labelled_frame_count':len(groups),'gt_objects':len(rows),'unique_gt_vehicles':len({r['gt_track'] for r in rows}),'unmatched_predictions':len(extras),'threshold':.5,'40m_conditional':metrics('40m',True),'40m_strict_proxy':metrics('40m',False),'all_conditional':metrics('all',True),'all_strict':metrics('all',False),'note':'40m strict proxy assumes unmatched predictions with no GT match have unknown predicted depth and counts dynamic unknown conservatively; no predicted-depth test; static missing counted FP by historical convention; skip frames with no GT observations.'}
 d=O/sc/'evaluation_pre20';d.mkdir(exist_ok=False)
 for name,data in [('vehicle_rows.csv',rows),('unmatched_rows.csv',extras)]:
  if data:
   with (d/name).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (d/'summary.json').write_text(json.dumps(summary,indent=2));allsummary[sc]=summary;print(sc,json.dumps(summary,ensure_ascii=False),flush=True)
(O/'four_scenes_pre20_summary.json').write_text(json.dumps(allsummary,indent=2));print('DONE scoring',flush=True)
