import json,csv,sys,collections,math
from pathlib import Path
import numpy as np
from PIL import Image
scene=sys.argv[1];assert scene in ('Scene01','Scene06');b=Path('/root/autodl-tmp/chy_vkitti_kitti_style_20261103');run=Path('/root/autodl-tmp/stage2_new_model_pi3mos/run_20261105')/scene
if scene=='Scene01':
 pred=b/'day1_v1_20261103/predictions/Scene01_p1_bins1/Scene01/clone';matched=b/'day1_v1_20261103/matched_Scene01_p1_bins1.json';score=b/'day1_v1_20261103/score_Scene01_p1_bins1'
else:
 pred=b/'scene06_joint_20261104/p1/Scene06/clone';matched=b/'scene06_joint_20261104/matched_p1.json';score=b/'scene06_joint_20261104/score_p1'
read=lambda p:list(csv.DictReader(open(p,newline='',encoding='utf-8')))
prs=read(score/'gt_rows.csv');extras=read(score/'unmatched_predictions.csv'); recs=json.loads(matched.read_text())['predictions'];mapping={(int(r['frame_t']),str(r['track_id'])):str(r['predicted_track_id']) for r in recs if not str(r['track_id']).startswith('pred:')}
validframes={int(p.name[:6]) for p in run.glob('*_prob.npz')};values={};sizes={}; gtdepth={};V=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted')/scene/'clone/frames'
for t in sorted(validframes):
 z=np.load(run/f'{t:06d}_prob.npz')['prob'];meta=json.loads((pred/'instances'/f'{t:06d}.json').read_text())['instances'];masks=np.load(pred/'instances'/f'{t:06d}_masks.npz')['masks'];
 assert z.shape==masks.shape[1:]==(375,1242) and np.isfinite(z).all()
 for it in meta:
  mask=masks[int(it['mask_index'])].astype(bool); vals=z[mask];key=(t,str(it['track_id']));assert key not in values
  values[key]=float(np.median(vals)) if len(vals) else None;sizes[key]=int(len(vals))
 lab=np.asarray(Image.open(V/'instanceSegmentation/Camera_0'/f'instancegt_{t:05d}.png'));dep=np.asarray(Image.open(V/'depth/Camera_0'/f'depth_{t:05d}.png'))
 for r in (q for q in prs if int(q['frame'])==t):
  tid=str(r['track_id']);d=dep[(lab==int(tid)+1)&(dep>0)&(dep<65535)];gtdepth[(t,tid)]=float(np.median(d)/100) if len(d) else None
rows=[]
for r in prs:
 t=int(r['frame']);tid=str(r['track_id']);
 if t not in validframes:continue
 pid=mapping.get((t,tid));ps=values.get((t,pid)) if pid is not None else None;p1=float(r['score']) if r['predicted']=='True' and r['score'] else None
 rows.append({'frame':t,'gt_id':tid,'state':r['state'],'range':r['gt_range'],'depth_median_m':gtdepth[(t,tid)],'pred_track_id':pid,'pixels':sizes.get((t,pid),0),'pi3_score':ps,'p1_score':p1})
ex=[]
for r in extras:
 t=int(r['frame']);
 if t not in validframes:continue
 pid=str(r['track_id']);ps=values.get((t,pid)); p1=float(r['score']) if r['score'] else None
 ex.append({**r,'pi3_score':ps,'p1_score':p1})
def calc(rs,es,model,thr,mode,common):
 c=collections.Counter()
 for r in rs:
  if mode=='40' and r['range']!='inside':continue
  d=r['depth_median_m']
  if mode in ('near','mid','far','depth_unknown'):
   band='depth_unknown' if d is None else 'near' if .5<=d<=20 else 'mid' if 20<d<=40 else 'far' if d>40 else 'depth_unknown'
   if mode!=band:continue
  v=r[model+'_score']; other=r['p1_score' if model=='pi3' else 'pi3_score'];dyn=r['state']=='Dynamic'
  c['gt_dynamic' if dyn else 'gt_static']+=1
  if common and (v is None or other is None):continue
  if v is None:
   c['missing']+=1
   if common: continue
   c['fn' if dyn else 'fp']+=1;continue
  c['scored']+=1;c['tp' if dyn and v>=thr else 'fn' if dyn else 'fp' if v>=thr else 'tn']+=1
 if not common and mode in ('40','all'):
  for r in es:
   v=r[model+'_score']
   if v is None:c['extra_unknown']+=1;continue
   if v<thr:continue
   if mode=='40' and (r['matched_t_gt_range'] in ('outside','unknown') or r['pred_range']=='outside'):continue
   c['extra_dynamic_fp']+=1;c['fp']+=1
 tp,fp,fn=c['tp'],c['fp'],c['fn']; c['f1']=2*tp/(2*tp+fp+fn) if 2*tp+2*fp+2*fn else None;c['precision']=tp/(tp+fp) if tp+fp else None;c['recall']=tp/(tp+fn) if tp+fn else None
 return dict(c)
summary={}
for mode in ('40','all','near','mid','far','depth_unknown'):
 summary[mode]={}
 for model in ('pi3','p1'):
  summary[mode][model]={'common':calc(rows,ex,model,.5,mode,True),'strict' if mode in ('40','all') else 'gt_only':calc(rows,ex,model,.5,mode,False)}
summary['thresholds_40_common']={str(t):calc(rows,ex,'pi3',t,'40',True) for t in (.25,.4,.5,.6,.75)}
summary['meta']={'valid_frames':len(validframes),'first':min(validframes),'last':max(validframes),'gt_rows':len(rows),'extras':len(ex),'unique_gt_vehicles':len({x['gt_id'] for x in rows}),'mask_frame':'t','ground_truth_label':'t to t+1; temporal mismatch acknowledged','frame_t_available_before_t1':True}
(run/'scores.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
with open(run/'vehicle_rows.csv','w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with open(run/'extra_rows.csv','w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=list(ex[0]));w.writeheader();w.writerows(ex)
print(json.dumps(summary,indent=2))
