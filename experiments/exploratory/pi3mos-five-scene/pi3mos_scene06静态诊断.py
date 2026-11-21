import json,numpy as np
from pathlib import Path
from PIL import Image
b=Path('/root/autodl-tmp/chy_vkitti_kitti_style_20261103');p=b/'scene06_joint_20261104/p1/Scene06/clone';c=b/'scene06_candidates_20261104/candidates';v=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted/Scene06/clone/frames/instanceSegmentation/Camera_0');out=Path('/root/autodl-tmp/stage2_new_model_pi3mos/diagnosis_20261105')
def best(t):
 gt=np.asarray(Image.open(v/f'instancegt_{t:05d}.png'))==15; meta=json.loads((p/'instances'/f'{t:06d}.json').read_text())['instances']; ms=np.load(p/'instances'/f'{t:06d}_masks.npz')['masks'];z=np.load(c/f'{t:06d}.npz');n,h,w=map(int,(z['n'],z['h'],z['w']));raw=np.unpackbits(z['packed'],axis=1)[:,:h*w].reshape(n,h,w).astype(bool);scores=z['scores'];areas=z['areas'];
 def metric(a):return float(np.count_nonzero(a&gt)/max(1,np.count_nonzero(a|gt)))
 a=[(metric(m),int(it['track_id'])) for m,it in zip(ms,meta)];q=[(metric(m),float(sc),int(ar)) for m,sc,ar in zip(raw,scores,areas)];return {'gt_pixels':int(gt.sum()),'post_best':max(a,default=(0,None)),'post_iou50_tracks':[tid for i,tid in a if i>=.5],'raw_best':max(q,default=(0,None,None)),'raw_iou50_count':sum(i>=.5 for i,sc,ar in q),'scorearea_iou50_count':sum(i>=.5 and sc>=.3 and ar>=32 for i,sc,ar in q)}
r=[]
for t in [243,244,245,247,248,249]:
 a=best(t);d=best(t+1);r.append({'frame':t,'t':a,'t1':d,'common_tracks':sorted(set(a['post_iou50_tracks'])&set(d['post_iou50_tracks']))})
(out/'scene06_static_diagnosis.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
