import json,csv,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/root/autodl-tmp/chy_vkitti_kitti_style_20261103');base=Path('/root/autodl-tmp/stage2_new_model_pi3mos');run=base/'run_20261105/Scene01'; out=base/'diagnosis_20261105';out.mkdir(exist_ok=True)
V=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted/Scene01/clone/frames'); P=B/'day1_v1_20261103/predictions/Scene01_p1_bins1/Scene01/clone/instances'
rows=list(csv.DictReader(open(run/'vehicle_rows.csv',encoding='utf8')));outrows=[]
for t in range(41,50):
 row=next(r for r in rows if int(r['frame'])==t and r['gt_id']=='8');pid=int(row['pred_track_id']);
 im=np.asarray(Image.open(V/f'rgb/Camera_0/rgb_{t:05d}.jpg').convert('RGB')); lab=np.asarray(Image.open(V/f'instanceSegmentation/Camera_0/instancegt_{t:05d}.png'));gt=lab==9
 meta=json.loads((P/f'{t:06d}.json').read_text())['instances'];masks=np.load(P/f'{t:06d}_masks.npz')['masks'];its=[m for m in meta if int(m['track_id'])==pid];assert len(its)==1
 pm=masks[its[0]['mask_index']].astype(bool); prob=np.load(run/f'{t:06d}_prob.npz')['prob'];union=np.logical_or(gt,pm).sum();inter=(gt&pm).sum();gy,gx=np.where(gt); py,px=np.where(pm);xs=np.r_[gx,px];ys=np.r_[gy,py];x0=max(0,int(xs.min())-25);x1=min(im.shape[1],int(xs.max())+26);y0=max(0,int(ys.min())-25);y1=min(im.shape[0],int(ys.max())+26)
 vals=prob[gt];pv=prob[pm]; record={'frame':t,'gt_track':8,'pred_track':pid,'gt_pixels':int(gt.sum()),'pred_pixels':int(pm.sum()),'iou':float(inter/union),'gt_covered':float(inter/gt.sum()),'pred_purity':float(inter/pm.sum()),'pred_median':float(np.median(pv)),'gt_median_oracle':float(np.median(vals)),'intersection_median':float(np.median(prob[pm&gt])),'gt_frac_ge_025':float((vals>=.25).mean()),'gt_frac_ge_05':float((vals>=.5).mean()),'pred_frac_ge_025':float((pv>=.25).mean()),'pred_frac_ge_05':float((pv>=.5).mean())};outrows.append(record)
 fig,ax=plt.subplots(1,4,figsize=(14,3.4)); sl=np.s_[y0:y1,x0:x1];ax[0].imshow(im[sl]);ax[0].set_title('RGB');
 for k,mask,title,color in [(1,pm,'Predicted mask','cyan'),(2,gt,'GT mask (diagnostic only)','lime')]:
  ax[k].imshow(im[sl]);ax[k].contour(mask[sl].astype(float),levels=[.5],colors=[color],linewidths=1.3);ax[k].set_title(title)
 ax[3].imshow(im[sl]);heat=ax[3].imshow(prob[sl],cmap='inferno',vmin=0,vmax=1,alpha=.65);ax[3].contour(gt[sl].astype(float),levels=[.5],colors=['lime'],linewidths=.8);ax[3].set_title('Motion probability (GT outline)');fig.colorbar(heat,ax=ax[3],fraction=.046,pad=.04)
 for a in ax:a.axis('off')
 fig.suptitle(f'Scene01 frame {t}, GT 8, pred {pid}  IoU={record["iou"]:.3f}  pred median={record["pred_median"]:.3f}  GT median={record["gt_median_oracle"]:.3f}',fontsize=11);fig.tight_layout();fig.savefig(out/f'whitecar_{t:03d}.png',dpi=120);plt.close(fig)
with (out/'whitecar.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
print(json.dumps(outrows,indent=2))
