"""Independent CPU compositor and branch counterfactuals for captured native buffers.

Counterfactual images are diagnostic only. They must never score real tracking proposals.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
F=Path(__file__).resolve().parents[1]

def tile_pixels(shape,tile):
 H,W=shape;ncol=(W+15)//16;y,x=divmod(tile,ncol)
 yy,xx=np.mgrid[y*16:min((y+1)*16,H),x*16:min((x+1)*16,W)]
 return yy.ravel(),xx.ravel()

def composite(a,tile,ids=None,frozen=None):
 """Reconstruct native front-to-back RGB, unnormalized depth, opacity and cutoffs."""
 H,W=a['support'].shape;ys,xs=tile_pixels((H,W),tile)
 if ids is None:
  lo,hi=a['ranges'][tile];ids=a['point_list'][lo:hi]
 ids=np.asarray(ids,dtype=int);N=len(xs)
 T=np.ones(N,dtype=np.float32);C=np.zeros((N,3),dtype=np.float32);D=np.zeros(N,dtype=np.float32);A=np.zeros(N,dtype=np.float32)
 done=np.zeros(N,dtype=bool);last=np.zeros(N,dtype=np.uint32)
 active_rows=[];clamp_rows=[];stop_rows=[];cut_rows=[]
 for j,gid in enumerate(ids):
  xy=a['means2D'][gid];co=a['conic_opacity'][gid]
  dx=xy[0]-xs.astype(np.float32);dy=xy[1]-ys.astype(np.float32)
  power=np.float32(-.5)*(co[0]*dx*dx+co[2]*dy*dy)-co[1]*dx*dy
  # Native power is non-positive for contributing samples. Clamp only the unused
  # positive-power branch here to avoid irrelevant exp overflow in diagnostics.
  raw=co[3]*np.exp(np.minimum(power,np.float32(0)))
  saturated=raw>=np.float32(.99)
  alpha=np.minimum(np.float32(.99),raw)
  eligible=(~done)&(power<=0)&(alpha>=np.float32(1/255))
  stop=eligible&(T*(1-alpha)<np.float32(.0001))
  use=eligible&~stop
  cut=(~done)&((power>0)|(alpha<np.float32(1/255)))
  if frozen is not None:
   assert np.array_equal(ids,frozen['ids'])
   saturated=frozen['saturated'][j]
   alpha=np.where(saturated,np.float32(.99),raw)
   use=frozen['active'][j];stop=frozen['stopped'][j]
  weight=alpha*T
  C[use]+=a['rgb'][gid]*weight[use,None]
  D[use]+=a['depths'][gid]*weight[use]
  A[use]+=weight[use]
  T[use]*=1-alpha[use]
  last[use]=j+1;done|=stop
  active_rows.append(use.copy());clamp_rows.append(saturated.copy());stop_rows.append(stop.copy());cut_rows.append(cut.copy())
 return {'image':C,'depth':D,'acc':A,'final_T':T,'last':last,'ys':ys,'xs':xs,
         'branch':{'ids':ids.copy(),'active':np.asarray(active_rows,dtype=bool).reshape(-1,N),
                   'saturated':np.asarray(clamp_rows,dtype=bool).reshape(-1,N),
                   'stopped':np.asarray(stop_rows,dtype=bool).reshape(-1,N),
                   'cut':np.asarray(cut_rows,dtype=bool).reshape(-1,N)}}

def weighted_loss(a,q):
 y,x=q['ys'],q['xs'];support=a['support'][y,x]
 rgb=np.abs(q['image'].astype(float)-a['observed_color'][y,x].astype(float)).sum(-1)
 dep=np.minimum(np.abs(q['depth'].astype(float)-a['observed_depth'][y,x].astype(float)),1000.)
 return float((.9*rgb+.1*dep)[support].sum())

def native_tile(a,tile):
 y,x=tile_pixels(a['support'].shape,tile)
 return {'ys':y,'xs':x,'image':a['image'][y,x],'depth':a['render_depth'][y,x],'acc':a['acc'][y,x]}

def selfcheck():
 # Closed-form two-layer opacity composition detects ordering and depth convention.
 a={'support':np.ones((16,16),bool),'ranges':np.array([[0,2]],dtype=np.uint32),'point_list':np.array([0,1],dtype=np.uint32),
    'means2D':np.zeros((2,2),np.float32),'conic_opacity':np.array([[0,0,0,.5],[0,0,0,.5]],np.float32),
    'rgb':np.array([[1,0,0],[0,0,1]],np.float32),'depths':np.array([1,2],np.float32)}
 front=composite(a,0);reverse=composite(a,0,ids=np.array([1,0]))
 assert np.allclose(front['image'],[.5,0,.25]) and np.allclose(reverse['image'],[.25,0,.5])
 assert np.allclose(front['depth'],1.) and np.allclose(reverse['depth'],1.25)
 assert np.allclose(front['acc'],.75) and np.allclose(front['final_T'],.25)
 a['conic_opacity'][:,3]=.001;cut=composite(a,0);assert not cut['image'].any() and not cut['acc'].any()
 a['conic_opacity'][:,3]=.99;stop=composite(a,0);assert np.allclose(stop['image'],[.99,0,0],atol=1e-7) and (stop['last']==1).all()
 frozen=composite(a,0,frozen=stop['branch']);assert np.array_equal(frozen['image'],stop['image'])
 result={'status':'PASS','cases':['two-layer analytic color ordering','unnormalized depth','opacity/transmittance','alpha cutoff','early termination','frozen branch identity'],
  'limit':'Synthetic closed-form check only; actual native-buffer reconstruction remains pending until server returns.'}
 (F/'receipts/compositor_selfcheck.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

def load(p):
 with np.load(p) as z:return {k:v.copy() for k,v in z.items()}

def run():
 manifest=json.loads((F/'trace_result.json').read_text(encoding='utf-8'))
 for row in manifest['captures']:
  assert hashlib.sha256((F/row['path']).read_bytes()).hexdigest()==row['sha256']
 def capture(axis,h):return load(F/next(r['path'] for r in manifest['captures'] if r['frame']==47 and r['axis']==axis and r['h']==h))
 summary=[]
 for axis in [3,4]:
  before=capture(axis,.0035);after=capture(axis,.00375)
  assert np.array_equal(before['support'],after['support'])
  H,W=before['support'].shape;count=((H+15)//16)*((W+15)//16)
  tiles=[];order_changes=[]
  for tile in range(count):
   na=native_tile(before,tile);nb=native_tile(after,tile)
   change=weighted_loss(after,nb)-weighted_loss(before,na);tiles.append((abs(change),tile,change))
   alo,ahi=before['ranges'][tile];blo,bhi=after['ranges'][tile]
   ai=before['point_list'][alo:ahi];bi=after['point_list'][blo:bhi]
   common=np.intersect1d(ai,bi);aa=ai[np.isin(ai,common)];bb=bi[np.isin(bi,common)]
   order_changes.append({'tile':tile,'membership_changes':int(len(np.setxor1d(ai,bi))),
                         'common_order_changed':not np.array_equal(aa,bb),'loss_change':change})
  selected=sorted(tiles,reverse=True)[:8];details=[]
  for _,tile,change in selected:
   ca=composite(before,tile);cb=composite(after,tile);na=native_tile(before,tile);nb=native_tile(after,tile)
   errors={'before_RGB_max':float(np.max(np.abs(ca['image']-na['image']))),'after_RGB_max':float(np.max(np.abs(cb['image']-nb['image']))),
           'before_depth_max':float(np.max(np.abs(ca['depth']-na['depth']))),'after_depth_max':float(np.max(np.abs(cb['depth']-nb['depth'])))}
   reproduced=max(errors['before_RGB_max'],errors['after_RGB_max'])<=1e-4 and max(errors['before_depth_max'],errors['after_depth_max'])<=1e-3
   # Freeze order/membership and then contributor decisions independently. Never
   # use either counterfactual as the optimizer's real proposal score.
   same_order=composite(after,tile,ids=ca['branch']['ids'])
   same_branches=composite(after,tile,ids=ca['branch']['ids'],frozen=ca['branch'])
   y,x=na['ys'],na['xs'];valid=before['support'][y,x]
   ar=na['image'].astype(float)-before['observed_color'][y,x].astype(float)
   br=nb['image'].astype(float)-after['observed_color'][y,x].astype(float)
   signed_rgb_change=float((.9*(np.sign(ar)*(br-ar)).sum(-1))[valid].sum())
   actual_rgb_change=float((.9*(np.abs(br)-np.abs(ar)).sum(-1))[valid].sum())
   details.append({'tile':tile,'native_loss_change':change,'CPU_reconstruction':errors,'CPU_reconstruction_pass':reproduced,
     'CPU_natural_loss_change':weighted_loss(after,cb)-weighted_loss(before,ca),
     'fixed_order_membership_loss_change':weighted_loss(after,same_order)-weighted_loss(before,ca),
     'fixed_order_and_contributor_branch_loss_change':weighted_loss(after,same_branches)-weighted_loss(before,ca),
     'RGB_residual_sign_changes':int(np.sum(((np.sign(ar)!=np.sign(br))&valid[:,None]))),
     'actual_RGB_loss_change':actual_rgb_change,'fixed_residual_sign_RGB_change':signed_rgb_change,
     'membership_changes':order_changes[tile]['membership_changes'],'common_order_changed':order_changes[tile]['common_order_changed']})
  summary.append({'axis':axis,'whole_frame_loss_change':float(sum(t[2] for t in tiles)),
    'selected_tile_loss_change':float(sum(t[2] for t in selected)),'selected_absolute_tile_change_fraction':float(sum(t[0] for t in selected)/max(sum(t[0] for t in tiles),1e-30)),
    'tiles_with_order_changes':sum(r['common_order_changed'] for r in order_changes),
    'tiles_with_membership_changes':sum(r['membership_changes']>0 for r in order_changes),'selected':details,'all_tiles':order_changes})
 result={'status':'DIAGNOSTIC_COMPLETED','axes':summary,'reconstruction_gate':'RGB max<=1e-4 and depth max<=1e-3 on selected tiles. Failed reconstruction prohibits mechanism attribution.',
         'limits':'Top8 tiles are selected by absolute tile loss change; full-frame contribution fraction reported. Counterfactuals diagnose sorting/membership and contributor branches, not quality or adoption.'}
 (F/'receipts/branch_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print('BRANCH_COMPARISON_WRITTEN')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--selfcheck',action='store_true');args=a.parse_args()
 if args.selfcheck:selfcheck()
 else:run()
