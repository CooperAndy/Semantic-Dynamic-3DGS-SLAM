"""Read native forward buffers without modifying kernels, maps or the objective."""
from pathlib import Path
import hashlib,json,sys,time
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1]
P=json.loads((F/'trace_protocol.json').read_text())
sys.path.insert(0,P['T15_root']+'/scripts')
import replay_runtime as rt
import numpy as np,torch
import diff_gaussian_rasterization_depth_pose as raster
LOCK=json.loads((F/'trace_lock.json').read_text())
assert rt.sha(F/'trace_protocol.json')==LOCK['protocol_sha256']
assert rt.sha(Path(__file__))==LOCK['script_sha256']
def decode(buf,fields):
 assert buf.data_ptr()%128==0
 data=buf.detach().cpu().numpy();offset=0;out={}
 for name,dtype,shape in fields:
  offset=(offset+127)//128*128;count=int(np.prod(shape));dt=np.dtype(dtype)
  assert offset+count*dt.itemsize<=len(data)
  out[name]=np.frombuffer(data,dtype=dt,count=count,offset=offset).reshape(shape).copy();offset+=count*dt.itemsize
 return out
original=raster._C.rasterize_gaussians;capture={}
def observed(*args):
 global capture
 result=original(*args);n,color,depth,acc,radii,geom,binning,img=result
 G=args[1].shape[0];H,W=color.shape[-2:];tiles=((H+15)//16)*((W+15)//16)
 a=decode(geom,[('depths','float32',(G,)),('clamped','bool',(G,3)),('radii_internal','int32',(G,)),('means2D','float32',(G,2)),('cov3D','float32',(G,6)),('conic_opacity','float32',(G,4)),('rgb','float32',(G,3)),('tiles_touched','uint32',(G,))])
 b=decode(binning,[('point_list','uint32',(n,)),('point_list_unsorted','uint32',(n,)),('keys','uint64',(n,)),('keys_unsorted','uint64',(n,))])
 c=decode(img,[('final_T','float32',(H,W)),('last_contributor','uint32',(H,W)),('ranges','uint32',(H*W,2))]);c['ranges']=c['ranges'][:tiles]
 capture={**a,**b,**c,'native_radii':radii.detach().cpu().numpy().copy()}
 # The binding supplies external radii, so geom.internal_radii is allocated but
 # unused (rasterizer_impl.cu). Preserve its offset, discard its undefined bytes.
 del capture['radii_internal']
 assert (b['keys'][1:]>=b['keys'][:-1]).all()
 assert np.all(b['point_list']<G)
 used=b['point_list'];key_depth=(b['keys']&np.uint64(0xffffffff)).astype(np.uint32).view(np.float32)
 assert np.array_equal(key_depth,a['depths'][used])
 for tile,(start,end) in enumerate(c['ranges']):
  if end>start:assert np.all(b['keys'][start:end]>>np.uint64(32)==tile)
 return result
raster._C.rasterize_gaussians=observed
rows=[];started=time.time();out=F/'traces';out.mkdir(exist_ok=False)
for frame in P['frames']:
 rt.seed();ctx=rt.Context(frame);rng=rt.rnghash()
 with np.load(Path(P['T15_root'])/f'receipts/numeric_frame{frame}.npz') as z:
  assert np.array_equal(ctx.T0.cpu().numpy(),z['T0']) and np.array_equal(ctx.support.cpu().numpy(),z['support'])
 positions=P['frame47_positions'] if frame==47 else P['control_positions']
 for axis in P['axes']:
  for h in positions:
   if h==0 and axis!=P['axes'][0]:continue
   u=np.zeros(6);u[axis]=h
   with torch.no_grad():
    T=ctx.step(ctx.T0,u);q=ctx.render(T);value=float(ctx.loss(q))
    im=q['image'].cpu().numpy();dep=q['depth'].cpu().numpy();acc=q['acc'].cpu().numpy()
   name=f'f{frame}_a{axis}_h{h:+.5f}.npz'
   np.savez_compressed(out/name,**capture,T=T.cpu().numpy(),image=im,render_depth=dep,acc=acc,
     observed_color=ctx.color.cpu().numpy(),observed_depth=ctx.depth.cpu().numpy(),support=ctx.support.cpu().numpy())
   rows.append({'frame':frame,'axis':axis,'h':h,'loss':value,'path':'traces/'+name,'sha256':rt.sha(out/name)})
   print('CAPTURED',frame,axis,h,flush=True)
 ctx.finish();assert rt.rnghash()==rng
assert len(rows)<=P['maximum_captures']
rt.dump(F/'trace_result.json',{'status':'COMPLETED','captures':rows,'protocol_sha256':rt.sha(F/'trace_protocol.json'),'lock_sha256':rt.sha(F/'trace_lock.json'),'GT_audit':rt.GT_AUDIT,'map_unchanged':True,'rng_unchanged':True,'seconds':time.time()-started})
print('TRACE_COMPLETED',len(rows),flush=True)
