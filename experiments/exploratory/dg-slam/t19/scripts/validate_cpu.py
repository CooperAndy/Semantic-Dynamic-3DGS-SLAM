"""Execute candidate Python control flow on CPU with explicit SE3/network/native test doubles.
These tests cannot validate real neural updates or CUDA kernels.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast,contextlib,copy,hashlib,importlib.util,json,sys,types,warnings,tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm,logm
from scipy.spatial.transform import Rotation
warnings.filterwarnings('ignore', category=FutureWarning)
F=Path(__file__).resolve().parents[1]
B=Path(os.environ.get('T19_BASE_ROOT',str(F.parent/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012')))
ART=Path(tempfile.mkdtemp(prefix='cpu_cases_',dir=F/'receipts'))
if os.environ.get('T19_BASE_ROOT'):
    snapshot_manifest=json.loads((F/'receipts/base_snapshot_manifest.json').read_text(encoding='utf-8'))
    for entry in snapshot_manifest['files']:
        assert hashlib.sha256((B/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
sys.path.insert(0,str(F/'candidate'))
from dg_slam.filler_contract import batch_plan,ordered_indices,anchor_policy
from dg_slam.filler_trace import FillerTrace,TraceBudgetExceeded
checks=[]
def passed(name):checks.append(name)
def raises(kind,fn):
 try:fn()
 except kind:return
 raise AssertionError('Expected '+kind.__name__)

# Any attempt to initialize CUDA is an immediate test failure, even if a GPU is present.
def deny_cuda(*args,**kwargs):raise AssertionError('CUDA initialization is forbidden in CPU tests')
torch.cuda._lazy_init=deny_cuda
class CPUAutocast(contextlib.ContextDecorator):
 def __init__(self,*args,**kwargs):pass
 def __enter__(self):return self
 def __exit__(self,*args):return False
torch.cuda.amp.autocast=CPUAutocast
old_as_tensor=torch.as_tensor
def cpu_tensor(*args,**kwargs):
 if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
 return old_as_tensor(*args,**kwargs)
torch.as_tensor=cpu_tensor
torch.Tensor.cuda=lambda self,*args,**kwargs:self

class CPUSE3:
 def __init__(self,data):self.data=data.clone()
 def __getitem__(self,index):return CPUSE3(self.data[index])
 def matrices(self):
  x=self.data.detach().numpy();x=x.reshape(-1,7)
  T=np.tile(np.eye(4),(len(x),1,1));T[:,:3,:3]=Rotation.from_quat(x[:,3:]).as_matrix();T[:,:3,3]=x[:,:3];return T
 @classmethod
 def from_matrices(cls,T):
  x=np.concatenate([T[:,:3,3],Rotation.from_matrix(T[:,:3,:3]).as_quat()],axis=1)
  return cls(torch.tensor(x,dtype=torch.float32))
 def inv(self):return self.from_matrices(np.linalg.inv(self.matrices()))
 def __mul__(self,other):return self.from_matrices(self.matrices()@other.matrices())
 def log(self):
  result=[]
  for T in self.matrices():
   L=np.real(logm(T));result.append(np.r_[L[:3,3],[L[2,1],L[0,2],L[1,0]]])
  return torch.tensor(np.asarray(result),dtype=torch.float32)
 @classmethod
 def exp(cls,twist):
  result=[]
  for v in twist.detach().numpy():
   x,y,z=v[3:];L=np.zeros((4,4));L[:3,3]=v[:3];L[:3,:3]=[[0,-z,y],[z,0,-x],[-y,x,0]];result.append(expm(L))
  return cls.from_matrices(np.asarray(result))

class NativeSpy:
 def __init__(self):self.calls=[]
 def ba(self,poses,disps,intrinsics,sens,target,weight,eta,ii,jj,t0,t1,itrs,lm,ep,motion_only):
  self.calls.append({'target':target.clone(),'weight':weight.clone(),'ii':ii.clone(),'jj':jj.clone(),'parameters':(t0,t1,itrs,lm,ep,motion_only)})
  poses[t0:t1,0] += .001
backend=NativeSpy()
def method(path,classname,name):
 tree=ast.parse(path.read_text(encoding='utf-8'))
 node=next(x for c in tree.body if isinstance(c,ast.ClassDef) and c.name==classname for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name)
 node=copy.deepcopy(node);node.decorator_list=[]
 env={'torch':torch,'droid_backends':backend,'SE3':CPUSE3}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env)
 return env[name]
new_ba=method(F/'candidate/dg_slam/depth_video.py','DepthVideo','ba')
old_ba=method(B/'source/dg_slam/depth_video.py','DepthVideo','ba')
new_update=method(F/'candidate/dg_slam/factor_graph.py','FactorGraph','update')
old_update=method(B/'source/dg_slam/factor_graph.py','FactorGraph','update')

class Video:
 def __init__(self,keys=(0,10,14,18,24,27,31,35,40,47,56,82,95,103,116,130,137,143,153,157,172,181,194,198)):
  n=len(keys);self.counter=types.SimpleNamespace(value=n);self.ht=self.wd=16
  self.tstamp=torch.zeros(232);self.tstamp[:n]=torch.tensor(keys,dtype=torch.float32)
  self.poses=torch.zeros(232,7);self.poses[:,6]=1;self.poses[:n,0]=torch.tensor(keys)*.001
  self.disps=torch.ones(232,2,2);self.disps_sens=self.disps.clone()
  self.fmaps=torch.zeros(232,1,128,2,2);self.intrinsics=torch.ones(232,4)
  self.seg_masks=torch.zeros(232,2,2,dtype=torch.bool);self.seg_masks[0,0,0]=True
  self.use_segmask=True;self.use_depth_warp=True;self.ba=types.MethodType(new_ba,self)
 def get_lock(self):return contextlib.nullcontext()
 def __setitem__(self,index,item):
  self.tstamp[index]=item[0];self.poses[index]=item[2];self.fmaps[index]=item[6];self.intrinsics[index]=item[5]
  self.disps[index]=1/item[3][:,3::8,3::8];self.disps_sens[index]=self.disps[index];self.seg_masks[index]=item[9][:,3::8,3::8].bool()
 def reproject(self,ii,jj):
  out=torch.zeros(1,len(ii),2,2,2);out[0,:,:,:,0]=self.poses[jj,0,None,None]
  return out,torch.ones(1,len(ii),2,2,1,dtype=torch.bool)

class MockGraph:
 fail_at=None
 instances=[]
 def __init__(self,video,update):self.video=video;self.edges=[];self.calls=0;self.instances.append(self)
 def add_factors(self,ii,jj):
  for edge in zip(ii.tolist(),jj.tolist()):
   if edge not in self.edges:self.edges.append(edge)
 def update(self,t0,t1,motion_only,diagnostic_hook=None):
  self.calls+=1
  if self.fail_at==self.calls:raise RuntimeError('Injected optimizer failure')
  assert motion_only
  self.video.poses[t0:t1,0]+=.001
  if diagnostic_hook:diagnostic_hook('mock_step',poses_w2c=self.video.poses[t0:t1])

lie=types.ModuleType('lietorch');lie.SE3=CPUSE3;lie.cat=lambda poses,axis:CPUSE3(torch.cat([p.data for p in poses],axis))
sys.modules['lietorch']=lie
fg=types.ModuleType('factor_graph');fg.FactorGraph=MockGraph;sys.modules['factor_graph']=fg
dn=types.ModuleType('droid_net');dn.DroidNet=object;sys.modules['droid_net']=dn
geom=types.ModuleType('geom');geom.__path__=[];sys.modules['geom']=geom;sys.modules['geom.projective_ops']=types.ModuleType('geom.projective_ops')
spec=importlib.util.spec_from_file_location('cpu_candidate_filler',F/'candidate/dg_slam/trajectory_filler.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Filler=module.PoseTrajectoryFiller
net=types.SimpleNamespace(cnet=None,update=None,fnet=lambda images:torch.zeros(len(images),1,128,2,2))
def stream(frames):
 return [(int(i),torch.zeros(1,3,16,16),torch.ones(16,16),torch.ones(4),None,torch.zeros(16,16,dtype=torch.bool)) for i in frames]

slots,pos=batch_plan([0,1,2,4,5],[0,2,4]);assert pos==[1,4] and slots==[('anchor',0),('temporary',0),('anchor',1),('anchor',2),('temporary',1)]
passed('planner exact identity, interleaving and post-last-anchor tail')
for values in [[],[0,0],[2,1],[-1],[float('nan')],[float('inf')],[.5],[2**24+1],[True]]:
 if values:raises(ValueError,lambda v=values:ordered_indices(v))
raises(ValueError,lambda:batch_plan([0],[]));raises(ValueError,lambda:batch_plan([0],[1]))
passed('invalid, duplicate, unsorted, fractional and uncovered frame indices rejected')

# Execute actual __call__, spy only the private optimizer, checking runtime routing on real saved IDs.
for seed in [1729,2718,31415]:
 for arm in ['NO_GS_REFINE','LOCAL_ID']:
  raw=np.load(B/f'runs/DEV200_s{seed}_{arm}/keyframe_poses.npz');keys=raw['indices'].astype(int)
  video=Video(keys);f=Filler(net,video,device='cpu');calls=[]
  def spy(self,ts,*unused):
   calls.append(list(ts));data=torch.zeros(len(ts),7);data[:,6]=1;data[:,0]=1000+torch.tensor(ts);return CPUSE3(data)
  f._PoseTrajectoryFiller__fill=types.MethodType(spy,f)
  result=f(stream(range(200))).data
  assert result.shape==(200,7) and torch.equal(result[keys],video.poses[:len(keys)])
  nonkeys=np.setdiff1d(np.arange(200),keys);assert np.array_equal(result[nonkeys,0].numpy(),1000+nonkeys)
  assert sum(map(len,calls))==176 and all(not set(c)&set(keys) for c in calls)
  expected=[[i for i in range(s,min(s+16,200)) if i not in keys] for s in range(0,200,16)]
  assert calls==[c for c in expected if c]
passed('actual routing on all six saved runs: 24 exact anchors + 176 nonKF, original input batch boundaries')

# Random and corner cases stress reassembly rather than merely repeat the planner implementation.
rng=np.random.default_rng(2718)
for trial in range(40):
 count=int(rng.integers(1,80));keys=np.sort(rng.choice(np.arange(count),size=int(rng.integers(1,count+1)),replace=False));keys=np.unique(np.r_[0,keys])
 video=Video(keys);f=Filler(net,video,device='cpu');seen=[]
 def spy2(self,ts,*unused):
  seen.extend(ts);x=torch.zeros(len(ts),7);x[:,6]=1;x[:,0]=1000+torch.tensor(ts);return CPUSE3(x)
 f._PoseTrajectoryFiller__fill=types.MethodType(spy2,f);out=f(stream(range(count))).data
 assert len(out)==count and torch.equal(out[keys],video.poses[:len(keys)])
 assert seen==[i for i in range(count) if i not in keys]
passed('40 randomized routing windows, all-anchor batches, short tails, one-anchor window')
for frames in [[],[0,0],[1,0],[.5]]:
 f=Filler(net,Video(),device='cpu')
 if frames==[.5]:raises(ValueError,lambda:f([(0.5,*stream([0])[0][1:])]))
 else:raises(ValueError,lambda:f(stream(frames)))
passed('actual stream rejects empty, repeated, reversed or fractional input indices')

# Execute actual __fill__ with CPU SE3 and graph doubles, including exception cleanup.
video=Video();anchors=video.poses[:24].clone();f=Filler(net,video,device='cpu')
out=f(stream(range(200))).data
assert torch.equal(out[video.tstamp[:24].long()],anchors) and torch.equal(video.poses[:24],anchors) and video.counter.value==24
assert all(g.calls==6 for g in MockGraph.instances)
passed('actual fill control flow: six mocked updates, anchors immutable, counter restored, 200-frame output')
MockGraph.fail_at=3;video=Video();sentinel=lambda *args,**kw:None;video.ba_diagnostic_hook=sentinel;f=Filler(net,video,device='cpu')
raises(RuntimeError,lambda:f(stream(range(16))))
assert video.counter.value==24 and video.ba_diagnostic_hook is sentinel
MockGraph.fail_at=None
passed('optimizer failure restores temporary count and previous diagnostic hook')
video=Video();f=Filler(net,video,device='cpu');video.poses=video.poses[:25]
raises(ValueError,lambda:f(stream(range(16))));assert video.counter.value==24
passed('temporary buffer overflow rejected without counter mutation')

# Compare original and patched Python factor update + BA with identical deterministic CPU backend.
def make_graph(ba):
 v=Video([0,1]);v.ba=types.MethodType(ba,v);v.poses[2:4,0]=torch.tensor([.02,.03])
 g=types.SimpleNamespace(video=v,coords0=torch.zeros(2,2,2),ii=torch.tensor([0,1]),jj=torch.tensor([2,3]),
                         net=torch.zeros(1),inp=torch.zeros(1),damping=torch.zeros(232,2,2),upsample=False,age=torch.zeros(2,dtype=torch.long))
 g.target=v.reproject(g.ii,g.jj)[0];g.corr=lambda c:c
 g.update_op=lambda net,inp,corr,motn,ii,jj:(net,torch.ones_like(corr)*.002,torch.ones_like(corr)*.75,torch.ones(2,2,2)*.01,None)
 return g
base=make_graph(old_ba);backend.calls=[];old_update(base,2,4,motion_only=True);base_calls=backend.calls
traced=make_graph(new_ba);backend.calls=[];events=[]
hook=lambda stage,**values:events.append((stage,{k:(v.detach().numpy().copy() if isinstance(v,torch.Tensor) else np.array(v)) for k,v in values.items()}))
traced.video.ba_diagnostic_hook=hook;new_update(traced,2,4,motion_only=True,diagnostic_hook=hook)
assert torch.equal(base.video.poses,traced.video.poses) and torch.equal(base.video.disps,traced.video.disps)
assert len(base_calls)==len(backend.calls)==1
for key in base_calls[0]:
 a,b=base_calls[0][key],backend.calls[0][key];assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
assert [e[0] for e in events]==['graph_before_ba','native_before_ba','native_after_ba','graph_after_ba']
before=dict(events)['native_before_ba'];assert before['effective_weight'][0,:,0,0].sum()==0
assert np.array_equal(before['effective_weight'],backend.calls[0]['weight'].numpy())
after=dict(events)['graph_after_ba'];assert np.allclose(after['residual_after'],.001*np.array([1,2])[None,None,None,:],atol=1e-8)
passed('original-vs-patched update and BA identical native inputs/outputs under CPU backend double')
passed('trace captures effective mask-applied weights and post-BA reprojection residual')

trace=FillerTrace(ART/'test_trace');trace.begin([153,154]);trace.record('copy',x=torch.tensor([1.,2.]))
saved=np.load(next(trace.root.glob('*.npz')))['x'];assert np.array_equal(saved,[1,2])
tiny=FillerTrace(ART/'test_trace_budget',max_bytes=1);tiny.begin([153]);raises(TraceBudgetExceeded,lambda:tiny.record('oversize',x=np.zeros(2)))
assert not list(tiny.root.glob('*.npz'))
passed('trace serialization copies CPU arrays; budget failure is explicit, no partial npz')
video=Video();f=Filler(net,video,device='cpu');f.diagnostic_trace=FillerTrace(ART/'actual_fill_trace')
anchors=video.poses[:24].clone();result=f(stream([153,154,155])).data
events_actual=[json.loads(x) for x in (f.diagnostic_trace.root/'events.jsonl').read_text().splitlines()]
assert len(events_actual)==7 and events_actual[0]['stage']=='initialization'
assert np.array_equal(np.load(f.diagnostic_trace.root/events_actual[0]['file'])['temporary_frames'],[154,155])
assert torch.equal(result[0],anchors[18]) and torch.equal(video.poses[:24],anchors)
assert video.counter.value==24 and video.ba_diagnostic_hook is None
passed('actual fill trace records nonKF initialization and all six mock steps; fixed KF excluded')
video=Video();f=Filler(net,video,device='cpu');f.preserve_keyframes=False;anchors=video.poses[:24].clone()
legacy=f(stream([153,154])).data
assert not torch.equal(legacy[0],anchors[18]) and torch.equal(video.poses[:24],anchors)
passed('explicit legacy mode reproduces movable KF duplicates with fixed stored anchors in CPU fixture')
video=Video();f=Filler(net,video,device='cpu');f.diagnostic_trace=tiny
raises(TraceBudgetExceeded,lambda:f(stream([153,154])));assert video.counter.value==24 and video.ba_diagnostic_hook is None
passed('trace failure during actual fill restores count and callback, aborts without fallback')
reader_spec=importlib.util.spec_from_file_location('cpu_trace_reader',F/'scripts/analyze_trace.py')
reader=importlib.util.module_from_spec(reader_spec);reader_spec.loader.exec_module(reader)
native_trace=FillerTrace(ART/'native_trace_fixture',frames=[2,3]);native_trace.begin([2,3]);native_trace.iteration=0
graph=make_graph(new_ba);graph.video.ba_diagnostic_hook=native_trace.record
new_update(graph,2,4,motion_only=True,diagnostic_hook=native_trace.record)
inspection=reader.inspect(native_trace.root)
assert len(inspection)==1 and abs(inspection[0]['max_c2w_step_cm']-.1)<1e-5
assert inspection[0]['fixed_learned_target_weighted_residual_after'] < inspection[0]['fixed_learned_target_weighted_residual_before']
raises(ValueError,lambda:reader.inspect(ART/'actual_fill_trace'))
passed('CPU trace reader computes c2w steps/residuals and rejects incomplete mocked-only traces')

from dg_slam.filler_adapter import configure_filler
for arm,policy,allow in [('LOCAL_ID','native_final_DROID',False),('LOCAL_ID','mapping_snapshot',False),('NO_GS_REFINE','mapping_snapshot',False)]:
 raises(ValueError,lambda:anchor_policy(arm,policy,allow))
passed('ID refinement cannot be silently discarded or injected as supposedly coherent geometry')
v=Video([0,10]);model=types.SimpleNamespace(video=v,traj_filler=Filler(net,v,device='cpu'));before=v.poses.clone();mapped=np.tile(np.eye(4),(2,1,1));mapped[1,0,3]=.123
r=configure_filler(model,mapped,'NO_GS_REFINE','native_final_DROID',ART/'test_adapter_native')
assert torch.equal(v.poses,before) and not r['mixed_geometry'] and model.traj_filler.preserve_keyframes
v=Video([0,10]);model=types.SimpleNamespace(video=v,traj_filler=Filler(net,v,device='cpu'))
r=configure_filler(model,mapped,'LOCAL_ID','mapping_snapshot',ART/'test_adapter_mixed',mixed_geometry_diagnostic=True)
assert r['mixed_geometry'] and r['diagnostic_only'] and abs(float(v.poses[1,0])+.123)<1e-6
v=Video([0,10]);model=types.SimpleNamespace(video=v,traj_filler=Filler(net,v,device='cpu'))
r=configure_filler(model,mapped,'LOCAL_ID','mapping_snapshot',ART/'test_adapter_legacy',mixed_geometry_diagnostic=True,preserve_keyframes=False)
assert not r['anchor_preservation'] and not r['preserve_keyframes']
passed('adapter preserves native state or explicitly labels diagnostic mapped injection')
assert not torch.cuda.is_initialized()
passed('CUDA lazy initialization blocked; no real neural network or native extension loaded')
receipt={'status':'PASS','checks':checks,'check_groups':len(checks),'actual_saved_run_routing_cases':6,'randomized_routing_cases':40,'CPU_only':True,
         'test_artifacts':str(ART.relative_to(F)),'input_root':str(B),'torch_version':torch.__version__,'CUDA_initialized':torch.cuda.is_initialized(),'real_CUDA_kernels_executed':0,
         'real_neural_updates_executed':0,'remote_connections':0,'limits':'CPUSE3, graph, correlation and native BA doubles; Python routing/cleanup and trace invariance only. No quality or GPU runtime acceptance.'}
(F/'receipts/CPU_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
