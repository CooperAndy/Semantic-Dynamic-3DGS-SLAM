"""Common pinned runtime and explicit GT boundary."""
from pathlib import Path
import hashlib,json,sys,os
F=Path(__file__).resolve().parents[1];R=F.parent;DATA=R/'02_data_masks/TUM/rgbd_dataset_freiburg3_walking_xyz'
BASE=R/'07_formal_tracking_T14_20261012'
P=json.loads((BASE/'T14_formal_tracking_protocol.json').read_text());P.update(json.loads((F/'protocol.json').read_text()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def bootstrap():
 source=F/'source';sys.path[:0]=[str(BASE/'native/faiss_python'),str(BASE/'native/lietorch'),str(BASE/'native/droid'),str(BASE/'native/raster'),str(source),str(source/'dg_slam')]
 os.chdir(source);os.environ['WANDB_MODE']='disabled'
 import torch,lietorch,droid_backends,lietorch_backends,lietorch_extras,faiss
 import diff_gaussian_rasterization_depth_pose._C as raster
 assert tuple(torch.cuda.get_device_capability())==(12,0)
 import importlib
 faiss_native=importlib.import_module('faiss._swigfaiss')
 assert faiss.__version__=='1.10.0' and Path(faiss.__file__).resolve().is_relative_to((BASE/'native/faiss_python').resolve())
 mods=[droid_backends,lietorch_backends,lietorch_extras,raster,faiss_native]
 for m in mods:assert Path(m.__file__).resolve().is_relative_to((BASE/'native').resolve()),m.__file__
 return {m.__name__:{'path':m.__file__,'sha256':sha(m.__file__)} for m in mods}
def guard_gt():
 audit={'blocked_GT_opens':0}
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   name=Path(os.fsdecode(args[0])).name.lower()
   if name=='groundtruth.txt' or name.endswith('_evaluation.csv') or 'evaluation_reference' in name or 'evaluation_inputs' in str(args[0]):
    audit['blocked_GT_opens']+=1;raise PermissionError('No GT values or evaluation associations in runtime')
 sys.addaudithook(guard);return audit
