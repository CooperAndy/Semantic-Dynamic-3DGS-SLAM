from common import *
import numpy as np,cv2,importlib.util,time,shutil,csv
cv2.setNumThreads(2)
spec=importlib.util.spec_from_file_location('legacy',Path(__file__).resolve().parent/'provenance/legacy_confidence.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
def compute(left,right,dl,dr,pose,previous_depth,previous_pose,calib):
 valid=(dl>=.5)&(dl<=40.);lr,_,_,_=legacy.confidence_from_stereo(left,right,dl,dr,calib['fx']*calib['baseline_m'],valid,1.,.1)
 if previous_depth is None:tc=np.ones_like(dl,dtype=np.float32)
 else:tc,_,_=legacy.temporal_confidence(dl,previous_depth,pose,previous_pose,calib,valid,.1)
 confidence=(lr*tc).astype(np.float32);confidence[~valid]=0;seed=confidence>=.8;loss=confidence>=.2;hard=dl.copy();hard[~seed]=0
 return dict(confidence=confidence,seed_mask=seed,loss_mask=loss,depth_confhard=hard)
def main():
 start=time.monotonic();reg=json.loads((R/'registration.json').read_text());assert not (R/'cache').exists();(R/'cache').mkdir()
 for name in ['confidence','seed_mask','loss_mask','depth_confhard']:(R/'cache'/name).mkdir()
 previous_depth=previous_pose=None
 with (R/'seals.log').open('x',buffering=1) as seal,(R/'per_frame.jsonl').open('x',buffering=1) as ledger:
  for i,row in enumerate(reg['rows']):
   assert i==row['index'] and time.monotonic()-start<1200 and shutil.disk_usage(R).free>10*2**30
   paths={k:Path(x['path']) for k,x in row['inputs'].items()}
   left=cv2.cvtColor(cv2.imread(str(paths['left'])),cv2.COLOR_BGR2RGB);right=cv2.cvtColor(cv2.imread(str(paths['right'])),cv2.COLOR_BGR2RGB)
   dl=legacy.load_depth(paths['depth_left'],tuple(reg['shape']));dr=legacy.load_depth(paths['depth_right'],tuple(reg['shape']));pose=np.load(paths['pose'],allow_pickle=False)
   assert left.shape==right.shape==tuple(reg['shape'])+(3,) and pose.shape==(4,4) and pose.dtype==np.float64 and np.isfinite(pose).all()
   outputs=compute(left,right,dl,dr,pose,previous_depth,previous_pose,reg['calibration']);saved={}
   for name,array in outputs.items():
    suffix=dict(confidence='confidence',seed_mask='seed_valid',loss_mask='loss_valid',depth_confhard='depth')[name];path=R/'cache'/name/f'{i:06d}_{suffix}.npy';np.save(path,array,allow_pickle=False);saved[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
   ledger.write(json.dumps(dict(index=i,previous_index=i-1 if i else None,max_input_index=i,mean_confidence=float(outputs['confidence'].mean()),seed_ratio=float(outputs['seed_mask'].mean()),outputs=saved))+'\n');seal.write(f'sealed {i}\n');previous_depth=dl;previous_pose=pose
   if i==0 or (i+1)%100==0:print('SEALED',i+1,flush=True)
 dump(R/'producer_receipt.json',dict(status='COMPLETE_1000_PAST_ONLY_CACHE_PENDING_AUDIT',frames=1000,seconds=time.monotonic()-start,GT_runtime=False,quality_adoption=False))
if __name__=='__main__':main()
