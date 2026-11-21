"""Audit all200 pairs and lock sparse GT without densifying labels."""
import csv,hashlib,json
from pathlib import Path
import numpy as np,yaml
from semantickitti_projection import calibration,project
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/semantickitti08_v1';B=R/'02_基础因子DEV与Oracle诊断/semantickitti08_gt_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 a=json.loads((R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1/correspondence_audit.json').read_text());assert a['status']=='PASS_SMALL_CORRESPONDENCE_AUDIT' and a['visual_review_passed']
 frozen=R/'protocol/SemanticKITTI08稀疏GT评价预声明_20261007_v1.json';protocol=json.loads(frozen.read_text());assert protocol['base_sha256']==sha(R/'inputs/base_map/params.npz');download=json.loads((D/'full_download_manifest.json').read_text());assert download['frames']==list(range(3000,3200));P,T=calibration(D/'calib.txt');(D/'sparse_gt').mkdir(exist_ok=True);records=[];inventory=list(csv.DictReader((R/'inputs/input_inventory_20261007.csv').open()));known=yaml.safe_load((R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1/official_semantic_kitti.yaml').read_text())['labels']
 with np.load(R/'inputs/base_map/params.npz') as b:h,w=int(b['org_height']),int(b['org_width'])
 for f in download['frames']:
  p=D/'velodyne'/f'{f:06d}.bin';l=D/'labels'/f'{f:06d}.label';assert p.stat().st_size%16==0 and l.stat().st_size%4==0 and p.stat().st_size//16==l.stat().st_size//4;pts=np.fromfile(p,dtype='<f4').reshape(-1,4);labels=np.fromfile(l,dtype='<u4');assert set(np.unique(labels&0xffff).tolist())<=set(known);gt,stats,_=project(pts,labels,P,T,h,w);assert stats['mapped_GT_pixels']>500
  inputlocks={}
  for kind in ['image_2','labels','depth_sgbm_candidate']:
   r=next(x for x in inventory if x['kind']==kind and int(x['original_frame'])==f);assert sha(r['stage3_reference'])==r['sha256'];inputlocks[kind]={'path':r['stage3_reference'],'sha256':r['sha256']}
  target=D/'sparse_gt'/f'{f:06d}.npz';np.savez_compressed(target,**gt);records.append({'original_frame':f,'local_frame':f-3000,'point_file':{'path':str(p),'sha256':sha(p)},'label_file':{'path':str(l),'sha256':sha(l)},'GT_file':{'path':str(target),'sha256':sha(target)},'input_locks':inputlocks,**stats});print(json.dumps({'phase':'prepare_GT','frame':f}),flush=True)
 dump(D/'evaluation_input_manifest.json',{'status':'ALL200_AUDITED_SPARSE_GT_READY','records':records,'image_height':h,'image_width':w,'protocol_sha256':sha(frozen),'calibration_sha256':sha(D/'calib.txt'),'base_sha256':protocol['base_sha256'],'state_sha256':protocol['state_sha256'],'exposure':'HISTORICALLY_EXPOSED_DEV','GT':'independent sparse projectedSemanticKITTI; not dense2D'})
if __name__=='__main__':main()
