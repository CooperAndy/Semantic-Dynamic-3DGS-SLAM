"""Metadata-only choice of two new windows; no semantic pixels read here."""
import io,json,zipfile,hashlib,re
from pathlib import Path
import numpy as np,requests
from http_range_zip import RangeZip
ROOT=Path('/root/autodl-tmp/stage3_chy_20261006');D=ROOT/'inputs/kitti360_isolated_v1'
BASE='https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/'
URLS={'calibration':BASE+'384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip','poses':BASE+'89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip','labels':BASE+'ed180d24c0a144f2f1ac71c2c655a3e986517ed8/data_2d_semantics.zip'}
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 D.mkdir(parents=True,exist_ok=True);locks={};poses={}
 for kind in ['calibration','poses']:
  p=D/(kind+'.zip')
  if not p.exists():
   r=requests.get(URLS[kind],stream=True,timeout=60)
   try:r.raise_for_status();b=r.raw.read(16*1024*1024+1);assert len(b)<=16*1024*1024
   finally:r.close()
   p.write_bytes(b)
  locks[kind]={'url':URLS[kind],'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  with zipfile.ZipFile(p) as z:
   for n in z.namelist():
    if kind=='calibration' and n.endswith('.txt') and not n.endswith('/'): 
     target=D/'calibration'/Path(n).name;target.parent.mkdir(exist_ok=True);target.write_bytes(z.read(n))
    if kind=='poses' and n.endswith('cam0_to_world.txt'):
     seq=Path(n).parent.name;rows=np.loadtxt(io.BytesIO(z.read(n)));poses[seq]=set(rows[:,0].astype(int).tolist())
 labels=RangeZip(URLS['labels'],128*1024*1024);chosen=[]
 for seq in ['2013_05_28_drive_0000_sync','2013_05_28_drive_0002_sync']:
  suffix=f'/{seq}/image_00/semantic/';available={int(Path(n).stem):n for n in labels.info if suffix in n and n.endswith('.png')}
  candidates=sorted(set(available)&poses[seq]);assert candidates,'no label/pose overlap'
  # Earliest 26 pose+annotation frames; deterministic file metadata only.
  frames=candidates[:26];assert len(frames)==26
  images={};image_locks={}
  for cam in ['00','01']:
   url=BASE+f'data_2d_raw/{seq}_image_{cam}.zip';z=RangeZip(url,32*1024*1024)
   images[cam]={f:next(n for n in z.info if n.endswith(f'image_{cam}/data_rect/{f:010d}.png')) for f in frames};image_locks[cam]=z.provenance()
  chosen.append({'sequence':seq,'frames':frames,'fusion_local':[i for i in range(26) if i%5!=4],'evaluation_local':[i for i in range(26) if i%5==4],'semantic_members':{str(f):available[f] for f in frames},'instance_members':{str(f):available[f].replace('/semantic/','/instance/') for f in frames},'image_members':{k:{str(f):n for f,n in v.items()} for k,v in images.items()},'image_archives':image_locks})
 dump(D/'metadata_selection.json',{'status':'METADATA_ONLY_NO_LABEL_PIXELS','selection':'earliest26 sorted frame IDs with cam0 pose + semantic member + both rectified RGB members in sequence0000 and0002','sequences':chosen,'labels_archive':labels.provenance(),'small_sources':locks,'URLS':URLS,'official_download_page':'https://www.cvlibs.net/datasets/kitti-360/download.php','GT_pixels_decoded':False,'stage2_test_reused':False})
 print(json.dumps({'status':'METADATA_SELECTION_READY','sequences':[{k:s[k] for k in ['sequence','frames']} for s in chosen],'labels_metadata':labels.provenance()}),flush=True)
if __name__=='__main__':main()
