from pathlib import Path
from PIL import Image
import csv, hashlib, json, re
root=Path('/root/autodl-tmp/seq21_chy_20260926')
seq=root/'sequences/21_1200_1399'
orig=Path('/root/autodl-tmp/slam_project/data/kitti/dataset/sequences/21')
lines=(orig/'times.txt').read_text().splitlines()
assert len(lines)>1399
for name,target in [('image_0','image_2'),('image_1','image_3')]:
 p=seq/name
 if not p.exists(): p.symlink_to(seq/target, target_is_directory=True)
(seq/'times.txt').write_text('\n'.join(lines[1200:1400])+'\n')
rows=[]
for i in range(200):
 row={'source_frame':1200+i,'local_index':i,'timestamp':lines[1200+i]}
 for cam in (2,3):
  p=seq/f'image_{cam}/{i:06d}.png'
  assert p.is_file()
  with Image.open(p) as im: im.verify()
  row[f'image_{cam}_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 rows.append(row)
with (root/'input_manifest.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
cal=(seq/'calib.txt').read_text()
P={}
for n in ('P2','P3'):
 vals=list(map(float,re.search(r'^'+n+r':\s+(.+)$',cal,re.M).group(1).split()))
 P[n]=vals
fx,fy,cx,cy=P['P2'][0],P['P2'][5],P['P2'][2],P['P2'][6]
b=abs(P['P2'][3]/fx-P['P3'][3]/P['P3'][0])
assert b>0.1
stock=list(Path('/root/autodl-tmp/slam_project/work_chy_20260924').rglob('Examples/Stereo/KITTI00-02.yaml'))[0]
s=stock.read_text()
for key,val in {'Camera1.fx':fx,'Camera1.fy':fy,'Camera1.cx':cx,'Camera1.cy':cy,'Stereo.b':b,'Camera.width':1241,'Camera.height':376}.items():
 s=re.sub(r'^'+re.escape(key)+r':.*$',f'{key}: {val}',s,flags=re.M)
(root/'tracking').mkdir(exist_ok=True)
(root/'tracking/seq21_color.yaml').write_text(s)
(root/'input_manifest.json').write_text(json.dumps({'source_frames':[1200,1399],'count':200,'left':'image_2','right':'image_3','calib_sha256':hashlib.sha256((seq/'calib.txt').read_bytes()).hexdigest(),'fx':fx,'baseline_m':b,'image_shape':[376,1241],'times':200},indent=2))
print('INPUT_PASS',len(rows),'fx',fx,'baseline',b,'times',len(lines))
