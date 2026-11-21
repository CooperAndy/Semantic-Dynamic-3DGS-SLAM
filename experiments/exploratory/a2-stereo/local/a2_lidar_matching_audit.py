import json,pathlib,hashlib,sys,importlib.util,numpy as np
from PIL import Image,ImageDraw
root=pathlib.Path(r'C:\Users\LOCAL_USER\Desktop\fyp');d=root/'阶段3/inputs/semantickitti08_v1';o=root/'10.26创新点/A2_双目噪声鲁棒几何优化/A2_1_诊断_20261028';a=json.loads((o/'B200_runtime_input_manifest.json').read_text(encoding='utf8'));m=json.loads((d/'evaluation_input_manifest.json').read_text(encoding='utf8')); rows={r['frame_id']:r for r in a['rows']};recs={r['original_frame']:r for r in m['records']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('proj',root/'阶段3/scripts/semantickitti_projection.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);P,T=mod.calibration(d/'calib.txt');K=np.array(a['K']);print('K',K.tolist(),'P2left',P[:,:3].tolist()); bad=[]
for i in range(200):
 f=i+3000;r=rows[i];rec=recs[f];img=root/'slam_project/data/kitti/sequences/08/image_2'/f'{f:06d}.png';v=d/'velodyne'/f'{f:06d}.bin';g=d/'sparse_gt'/f'{f:06d}.npz';
 checks={'source_frame':r['source_frame_id']==f and rec['local_frame']==i,'rgb_manifest':r['hashes']['rgb']==rec['input_locks']['image_2']['sha256'],'rgb_actual':img.exists() and sha(img)==r['hashes']['rgb'],'lidar':v.exists() and sha(v)==rec['point_file']['sha256'],'sparse':g.exists() and sha(g)==rec['GT_file']['sha256']}
 for k,ok in checks.items():
  if not ok:bad.append([i,k])
print('200 frame/hash failures:',bad)
frames=[]
for i in range(19,200,20):
 f=i+3000; points=np.fromfile(d/'velodyne'/f'{f:06d}.bin',dtype='<f4').reshape(-1,4);labels=np.fromfile(d/'labels'/f'{f:06d}.label',dtype='<u4');gt,stats,uv=mod.project(points,labels,P,T,370,1226)
 with np.load(d/'sparse_gt'/f'{f:06d}.npz') as z: same={k:bool(np.array_equal(gt[k],z[k])) for k in gt}; depth=z['depth'];n=int((depth>0).sum());bins=[int(((depth>=lo)&(depth<hi)).sum()) for lo,hi in [(0.5,10),(10,30),(30,80)]]
 frames.append({'internal_frame':i,'source_frame':f,'sparse_pixels':n,'bins_0p5_10_30_80':bins,'exact_reprojection':same,'lidar_points':len(points)})
 if i==99:
  im=Image.open(root/'slam_project/data/kitti/sequences/08/image_2'/f'{f:06d}.png').convert('RGB');draw=ImageDraw.Draw(im); yy,xx=np.nonzero(depth>0)
  for y,x in zip(yy[::3],xx[::3]):
   z=float(depth[y,x]);color=(255,40,30) if z<10 else ((250,220,20) if z<30 else (30,190,255));draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
  im.save(o/'lidar_projection_003099.png')
print('reprojection',frames);print('intrinsic max abs',np.max(np.abs(K[:3,:3]-P[:,:3])));print('P2 column',P[:,3].tolist())
res={'scope':'A2 B200 already-exposed DEV, not held-out test','runtime_manifest_sha256':sha(o/'B200_runtime_input_manifest.json'),'calibration_sha256':sha(d/'calib.txt'),'total_checked':200,'failures':bad,'K':K.tolist(),'P2':P.tolist(),'Tr':T.tolist(),'frames':frames,'caveat':'Frame and hash agreement plus projection replication do not independently validate time synchronization, occlusion, rolling scan motion or mapper camera conventions.'};(o/'lidar_matching_audit.json').write_text(json.dumps(res,indent=2),encoding='utf8')
