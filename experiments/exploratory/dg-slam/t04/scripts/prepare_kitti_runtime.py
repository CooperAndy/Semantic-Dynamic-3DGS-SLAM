"""Generate same-frame stereo pseudo-depth; no GT, no temporal neighbours."""
from pathlib import Path
import hashlib,json,tarfile,time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FYP=ROOT.parents[1]
DATA=FYP/'slam_project/data/kitti/sequences/00'
OUT=ROOT/'03_数据索引/runtime200'
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4)
K=np.array([718.856,718.856,607.1928,185.2157])*np.array([640/1241,192/376,640/1241,192/376])
sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,preFilterCap=31,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
times=np.loadtxt(DATA/'times.txt')[:200];rows=[];audit=[]
start=time.time()
for i,t in enumerate(times):
    left=DATA/'image_2'/f'{i:06d}.png';right=DATA/'image_3'/f'{i:06d}.png'
    l=cv2.imdecode(np.frombuffer(left.read_bytes(),np.uint8),cv2.IMREAD_COLOR);r=cv2.imdecode(np.frombuffer(right.read_bytes(),np.uint8),cv2.IMREAD_COLOR);assert l.shape==r.shape==(376,1241,3)
    l=cv2.resize(l,(640,192),interpolation=cv2.INTER_AREA);r=cv2.resize(r,(640,192),interpolation=cv2.INTER_AREA)
    d=sgbm.compute(cv2.cvtColor(l,cv2.COLOR_BGR2GRAY),cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16
    z=np.zeros(d.shape,np.float32);valid=d>0;z[valid]=float(K[0]*0.532331857841)/d[valid];z[(z<.5)|(z>80)|~np.isfinite(z)]=0
    assert np.count_nonzero(z)>1000
    ok,encoded=cv2.imencode('.png',l);assert ok;(OUT/f'{i:06d}.png').write_bytes(encoded.tobytes());np.save(OUT/f'{i:06d}_depth.npy',z)
    rows.append({'frame_id':i,'rgb_timestamp':float(t),'rgb_path':f'{i:06d}.png','depth_path':f'{i:06d}_depth.npy'})
    old=DATA/'depth_sgbm'/f'{i:06d}_depth.npy'
    if old.exists():
        v=np.load(old,allow_pickle=False);positive=v[np.isfinite(v)&(v>0)]
        audit.append({'frame':i,'shape':list(v.shape),'dtype':str(v.dtype),'valid_fraction':float(len(positive)/v.size),'median_positive':float(np.median(positive)) if len(positive) else None,'role':'legacy presence/statistics only; not used'})
    if i%50==0:print('prepared',i,flush=True)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
protocol={'variant':'KITTI_STEREO_DERIVED_RGBD_NO_SEMANTIC_ENGINEERING','frames':200,'image_size':[192,640],'K':K.tolist(),'depth_units':'metres','depth_source':'fresh OpenCV SGBM on same-frame resized P2/P3, not sensor GT','depth_valid_range_m':[.5,80],'gaussian_support_max_depth_m':40,'semantic_mask':'explicit all-zero exclusion, no semantic prediction; not original OneFormer method','sgbm':{'minDisparity':0,'numDisparities':128,'blockSize':5,'P1':200,'P2':800,'disp12MaxDiff':1,'uniquenessRatio':10,'speckleWindowSize':100,'speckleRange':2,'preFilterCap':31,'mode':'3WAY'},'input_hashes':manifest,'rows':rows,'opencv':cv2.__version__,'preparation_seconds':time.time()-start,'gt_used':False,'orb_poses_used':False,'future_frames_used':False}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
(ROOT/'03_数据索引/旧深度缓存统计_本轮不使用.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
with tarfile.open(ROOT/'03_数据索引/runtime200.tar.gz','w:gz') as tar:tar.add(OUT,arcname='runtime200')
print('READY',time.time()-start,flush=True)
