"""Sparse independent LiDAR GT; nearest ALL-label surface before ignore mapping."""
import numpy as np
CITY_MAP={10:13,252:13,11:18,13:15,257:15,15:17,16:16,256:16,18:14,258:14,30:11,254:11,31:12,253:12,32:12,255:12,40:0,60:0,48:1,50:2,51:4,70:8,71:8,72:9,80:5,81:7}
MOVING_IDS=[252,253,254,255,256,257,258,259]
PRIMARY=[0,1,2,4,5,7,8,9,11,12,13,14,15,16,17,18]
PALETTE=np.array([[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]],np.uint8)
def calibration(path):
 ret={}
 for line in path.read_text().splitlines():
  if ':' in line:k,v=line.split(':',1);ret[k]=np.array([float(x) for x in v.split()])
 P=ret['P2'].reshape(3,4);T=np.eye(4);T[:3]=ret['Tr'].reshape(3,4)
 assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) and abs(np.linalg.det(T[:3,:3])-1)<1e-5
 return P,T
def project(points,packed,P,T,h,w):
 assert points.ndim==2 and points.shape[1]==4 and len(packed)==len(points)
 xyz=points[:,:3].astype(np.float64);hom=np.c_[xyz,np.ones(len(xyz))];camera=hom@T.T;projected=camera@P.T;z=projected[:,2];finite=np.isfinite(projected).all(1)&np.isfinite(points).all(1)&(z>0)
 uv=np.full((len(points),2),np.nan);uv[finite]=projected[finite,:2]/z[finite,None]
 px=np.zeros(len(points),np.int64);py=px.copy();px[finite]=np.rint(uv[finite,0]).astype(np.int64);py[finite]=np.rint(uv[finite,1]).astype(np.int64)
 inside=finite&(px>=0)&(px<w)&(py>=0)&(py<h);candidates=np.flatnonzero(inside);linear=py[candidates]*w+px[candidates];order=np.lexsort((candidates,z[candidates],linear));ordered=candidates[order];pix=linear[order];first=np.r_[True,pix[1:]!=pix[:-1]];chosen=ordered[first];pix=pix[first]
 rawsem=(packed&np.uint32(0xffff)).astype(np.uint16);rawinstance=(packed>>np.uint32(16)).astype(np.uint16)
 labels=np.full(h*w,255,np.uint8);sem=np.zeros(h*w,np.uint16);instance=sem.copy();depth=np.zeros(h*w,np.float32);identity=np.full(h*w,-1,np.int32)
 sem[pix]=rawsem[chosen];instance[pix]=rawinstance[chosen];depth[pix]=z[chosen];identity[pix]=chosen
 for raw,city in CITY_MAP.items():labels[pix[rawsem[chosen]==raw]]=city
 return {'label':labels.reshape(h,w),'raw_semantic':sem.reshape(h,w),'instance':instance.reshape(h,w),'depth':depth.reshape(h,w),'point_index':identity.reshape(h,w)}, {'points':len(points),'positive_finite_camera_points':int(finite.sum()),'in_image_points':len(candidates),'unique_projected_pixels':len(chosen),'mapped_GT_pixels':int((labels!=255).sum()),'all_point_semantic_counts':{str(i):int(n) for i,n in zip(*np.unique(rawsem,return_counts=True))}},uv
