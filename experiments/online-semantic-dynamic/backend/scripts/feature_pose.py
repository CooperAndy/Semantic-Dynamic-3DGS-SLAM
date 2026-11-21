import numpy as np,cv2
cv2.setNumThreads(4)
def gray(rgb):return cv2.cvtColor(np.uint8(np.rint(np.clip(rgb.transpose(1,2,0),0,1)*255)),cv2.COLOR_RGB2GRAY)
def bilinear(image,xy):
 h,w=image.shape;u,v=xy.T;valid=(u>=0)&(u<w-1)&(v>=0)&(v<h-1);x=np.clip(u,0,w-1.00001).astype(int);y=np.clip(v,0,h-1.00001).astype(int);a=u-x;b=v-y
 return (1-a)*(1-b)*image[y,x]+a*(1-b)*image[y,x+1]+(1-a)*b*image[y+1,x]+a*b*image[y+1,x+1],valid
def estimate_geometry(points,pixels,K,initial,seed=175028):
 points=np.asarray(points,np.float64);pixels=np.asarray(pixels,np.float64);K=np.asarray(K,np.float64);cv2.setRNGSeed(int(seed))
 if len(points)<20:return dict(ok=False,reason='fewer_than_20_correspondences',matches=len(points))
 ok,rv,tv,inliers=cv2.solvePnPRansac(points,pixels,K,None,iterationsCount=200,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
 if not ok or inliers is None:return dict(ok=False,reason='PnP_RANSAC_failed',matches=len(points))
 idx=inliers.reshape(-1);rv,tv=cv2.solvePnPRefineLM(points[idx],pixels[idx],K,None,rv,tv,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,30,1e-9));R,_=cv2.Rodrigues(rv);T=np.eye(4);T[:3,:3]=R;T[:3,3]=tv.reshape(3)
 pred=cv2.projectPoints(points[idx],rv,tv,K,None)[0].reshape(-1,2);errors=np.linalg.norm(pred-pixels[idx],axis=1);initial_pixels=project(points[idx],initial,K);initial_error=float(np.median(np.linalg.norm(initial_pixels-pixels[idx],axis=1)));median=float(np.median(errors));ratio=len(idx)/len(points)
 delta=T@np.linalg.inv(initial);angle=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))));translation=float(np.linalg.norm(delta[:3,3]));positive=float(np.mean((points[idx]@R.T+tv.reshape(3))[:,2]>0))
 gates=dict(positive_depth=positive>=.95,inliers=len(idx)>=20,inlier_fraction=ratio>=.5,median_reprojection=median<=1.,improvement=median<=.99*initial_error,translation=translation<=.5,rotation=angle<=3.)
 return dict(ok=all(gates.values()),reason='accepted_geometry' if all(gates.values()) else ','.join(k for k,v in gates.items() if not v),relative_w2c=T.tolist(),inliers=idx.tolist(),matches=len(points),inlier_fraction=ratio,median_reprojection_px=median,initial_median_reprojection_px=initial_error,translation_m=translation,rotation_deg=angle,gates=gates)
def project(points,T,K):
 X=np.asarray(points,np.float64)@T[:3,:3].T+T[:3,3];q=X@K.T;return q[:,:2]/q[:,2,None]

class Matcher:
 def __init__(self,target_rgb,target_mask=None):
  self.sift=cv2.SIFT_create(nfeatures=4000);self.target_kp,self.target_desc=self.sift.detectAndCompute(gray(target_rgb),None if target_mask is None else np.uint8(target_mask)*255);self.bf=cv2.BFMatcher(cv2.NORM_L2)
 def fit(self,rgb,depth,alpha,mask,K,initial,seed):
  kp,desc=self.sift.detectAndCompute(gray(rgb),np.uint8(mask)*255);base=dict(reference_keypoints=len(kp),target_keypoints=len(self.target_kp))
  if desc is None or self.target_desc is None or len(desc)<2 or len(self.target_desc)<2:return dict(base,ok=False,reason='insufficient_descriptors')
  forward=self.bf.knnMatch(desc,self.target_desc,k=2);reverse=self.bf.knnMatch(self.target_desc,desc,k=2);rev={pair[0].queryIdx:pair[0].trainIdx for pair in reverse if len(pair)==2 and pair[0].distance<.75*pair[1].distance}
  matched=[pair[0] for pair in forward if len(pair)==2 and pair[0].distance<.75*pair[1].distance and rev.get(pair[0].trainIdx)==pair[0].queryIdx];base['mutual_ratio_matches']=len(matched)
  if not matched:return dict(base,ok=False,reason='no_mutual_ratio_matches')
  xy=np.array([kp[m.queryIdx].pt for m in matched],np.float64);query=np.array([self.target_kp[m.trainIdx].pt for m in matched],np.float64);z,v=bilinear(depth,xy);a,va=bilinear(alpha,xy);keep=v&va&np.isfinite(z)&(z>0)&(a>.99);xy=xy[keep];query=query[keep];z=z[keep];points=np.column_stack(((xy[:,0]-K[0,2])/K[0,0]*z,(xy[:,1]-K[1,2])/K[1,1]*z,z))
  import semantic_runtime
  runtime=semantic_runtime.ACTIVE
  frame=int(seed)-180028;anchor=frame-1
  if anchor%20==19:anchor-=1
  base['pre_semantic']=dict(reference_pixels=xy.tolist(),image_points=query.tolist(),object_points=points.tolist())
  if runtime is not None:
   semantic_keep,report=runtime.match(anchor,frame,xy,query);base['semantic_filter']=report
   xy=xy[semantic_keep];query=query[semantic_keep];z=z[semantic_keep];points=points[semantic_keep]
  if len(xy)>=20 and np.median(np.linalg.norm(query-project(points,initial,K),axis=1))<=.05 and np.quantile(np.linalg.norm(query-project(points,initial,K),axis=1),.99)<=.1:return dict(base,ok=False,reason='subpixel_flow_identity',relative_w2c=initial.tolist(),matches=len(xy))
  result=estimate_geometry(points,query,K,initial,seed)
  if result.get('inliers'):
   q=query[result['inliers']];h,w=depth.shape;cell=np.clip((q[:,1]/h*4).astype(int),0,3)*4+np.clip((q[:,0]/w*4).astype(int),0,3);occupied=len(np.unique(cell));result['occupied_cells']=occupied
   if occupied<6:result.update(ok=False,reason='fewer_than_6_inlier_cells')
  return dict(base,**result,object_points=points.tolist(),image_points=query.tolist(),reference_pixels=xy.tolist())
