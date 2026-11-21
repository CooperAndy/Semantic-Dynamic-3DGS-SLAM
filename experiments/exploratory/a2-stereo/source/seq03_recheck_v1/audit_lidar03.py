from pathlib import Path
import numpy as np,hashlib,json
R=Path('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_recheck_v1');D=Path('/root/autodl-tmp/A2_stereo_robust_geometry/reference_seq03_000000_000199_lidar');I=Path('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_input_20261029_retry1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P=R/'lidar_projected';pm=json.loads((P/'manifest.json').read_text());raw=json.loads((D/'manifest.json').read_text());r={x['source_frame']:x for x in raw['frames']}
cal=I/'input_view/calib.txt';assert sha(cal)==pm['calibration_sha256']
v={k:np.array([float(q) for q in vals.split()]) for line in cal.read_text().splitlines() if ':' in line for k,vals in [line.split(':',1)]};p=v['P2'].reshape(3,4);tr=v['Tr'].reshape(3,4);K=p[:,:3];t=np.linalg.solve(K,p[:,3]);rows=[]
for row in pm['frames']:
 i=row['frame'];src=D/r[i]['filename'];assert sha(src)==r[i]['sha256']==row['source_sha256'];points=np.fromfile(src,dtype='<f4').reshape(-1,4).astype(np.float64);cam=points[:,:3]@tr[:,:3].T+tr[:,3]+t;q=cam@K.T;z=cam[:,2]
 ok=np.isfinite(points).all(1)&np.isfinite(q).all(1)&(q[:,2]>0);uv=np.zeros((len(z),2),np.int64);uv[ok]=np.rint(q[ok,:2]/q[ok,2,None]).astype(np.int64);ok&=(uv[:,0]>=0)&(uv[:,0]<1242)&(uv[:,1]>=0)&(uv[:,1]<375)
 image=np.full(375*1242,np.inf);np.minimum.at(image,uv[ok,1]*1242+uv[ok,0],z[ok]);image[~np.isfinite(image)]=0;image=image.reshape(375,1242).astype(np.float32)
 saved=P/f'{i:06d}.npz';assert sha(saved)==row['projected_sha256'];ref=np.load(saved)['depth'];assert np.array_equal(image,ref),i
 rows.append({'frame':i,'matched_pixels':int((ref>0).sum()),'projection_sha256':sha(saved)})
out={'status':'PASS_INDEPENDENT_REPROJECTION','method':'camera-center transform and minimum.at z buffer','frames':rows,'scope':'Projection arithmetic only; no verification of temporal alignment, occlusion or map accuracy'}
(R/'independent_projection_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
