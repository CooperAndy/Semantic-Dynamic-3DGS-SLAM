from common import *
import numpy as np,time
from trajectory_metrics import evaluate
start=time.monotonic();assert json.loads((R/'receipts/independent_audit.json').read_text())['status'].startswith('PASS');plan=json.loads((R/'plan.json').read_text());m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());ids=np.array([x['source_frame_id'] for x in m['rows'][:plan['frames']]],int)
gtpath=R.parent/'slam_project/data/kitti/dataset/poses/08.txt';calibpath=R/'provenance/calib.txt';cal={}
for line in calibpath.read_text().splitlines():
 name,values=line.split(':',1);a=np.fromstring(values,sep=' ')
 if name.strip() in ['P0','P2']:cal[name.strip()]=a.reshape(3,4)
P2=cal['P2'];assert np.allclose(P2[:,:3],np.array(m['K']),rtol=0,atol=5e-4);translation_cam2_from_rect0=np.linalg.solve(P2[:,:3],P2[:,3]);C0_from_C2=np.eye(4);C0_from_C2[:3,3]=-translation_cam2_from_rect0
raw=np.loadtxt(gtpath,dtype=np.float64);assert raw.shape[1]==12 and ids.max()<len(raw);G0=np.repeat(np.eye(4)[None],len(ids),axis=0);G0[:,:3]=raw[ids].reshape(-1,3,4);G2=G0@C0_from_C2;out=R/'evaluation_only';out.mkdir();results=[]
for variant in plan['variants']:
 path=R/'runs'/variant/'estimated_trajectory.npz'
 with np.load(path) as z:Epose=np.linalg.inv(z['w2c'].astype(np.float64));saved=z['c2w'].astype(np.float64);assert np.max(np.abs(saved-Epose))<.0003
 metrics,arrays=evaluate(Epose,G2);legacy,_=evaluate(Epose,G0);p=out/(variant+'_metrics.npz');np.savez_compressed(p,**arrays,ground_truth_P2_c2w=G2,estimated_c2w=Epose,source_frame_ids=ids);results.append(dict(variant=variant,primary_camera='P2_rectified_color_left',metrics=metrics,legacy_cam0_sensitivity=legacy,trajectory_sha256=sha(path),arrays_sha256=sha(p)))
dump(out/'trajectory_results.json',dict(at=now(),scope='1000-frame same-window DEV interaction study; not formal quality adoption',GT_path=str(gtpath),GT_sha256=sha(gtpath),calibration_path=str(calibpath),calibration_sha256=sha(calibpath),rect0_from_cam2=C0_from_C2.tolist(),scale_corrected=False,source_frame_ids=ids.tolist(),results=results,quality_adoption=False,seconds=time.monotonic()-start));print((out/'trajectory_results.json').read_text())
