from pathlib import Path
import importlib.util,json,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e',ROOT/'scripts/evaluate_kitti_tracking.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
gt=e.load_kitti(ROOT/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt');out=ROOT/'02_Gaussian适配审计/差异排查_20261012';out.mkdir(exist_ok=True)
reports={}
for name,path in [('Gaussian',ROOT/'05_报告与归档/首次tracking评测_20261012/Gaussian_eval_P0.txt'),('ORB',ROOT/'01_ORB基线/历史评价/seq00_orb_kitti_200_p0_eval.txt')]:
    p=e.load_kitti(path);x=p[:,:3,3];y=gt[:,:3,3]
    rotation,_=Rotation.align_vectors(y-y.mean(0),x-x.mean(0));aligned=rotation.apply(x-x.mean(0))+y.mean(0)
    ate=float(np.sqrt(np.mean(np.sum((aligned-y)**2,axis=1))))
    relr=Rotation.from_matrix(p[:-1,:3,:3]).inv()*Rotation.from_matrix(p[1:,:3,:3]);gtr=Rotation.from_matrix(gt[:-1,:3,:3]).inv()*Rotation.from_matrix(gt[1:,:3,:3])
    relt=Rotation.from_matrix(p[:-1,:3,:3]).inv().apply(np.diff(x,axis=0));gtt=Rotation.from_matrix(gt[:-1,:3,:3]).inv().apply(np.diff(y,axis=0))
    terr=np.linalg.norm(relt-gtt,axis=1);rerr=(gtr.inv()*relr).magnitude();prior=e.score(p,gt)[0]
    # Text KITTI rotations are rounded and slightly nonorthogonal; Rotation projects them to SO(3).
    assert abs(ate-prior['ate_translation_rmse_m'])<1e-7
    assert abs(float(np.sqrt(np.mean(terr**2)))-prior['rpe_delta1']['translation_rmse_m'])<1e-7
    reports[name]={'status':'PASS','independent_scipy_ate_m':ate,'independent_vector_rpe_m':float(np.sqrt(np.mean(terr**2))),'rotation_rpe_rad':float(np.sqrt(np.mean(rerr**2))),'rpe_max_difference_from_matrix_scorer_m':abs(float(np.sqrt(np.mean(terr**2)))-prior['rpe_delta1']['translation_rmse_m']),'largest_rpe_pair_start_frames':np.argsort(terr)[-12:][::-1].tolist(),'largest_errors_m':sorted(terr.tolist(),reverse=True)[:12],'trajectory_length_m':float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())}
g=np.load(ROOT/'04_后续运行/最终运行产物/gaussian_snapshot.npz');ids=g['keyframe_input_indices'].astype(int);s=g['estimated_keyframe_c2w'].astype(float).copy();delta=np.array(json.loads((ROOT/'01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation']);s[:,:3,3]+=s[:,:3,:3]@delta
pred=e.load_kitti(ROOT/'05_报告与归档/首次tracking评测_20261012/Gaussian_eval_P0.txt')
reports['keyframe_vs_filled']={'keyframes':len(ids),'max_position_change_m':float(np.linalg.norm(s[:,:3,3]-pred[ids,:3,3],axis=1).max()),'median_position_change_m':float(np.median(np.linalg.norm(s[:,:3,3]-pred[ids,:3,3],axis=1))),'max_orientation_change_deg':float(np.rad2deg((Rotation.from_matrix(s[:,:3,:3]).inv()*Rotation.from_matrix(pred[ids,:3,:3])).magnitude().max()))}
(out/'independent_metric_audit.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print(json.dumps(reports,indent=2))
