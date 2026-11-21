"""Copy historical ORB evidence and audit the 200-frame baseline, CPU only."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT.parents[1]
OLD = FYP / '2026.8.26/第4步_ORB-SLAM3轨迹评估结果'
DATA = FYP / 'slam_project/data/kitti/sequences/00'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def poses(path):
    a = np.loadtxt(path)
    assert a.shape == (200, 12) and np.isfinite(a).all(), str(path)
    m = np.tile(np.eye(4), (200, 1, 1))
    m[:, :3] = a.reshape(200, 3, 4)
    return m

def main():
    for name in ['00_任务边界与协议', '01_ORB基线/原始输出', '01_ORB基线/历史评价',
                 '01_ORB基线/标定与脚本', '02_Gaussian适配审计', '03_数据索引',
                 '04_后续运行', '05_报告与归档']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    pairs = []
    raw = FYP/'orbslam3_seq00_200_results'
    for name in ['CameraTrajectory.txt', 'KeyFrameTrajectory.txt', 'KITTI00_color_seq00.yaml',
                 'README.txt', 'run_orbslam3_seq00.log']:
        pairs.append((raw/name, ROOT/'01_ORB基线/原始输出'/name))
    for name in ['seq00_orb_kitti_200.txt', 'seq00_orb_kitti_200_p0_eval.txt',
                 'seq00_gt_kitti_200.txt', 'seq00_orb_p0_ape.txt', 'seq00_orb_p0_ape.zip',
                 'seq00_orb_p0_rpe.txt', 'seq00_orb_p0_rpe.zip',
                 'seq00_orb_p0_trajectory_plot_trajectories.png',
                 'seq00_orb_p0_trajectory_plot_xyz.png', 'seq00_orb_p0_trajectory_plot_rpy.png']:
        pairs.append((OLD/'evaluation'/name, ROOT/'01_ORB基线/历史评价'/name))
    pairs.append((OLD/'ORB-SLAM3前端_KITTI_seq00_验收报告_20260826.md',
                  ROOT/'01_ORB基线/历史验收报告_含完整序列背景.md'))
    for p in (OLD/'scripts_and_config').iterdir():
        if p.is_file(): pairs.append((p, ROOT/'01_ORB基线/标定与脚本'/p.name))
    for name in ['calib.txt', 'times.txt']:
        pairs.append((DATA/name, ROOT/'01_ORB基线/标定与脚本'/name))
    provenance = []
    for source, target in pairs:
        if target.exists() and sha(target) != sha(source):
            raise RuntimeError(f'Refusing to overwrite changed evidence: {target}')
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        provenance.append({'source':str(source), 'copy':str(target.relative_to(ROOT)),
                           'bytes':target.stat().st_size, 'sha256':sha(target)})
    write_json(ROOT/'05_报告与归档/ORB复制来源与SHA256.json', provenance)

    p2 = poses(ROOT/'01_ORB基线/历史评价/seq00_orb_kitti_200.txt')
    p0 = poses(ROOT/'01_ORB基线/历史评价/seq00_orb_kitti_200_p0_eval.txt')
    gt = poses(ROOT/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt')
    raw_poses = poses(ROOT/'01_ORB基线/原始输出/CameraTrajectory.txt')
    assert np.allclose(p2, raw_poses, atol=1e-8, rtol=0)
    calib = {}
    for line in (DATA/'calib.txt').read_text().splitlines():
        key, values = line.split(':', 1)
        calib[key] = np.fromstring(values, sep=' ').reshape(3, 4)
    def offset(k): return np.linalg.solve(calib[k][:,:3], calib[k][:,3])
    trans = offset('P2') - offset('P0')
    converted = p2.copy()
    converted[:,:3,3] += p2[:,:3,:3] @ trans
    assert np.allclose(converted, p0, atol=2e-9, rtol=0)
    full_gt = np.loadtxt(FYP/'slam_project/data/kitti/poses/00.txt')
    assert np.allclose(full_gt[:200], gt[:,:3,:].reshape(200,12), atol=1e-8, rtol=0)
    x,y = p0[:,:3,3],gt[:,:3,3]
    u,_,vt = np.linalg.svd((y-y.mean(0)).T @ (x-x.mean(0)))
    correction = np.eye(3);correction[2,2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    aligned = p0.copy()
    aligned[:,:3,:3] = rotation @ p0[:,:3,:3]
    aligned[:,:3,3] = (rotation @ (x-x.mean(0)).T).T+y.mean(0)
    ate = float(np.sqrt(np.mean(np.sum((aligned[:,:3,3]-y)**2,axis=1))))
    def rpe(delta):
        actual = np.linalg.inv(gt[:-delta]) @ gt[delta:]
        estimated = np.linalg.inv(aligned[:-delta]) @ aligned[delta:]
        error = np.linalg.inv(actual) @ estimated
        return float(np.sqrt(np.mean(np.sum(error[:,:3,3]**2,axis=1))))
    assert abs(ate-0.488110)<5e-7 and abs(rpe(1)-0.033239)<5e-7
    report = {'status':'PASS', 'scope':'historical standalone 200-frame baseline; no SLAM rerun',
              'frames':200, 'frame_range':[0,199], 'p2_to_p0_translation':trans.tolist(),
              'raw_vs_export_max_abs':float(np.max(np.abs(raw_poses-p2))),
              'p2_to_p0_conversion_max_abs':float(np.max(np.abs(converted-p0))),
              'ate_se3_no_scale_m':ate, 'rpe_delta1_translation_rmse_m':rpe(1),
              'rpe_delta10_translation_rmse_m':rpe(10),
              'rpe_delta10_role':'new CPU audit metric, not historical reported value',
              'rotation_orthogonality_max_abs':float(np.max(np.abs(
                  p2[:,:3,:3].transpose(0,2,1)@p2[:,:3,:3]-np.eye(3)))),
              'keyframes':len(np.loadtxt(ROOT/'01_ORB基线/原始输出/KeyFrameTrajectory.txt')),
              'timing_role':'historical WSL run; not hardware-matched to server GPU method'}
    write_json(ROOT/'01_ORB基线/独立CPU核验.json',report)

    timestamps = np.loadtxt(DATA/'times.txt')[:200]
    assert len(timestamps)==200 and np.all(np.diff(timestamps)>0)
    rows=[]; issues=[]
    for i,t in enumerate(timestamps):
        row={'frame_id':i,'timestamp_s':float(t)}
        for camera in ['image_2','image_3']:
            p=DATA/camera/f'{i:06d}.png'
            row[camera+'_path']=str(p)
            if not p.is_file():
                issues.append(str(p));row[camera+'_sha256']='MISSING';continue
            with Image.open(p) as img:
                assert img.size==(1241,376),(str(p),img.size)
                img.verify()
            row[camera+'_sha256']=sha(p)
        rows.append(row)
    with (ROOT/'03_数据索引/seq00_000000_000199_stereo_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    depth=[]
    for name in ['depth_sgbm','depth_sgbm_backup_before_regen']:
        folder=DATA/name
        depth.append({'path':str(folder),'exists':folder.exists(),
                      'first_files':[p.name for p in sorted(folder.glob('*'))[:5]],
                      'note':'presence only; units, generation parameters and future-frame usage not validated'})
    write_json(ROOT/'03_数据索引/数据核验.json',{'status':'PASS' if not issues else 'INCOMPLETE',
        'frames':200,'images_expected':400,'missing':issues,'image_shape':[376,1241],
        'time_first_s':float(timestamps[0]),'time_last_s':float(timestamps[-1]),
        'original_images_copied':False,'depth_candidates':depth})
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Copied files:',len(provenance),'Missing stereo images:',len(issues))

if __name__=='__main__': main()
