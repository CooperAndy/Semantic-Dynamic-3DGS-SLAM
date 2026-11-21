"""Independent CPU evaluator: KITTI P2->P0, SE(3), fixed input-frame RPE."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'04_后续运行/最终运行产物'
OUT=ROOT/'05_报告与归档/首次tracking评测_20261012'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_kitti(path):
    values=np.loadtxt(path);assert values.shape==(200,12) and np.isfinite(values).all()
    m=np.tile(np.eye(4),(200,1,1));m[:,:3]=values.reshape(200,3,4);return m
def rigid(m):
    assert np.isfinite(m).all()
    assert np.max(np.abs(m[:,:3,:3].transpose(0,2,1)@m[:,:3,:3]-np.eye(3)))<1e-5
    assert np.max(np.abs(np.linalg.det(m[:,:3,:3])-1))<1e-5
def align_se3(pred,gt):
    x,y=pred[:,:3,3],gt[:,:3,3]
    u,_,vt=np.linalg.svd((y-y.mean(0)).T@(x-x.mean(0)))
    d=np.eye(3);d[2,2]=np.linalg.det(u@vt);r=u@d@vt
    a=pred.copy();a[:,:3,:3]=r@pred[:,:3,:3];a[:,:3,3]=(r@(x-x.mean(0)).T).T+y.mean(0)
    return a
def score(pred,gt):
    aligned=align_se3(pred,gt);errors=np.linalg.norm(aligned[:,:3,3]-gt[:,:3,3],axis=1)
    result={'ate_translation_rmse_m':float(np.sqrt(np.mean(errors**2))),'ate_mean_m':float(errors.mean()),'ate_max_m':float(errors.max()),'frames':len(gt),'alignment':'SE3','scale_correction':False}
    for delta in [1,10]:
        truth=np.linalg.inv(gt[:-delta])@gt[delta:]
        estimated=np.linalg.inv(aligned[:-delta])@aligned[delta:]
        e=np.linalg.inv(truth)@estimated
        trans=np.linalg.norm(e[:,:3,3],axis=1);angle=Rotation.from_matrix(e[:,:3,:3]).magnitude()
        result[f'rpe_delta{delta}']={'pairs':len(e),'translation_rmse_m':float(np.sqrt(np.mean(trans**2))),'rotation_rmse_rad':float(np.sqrt(np.mean(angle**2))),'rotation_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(angle**2))))}
    return result,aligned,errors
def main():
    meta=json.loads((RUN/'run_manifest.json').read_text());assert meta['status']=='COMPLETED',meta.get('error')
    assert (RUN/'tracking.exit').read_text().strip()=='0'
    assert meta['counters']['input_frames']==meta['counters']['input_frames_submitted']==200
    assert meta['gt_access_audit']['blocked_gt_accesses']==0
    for name,key in [('trajectories.npz','trajectory_sha256'),('gaussian_snapshot.npz','gaussian_snapshot_sha256')]:
        assert sha(RUN/name)==meta[key]
    trajectory=np.load(RUN/'trajectories.npz',allow_pickle=False)
    a=trajectory['estimated'];assert a.shape==(200,7) and np.isfinite(a).all()
    assert np.allclose(np.linalg.norm(a[:,3:],axis=1),1,atol=1e-5)
    runtime=json.loads((ROOT/'03_数据索引/runtime200/protocol.json').read_text())
    times=np.array([row['rgb_timestamp'] for row in runtime['rows']])
    assert np.array_equal(times,trajectory['rgb_timestamps']) and np.all(np.diff(times)>0)
    gaussian=np.load(RUN/'gaussian_snapshot.npz',allow_pickle=False)
    assert all(np.isfinite(gaussian[k]).all() for k in gaussian.files)
    ids=gaussian['keyframe_input_indices'].astype(int)
    identities=json.loads((RUN/'mapped_frame_identity.json').read_text())
    assert len(identities)==len(ids)==meta['mapped_keyframes']==meta['droid_keyframes']
    assert len(set(ids))==len(ids) and np.all((ids>=0)&(ids<200))
    for item,frame in zip(identities,ids):assert item['actual_rgb_input_frame']==item['recorded_input_frame']==int(frame)
    gt=load_kitti(ROOT/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt')
    orb=load_kitti(ROOT/'01_ORB基线/历史评价/seq00_orb_kitti_200_p0_eval.txt')
    delta=np.array(json.loads((ROOT/'01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])
    dg=np.tile(np.eye(4),(200,1,1));dg[:,:3,:3]=Rotation.from_quat(a[:,3:]).as_matrix();dg[:,:3,3]=a[:,:3]
    raw=dg.copy();dg[:,:3,3]+=dg[:,:3,:3]@delta
    rigid(dg);rigid(gt);rigid(orb)
    results={};aligned={};errors={}
    for name,poses in [('ORB_historical_stereo',orb),('Gaussian_adapted_no_semantic',dg)]:
        results[name],aligned[name],errors[name]=score(poses,gt)
    reference=json.loads((ROOT/'01_ORB基线/独立CPU核验.json').read_text())
    assert abs(results['ORB_historical_stereo']['ate_translation_rmse_m']-reference['ate_se3_no_scale_m'])<1e-10
    assert abs(results['ORB_historical_stereo']['rpe_delta1']['translation_rmse_m']-reference['rpe_delta1_translation_rmse_m'])<1e-10
    sparse=gaussian['estimated_keyframe_c2w'].astype(np.float64).copy();rigid(sparse)
    sparse[:,:3,3]+=sparse[:,:3,:3]@delta
    sparse_score,_,_=score(sparse,gt[ids])
    sparse_score['role']='Separate sparse keyframe diagnostic; its RPE delta counts keyframes, not input frames'
    old,new=results.values()
    comparison={'ate_change_percent':100*(new['ate_translation_rmse_m']/old['ate_translation_rmse_m']-1)}
    for d in [1,10]:comparison[f'rpe_delta{d}_translation_change_percent']=100*(new[f'rpe_delta{d}']['translation_rmse_m']/old[f'rpe_delta{d}']['translation_rmse_m']-1)
    report={'status':'CPU_EVALUATED','date':'2026-10-12','run_integrity':'PASS','full_trajectory':results,'gaussian_sparse_keyframes':sparse_score,'comparison_gaussian_relative_to_orb':comparison,'frame_ids':list(range(200)),'runtime':{'wall_seconds':meta['wall_seconds'],'peak_torch_allocated_bytes':meta['max_gpu_memory_bytes'],'keyframes':len(ids),'gaussians':meta['gaussian_count'],'counters':meta['counters']},'limits':['Single exposed DEV window and one seed','Gaussian uses resized 640x192 stereo-derived RGB-D and no semantic mask; ORB uses original stereo 1241x376','Post-run/filler trajectory, not strictly per-frame online estimates','Historical ORB WSL timing not hardware-matched','No GT pose or ORB trajectory fed to GPU worker; GT used only by this independent CPU evaluator','Not original DG-SLAM paper reproduction or formal method adoption'],'input_hashes':{'run_manifest':sha(RUN/'run_manifest.json'),'trajectory':sha(RUN/'trajectories.npz'),'gaussian_snapshot':sha(RUN/'gaussian_snapshot.npz'),'gt':sha(ROOT/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt'),'scorer':sha(Path(__file__))}}
    (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,poses in [('Gaussian_raw_P2',raw),('Gaussian_eval_P0',dg)]:np.savetxt(OUT/(name+'.txt'),poses[:,:3].reshape(200,12),fmt='%.10f')
    with (OUT/'per_frame_ate.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['frame','time_s','ORB_error_m','Gaussian_error_m'])
        for i in range(200):w.writerow([i,times[i],errors['ORB_historical_stereo'][i],errors['Gaussian_adapted_no_semantic'][i]])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    ax[0].plot(gt[:,0,3],gt[:,2,3],label='GT',color='black')
    for name,label in [('ORB_historical_stereo','ORB stereo'),('Gaussian_adapted_no_semantic','Gaussian adapted')]:
        ax[0].plot(aligned[name][:,0,3],aligned[name][:,2,3],label=label)
        ax[1].plot(range(200),errors[name],label=label)
    ax[0].set(xlabel='X (m)',ylabel='Z (m)',title='Independent SE(3) alignment, no scale');ax[0].axis('equal')
    ax[1].set(xlabel='Input frame',ylabel='Translation error (m)',title='KITTI 00: frames 0-199')
    for a in ax:a.legend();a.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'trajectory_and_error.png',dpi=160);plt.close(fig)
    lines=['# KITTI seq00前200帧：首次tracking评测','', '评测日期：2026-10-12。GPU运行完成、退出码0；轨迹200帧完整，文件哈希、关键帧身份、时间戳和有限性检查通过。','', '**口径：P2→P0相机转换，SE(3)对齐，无尺度校正；固定输入帧间隔RPE。**','', '| 指标 | 历史ORB Stereo | Gaussian工程适配 |','|---|---:|---:|',f"| ATE RMSE（m） | {old['ate_translation_rmse_m']:.6f} | {new['ate_translation_rmse_m']:.6f} |"]
    for d in [1,10]:
        lines.extend([f"| {d}帧平移RPE RMSE（m） | {old[f'rpe_delta{d}']['translation_rmse_m']:.6f} | {new[f'rpe_delta{d}']['translation_rmse_m']:.6f} |",f"| {d}帧旋转RPE RMSE（°） | {old[f'rpe_delta{d}']['rotation_rmse_deg']:.6f} | {new[f'rpe_delta{d}']['rotation_rmse_deg']:.6f} |"])
    lines.extend(['',f"Gaussian相对ORB：ATE变化{comparison['ate_change_percent']:+.2f}%；1帧平移RPE变化{comparison['rpe_delta1_translation_change_percent']:+.2f}%；10帧平移RPE变化{comparison['rpe_delta10_translation_change_percent']:+.2f}%。正数表示误差增加。",'', '本轮是双目派生RGB-D、640×192、无语义剔除的单次工程适配；ORB为历史原分辨率双目结果。输入与实现条件不同，不能据此断言Gaussian方法总体优于或劣于ORB，也不能当作高斯精修的独立因果消融。', '',f"GPU worker耗时{meta['wall_seconds']:.2f}秒，{len(ids)}关键帧，{meta['gaussian_count']}个高斯；显存数字仅为Torch峰值已分配，完整计数见metrics.json。",'', '已保存原始P2轨迹、P0评价轨迹、逐帧ATE、完整指标JSON和可复跑评价脚本。稀疏关键帧结果单独保存在JSON，不能替代200帧全轨迹结论。','','![轨迹与逐帧误差](trajectory_and_error.png)'])
    (OUT/'评测与ORB对比.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'full_trajectory':results,'comparison':comparison,'runtime':report['runtime']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
