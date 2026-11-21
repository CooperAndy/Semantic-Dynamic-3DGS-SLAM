from pathlib import Path
import json
import shutil
import numpy as np
from export_s3_assets_reference import prepare_records,write_ply,camera_poses,trajectory_records,viewer_settings,sha256

R=Path(__file__).resolve().parent.parent

def fixed_settings(params, centers, rotations, frame):
    settings=viewer_settings(params['means3D'])
    pos=centers[frame].astype(float)
    target=pos+rotations[frame,:,2]*20
    # Same export coordinate convention as the established viewer settings.
    pos=(pos*np.array([-1,-1,1])).tolist();target=(target*np.array([-1,-1,1])).tolist()
    fov=float(np.degrees(2*np.arctan(float(params['org_height'])/(2*float(params['intrinsics'][1,1])))))
    settings['postEffectSettings']['sharpness']['enabled']=False
    settings['animTracks']=[{'name':f'fixed_source_{frame}','duration':1,'frameRate':30,'loopMode':'repeat','interpolation':'linear','smoothness':0,'keyframes':{'times':[0,1],'values':{'position':pos+pos,'target':target+target,'fov':[fov,fov]}}}]
    settings['cameras']=[{'initial':{'position':pos,'target':target,'fov':fov}}]
    return settings

def overview_settings(params):
    settings=viewer_settings(params['means3D'])
    low,high=np.quantile(params['means3D'],[0.01,0.99],axis=0)
    center=(low+high)/2;extent=high-low
    target=[-float(center[0]),-float(center[1]),float(center[2])]
    pos=[target[0],target[1]+max(95.,float(extent[2])*.65),target[2]-max(65.,float(extent[2])*.55)]
    settings['animTracks']=[{'name':'trajectory_overview','duration':1,'frameRate':30,'loopMode':'repeat','interpolation':'linear','smoothness':0,'keyframes':{'times':[0,1],'values':{'position':pos+pos,'target':target+target,'fov':[62,62]}}}]
    settings['cameras']=[{'initial':{'position':pos,'target':target,'fov':62}}]
    return settings

def main():
    for version,relative in [('F200','runs/F200_orb_raft_conf060/params.npz'),('F1','runs/F1_appearance800/params.npz')]:
        source=R/relative;out=R/'viewer_assets'/f'KITTI09_{version}'
        if out.exists(): assert not any(out.iterdir()), 'Refusing a populated export directory'
        out.mkdir(parents=True,exist_ok=True)
        with np.load(source,allow_pickle=False) as f: p={k:f[k] for k in f.files}
        centers,rots=camera_poses(p);assert len(centers)==200
        expected=np.loadtxt(R/'poses_estimated_camera2.txt').reshape(200,3,4)
        camera_error=float(np.linalg.norm(centers-expected[:,:3,3],axis=1).max())
        camera_tolerance=float(max(0.001,64*np.finfo(np.float32).eps*float(np.max(np.abs(expected[:,:3,3])))))
        assert camera_error<camera_tolerance, (camera_error,camera_tolerance)
        assert np.allclose(rots,expected[:,:3,:3],rtol=0,atol=1e-5)
        rgb=prepare_records(p['means3D'],p['rgb_colors'],p['log_scales'],p['unnorm_rotations'],p['logit_opacities'])
        traj=trajectory_records(centers,np.asarray(p['keyframe_time_indices'],dtype=np.int64))
        name=f'KITTI09_{version}'
        for suffix,records in [('RGB',rgb),('Trajectory',traj),('RGB_with_Trajectory',np.concatenate((rgb,traj)))]:
            path=out/f'{name}_{suffix}.ply';write_ply(path,records)
            raw=path.read_bytes();old=b'KITTI 21 frames 1200-1399';new=b'KITTI 09 frames 0300-0499';assert len(old)==len(new);assert old in raw[:1000]
            with path.open('r+b') as f:f.write(raw[:1000].replace(old,new))
        ks=p['intrinsics']
        cameras=[{'id':i,'source_frame':i,'img_name':f'{i:06d}.png','width':1226,'height':370,'position':centers[i].astype(float).tolist(),'rotation':rots[i].astype(float).tolist(),'fx':float(ks[0,0]),'fy':float(ks[1,1]),'pose_source':'ORB-SLAM3 estimated rectified P2 c2w, rebased'} for i in range(200)]
        (out/f'{name}_camera_poses.json').write_text(json.dumps(cameras,indent=2))
        (out/'viewer_settings_source_99.json').write_text(json.dumps(fixed_settings(p,centers,rots,99),indent=2))
        (out/'viewer_settings_trajectory.json').write_text(json.dumps(overview_settings(p),indent=2))
        shutil.copy2(source,out/'params.npz')
        manifest={'sequence':'09','source_frames':[0,199],'gaussian_count':len(rgb),'camera_count':200,'camera_numeric_error_m':camera_error,'camera_numeric_tolerance_m':camera_tolerance,'pose_source':'ORB-SLAM3 estimated P2, rebased; no GT input','depth_source':'RAFT-Stereo Middlebury, 32 iterations, confidence>=0.50, range 0.5-40m','mask_policy':'no semantic-category exclusion; no motion filtering','map_kind':version,'params_sha256':sha256(source),'status':'ALL_200_SEEN_DEPLOYMENT_NO_HELDOUT','files':{q.name:{'bytes':q.stat().st_size,'sha256':sha256(q)} for q in out.iterdir() if q.is_file()}}
        (out/f'{name}_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'variant':version,'gaussians':len(rgb),'manifest':str(out/f'{name}_manifest.json')}),flush=True)

if __name__=='__main__':main()
