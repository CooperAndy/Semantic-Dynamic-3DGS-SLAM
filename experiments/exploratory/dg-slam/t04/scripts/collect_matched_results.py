"""Invoke with an already authenticated SFTP client; never store credentials."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/root/autodl-tmp/t04_chy_kitti00_200_20261012'
def collect(sftp):
    states={}
    for mode in ['off','on']:
        try:
            name=sftp.open(REMOTE+'/v2_'+mode+'.run_name').read().decode().strip()
            m=json.loads(sftp.open(REMOTE+'/04_reproduction/'+name+'/run_manifest.json').read())
            states[mode]={'run':name,'status':m['status'],'frames':m['counters']['input_frames'],'seconds':m['wall_seconds']}
            if m['status']!='COMPLETED':continue
            dest=ROOT/'04_后续运行'/('matched_v2_'+mode);dest.mkdir(exist_ok=True)
            for file in ['run_manifest.json','trajectories.npz','pose_stages.npz','mapped_frame_identity.json']:
                sftp.get(REMOTE+'/04_reproduction/'+name+'/'+file,str(dest/file))
            for suffix,local in [('.exit','tracking.exit'),('.log','tracking.log')]:sftp.get(REMOTE+'/v2_'+mode+suffix,str(dest/local))
        except FileNotFoundError:states.setdefault(mode,{'status':'NOT_STARTED'})
    orb=ROOT/'04_后续运行/orb_matched_v2';orb.mkdir(exist_ok=True)
    for file in ['preparation.json','settings.yaml','stereo_kitti_headless.cc','build.log','tracking.exit','tracking.log','CameraTrajectory.txt']:
        try:
            sftp.stat(REMOTE+'/orb_matched_v2/'+file)
            temp=orb/(file+'.download');sftp.get(REMOTE+'/orb_matched_v2/'+file,str(temp));temp.replace(orb/file)
        except FileNotFoundError:pass
    for file in ['v2_preparation.json','projection_rasterizer_audit.json']:
        sftp.get(REMOTE+'/'+file,str(ROOT/'02_Gaussian适配审计/差异排查_20261012'/file))
    (ROOT/'04_后续运行/matched_collection_state.json').write_text(json.dumps(states,indent=2))
    return states
