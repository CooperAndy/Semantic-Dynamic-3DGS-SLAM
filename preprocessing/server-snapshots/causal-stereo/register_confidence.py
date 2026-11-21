"""Register confidence replay only after both full stereo traces and depth ABIs pass."""
from common import *
import argparse, importlib.util, time
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window',required=True)
    args=parser.parse_args();started=time.monotonic()
    initial=json.loads((R/'registration_cpu.json').read_text())
    win=next(x for x in initial['windows'] if x['id']==args.window)
    W=R/'windows'/args.window;F=I/'windows'/args.window
    assert not (W/'registration.json').exists()
    audits={}
    for side in ['left','right']:
        p=W/'receipts'/f'stereo_{side}_audit.json'
        assert p.exists(), f'Full stereo {side} audit required before confidence registration'
        audits[side]=json.loads(p.read_text())
        assert audits[side]['status']=='PASS_1000_STEREO_CURRENT_PAIR_TRACE_AND_FLOAT32_METRIC_DEPTH'
        assert audits[side]['CPU_registration_sha256']==sha(R/'registration_cpu.json')
        assert audits[side]['window']==args.window and len(audits[side]['files'])==1000
    context=json.loads((W/'cpu_context.json').read_text())
    assert context['window']==win and context['calib_sha256']==sha(W/'calib.txt')
    assert sha(F/'registration.json')==win['input_registration_sha256']
    input_reg=json.loads((F/'registration.json').read_text())
    assert len(input_reg['inputs'])==2000
    source_images={x['path']:x for x in input_reg['inputs']}
    assert sha(win['primary_pose_path'])==win['primary_pose_sha256']
    poses=np.loadtxt(win['primary_pose_path']);assert poses.shape==(1000,12) and np.isfinite(poses).all()
    assert sha(W/'provenance/legacy_confidence.py')==sha(R/'source/confidence.py')
    spec=importlib.util.spec_from_file_location('g21_confidence_registration',W/'provenance/legacy_confidence.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    calib=legacy.parse_calibration(W/'calib.txt')
    assert abs(calib['fx']*calib['baseline_m']-win['projection_beta'])<1e-9
    (W/'causal_pose_records').mkdir()
    rows=[]
    for i,pose in enumerate(poses):
        T=np.eye(4);T[:3]=pose.reshape(3,4)
        assert np.max(np.abs(T@np.linalg.inv(T)-np.eye(4)))<1e-9
        p=W/'causal_pose_records'/f'{i:06d}.npy';np.save(p,T,allow_pickle=False)
        inputs={}
        for side in ['left','right']:
            image=Path(win['input_view'])/f'image_{2 if side=="left" else 3}'/f'{i:06d}.png'
            parent=source_images[str(image)]
            assert image.stat().st_size==parent['bytes'] and sha(image)==parent['sha256']
            inputs[side]=dict(path=str(image),bytes=parent['bytes'],sha256=parent['sha256'])
            depth=audits[side]['files'][i];assert depth['index']==i
            assert Path(depth['path'])==W/f'raw_{side}'/f'{i:06d}_depth.npy'
            assert Path(depth['path']).stat().st_size==depth['bytes'] and sha(depth['path'])==depth['sha256']
            inputs['depth_'+side]={k:depth[k] for k in ['path','bytes','sha256']}
        inputs['pose']=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        rows.append(dict(index=i,source_frame=i+win['source_start'],inputs=inputs))
        assert time.monotonic()-started<600
    sources={p.relative_to(W).as_posix():sha(p) for folder in ['scripts','provenance']
             for p in (W/folder).rglob('*.py')}
    dump(W/'registration.json',dict(status='REGISTERED_PAST_ONLY_CONFIDENCE_AFTER_FULL_STEREO_AUDITS_NOT_EXECUTED',
        at=now(),window=args.window,source_start=win['source_start'],shape=win['shape'],K=win['K'],
        calibration=calib,calib_sha256=sha(W/'calib.txt'),frontend_primary_sha256=win['primary_pose_sha256'],
        source_files=sources,rows=rows,first_frame_temporal_multiplier=1.,temporal_neighbors='previous only',
        min_depth=.5,max_depth=40.,seed_threshold=.8,loss_threshold=.2,
        stereo_audit_sha256={side:sha(W/'receipts'/f'stereo_{side}_audit.json') for side in audits},
        registration_operator_sha256=sha(__file__),GT_runtime=False,quality_adoption=False))
    print('REGISTERED_CONFIDENCE_AFTER_STEREO_AUDITS',args.window,len(rows),flush=True)

if __name__=='__main__':main()
