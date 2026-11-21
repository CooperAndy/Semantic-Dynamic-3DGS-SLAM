"""Registered replay experiment with observation and partial failure preservation."""
from pathlib import Path
import sys, os, json, hashlib, time, importlib.util, traceback

R = Path(__file__).resolve().parents[1]
S = R/'source'
variant = sys.argv[1]
registration=json.loads((R/'registration.json').read_text())
assert variant in registration['variants']
out = R/'runs'/variant
assert not out.exists(), 'Keep prior outputs immutable'
out.mkdir(parents=True)
started, blocked = time.monotonic(), []
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p.endswith('groundtruth.txt') or '/poses/' in p or '/evaluation_only/' in p:
            blocked.append(p)
            raise RuntimeError('GT/evaluation file blocked in prediction worker')
sys.addaudithook(guard)
sys.path.insert(0, str(S))
os.chdir(S)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

import signal
def registered_time_stop(signum, frame):
    raise RuntimeError('Registered per-run time budget stop')
signal.signal(signal.SIGTERM, registered_time_stop)

receipt = dict(task='T3-G01_G02', variant=variant, status='RUNNING', quality='NOT_EVALUATED')
try:
    import torch, numpy as np
    torch.set_num_threads(4)
    config = load('smoke_config', S/'configs/kitti'/'G00_S0.py').config
    config['workdir']=str(R/'runs')
    config['run_name']=variant
    config['data']['num_frames']=200
    config['eval_only_frames']=[i for i in config['eval_only_frames'] if i < config['data']['num_frames']]
    config['save_checkpoints']=True
    config['checkpoint_interval']=99
    config['task_id']='C_STABLE_ID_DEV_20261027'
    count=int(config['data']['num_frames'])
    heldout=set(config['eval_only_frames'])
    receipt['task']=config['task_id']
    core = load('smoke_core', S/'scripts/splatam_external_tm.py')
    core.seed_everything(config['seed'])
    renderer_calls=[0]
    NativeRenderer=core.Renderer
    class CountedRenderer:
        def __init__(self,*args,**kwargs):self.renderer=NativeRenderer(*args,**kwargs)
        def __call__(self,*args,**kwargs):
            renderer_calls[0]+=1
            return self.renderer(*args,**kwargs)
        def __getattr__(self,name):return getattr(self.renderer,name)
    core.Renderer=CountedRenderer
    from c04n_addition import add_new_gaussians as selected_addition
    core.add_new_gaussians=lambda *a,**k:selected_addition(*a,**k,core=core,mode=config['mapping']['candidate_rule'],cap=config['mapping']['new_point_cap'],seed=config['mapping']['sampling_seed'],log_path=out/'addition_history.jsonl',fb=config['mapping']['residual_fb'])
    native_init=core.initialize_first_timestep
    def observed_initialization(*a,**k):
        result=native_init(*a,**k)
        from external_tracking import map_hash
        receipt['initial_gaussian_sha256']=map_hash(result[0])
        return result
    core.initialize_first_timestep=observed_initialization
    observation = dict(mapping_loss_calls={}, mapping_target_frames=[], additions=[], camera_backward_loss_calls=0)
    original_loss, original_add = core.get_loss, core.add_new_gaussians
    last_mapped_online = [0]
    def observe_loss(*args, **kwargs):
        if kwargs.get('mapping'):
            online = int(sys._getframe(1).f_locals['time_idx'])
            assert online not in heldout and int(args[3]) not in heldout
            last_mapped_online[0] = online
            key = str(online)
            observation['mapping_loss_calls'][key] = observation['mapping_loss_calls'].get(key, 0)+1
            if int(args[3]) not in observation['mapping_target_frames']:
                observation['mapping_target_frames'].append(int(args[3]))
        elif kwargs.get('tracking'):
            assert last_mapped_online[0] < int(args[3]), 'Current frame entered mapping before tracking'
            observation['camera_backward_loss_calls'] += 1
        return original_loss(*args, **kwargs)
    def observe_add(*args, **kwargs):
        frame = int(args[4])
        assert frame not in heldout
        count = len(args[0]['means3D'])
        result = original_add(*args, **kwargs)
        observation['additions'].append(dict(frame=frame, added=int(len(result[0]['means3D'])-count)))
        return result
    core.get_loss, core.add_new_gaussians = observe_loss, observe_add
    from stable_trace import install
    trace_state=install(core,out)
    core.rgbd_slam(config)
    torch.cuda.synchronize()
    receipt['algorithm_wall_seconds'] = time.monotonic()-started
    receipt['native_renderer_calls_algorithm']=renderer_calls[0]
    with np.load(out/'estimated_trajectory.npz') as z:
        ids, poses = z['frame_ids'].copy(), z['c2w'].copy()
    assert np.array_equal(ids, np.arange(count)) and np.isfinite(poses).all()
    events = [json.loads(s) for s in (out/'external_tracking.jsonl').read_text().splitlines()]
    assert len(events) == count and all(e['past_map_max_frame'] < e['frame'] for e in events)
    assert all(e['mapping_heldout'] == (e['frame'] in heldout) for e in events)
    refined = [e for e in events if 'map_before_sha256' in e]
    assert all(e['map_before_sha256'] == e['map_after_sha256'] for e in refined)
    steps = [s for e in refined for s in e['steps']]
    assert all(np.isfinite(s['gradient_norm']) and s['gradient_norm'] > 0 and s['finite_post'] for s in steps)
    if config['tracking']['camera_policy'] == 'external_init_refine':
        assert steps, 'No native camera optimization coverage'
    else:
        manifest = json.loads((R/'inputs/runtime/input_manifest.json').read_text())
        raw = np.loadtxt(manifest['pose_file']).reshape(200, 3, 4)
        receipt['fixed_ORB_max_abs_difference'] = float(np.max(np.abs(poses[:, :3]-raw[:count])))
        with np.load(R/'inputs/runtime/orb_adapter_reference.npz') as z:expected=z['c2w'][:count]
        assert np.array_equal(poses,expected), 'Fixed cameras must match independently audited native adapter'
    with np.load(out/'params.npz', allow_pickle=False) as z:
        assert not np.isin(z['timestep'],list(heldout)).any()
        assert not np.isin(z['keyframe_time_indices'],list(heldout)).any()
        keys = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales', 'cam_unnorm_rots', 'cam_trans']
        p = {k: torch.tensor(z[k], device='cuda') for k in keys}
        K = z['intrinsics'].copy()
        receipt['gaussians'] = int(len(z['means3D']))
    from external_tracking import render
    from utils.recon_helpers import setup_camera
    from PIL import Image
    data = dict(cam=setup_camera(1241, 376, K, np.eye(4)), w2c=torch.eye(4, device='cuda'))
    previews = []
    preview_ids=sorted(set([0,sorted(heldout)[len(heldout)//2] if heldout else count//2,count-1]))
    for index in preview_ids:
        rgb, ds = render(p, data, index, core)
        assert rgb.shape == (3,376,1241) and ds.shape == (3,376,1241)
        assert np.isfinite(rgb).all() and np.isfinite(ds).all()
        assert ds[1].min() >= -1e-6 and ds[1].max() <= 1+1e-5
        np.savez_compressed(out/f'engineering_render_{index:06d}.npz', rgb=rgb, depth_silhouette=ds)
        Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255)).save(out/f'engineering_render_{index:06d}.png')
        previews.append(dict(frame=index, alpha_gt099_fraction=float((ds[1]>.99).mean()), scope='Final-map engineering check, no quality acceptance'))
    receipt.update(status='PASS_ENGINEERING', frames=count, native_camera_updates=len(steps),
                   accepted=sum(e.get('accepted') is True for e in events),
                   fallbacks=sum(e.get('accepted') is False for e in events),
                   final_map_previews=previews, heldout_points_and_keyframes_absent=True,
                   GT_scope='Prediction process denied GT paths; no separate GT-value-perturbation replay')
except BaseException as error:
    receipt.update(status='FAIL_PRESERVED', error=repr(error), traceback=traceback.format_exc())
    try:
        tb = error.__traceback__
        while tb:
            f = tb.tb_frame
            if f.f_code.co_name == 'rgbd_slam' and 'params' in f.f_locals:
                p = f.f_locals['params']
                n = len(f.f_locals.get('_events', []))
                core.save_params_ckpt(p, str(out), max(n-1, 0))
                if n:
                    from external_tracking import camera
                    w = np.stack([camera(p,i,core).cpu().numpy() for i in range(n)])
                    np.savez_compressed(out/'partial_trajectory.npz', frame_ids=np.arange(n), w2c=w, c2w=np.linalg.inv(w))
                receipt['partial_logged_frames'] = n
                break
            tb = tb.tb_next
    except Exception as preserve_error:
        receipt['partial_preservation_error'] = repr(preserve_error)
    raise
finally:
    receipt.update(wall_seconds=time.monotonic()-started, GT_blocked_opens=len(blocked))
    if 'renderer_calls' in globals():
        receipt['native_renderer_calls_total']=renderer_calls[0]
    if 'observation' in globals():
        receipt['execution_observation'] = observation
    if 'torch' in globals() and torch.cuda.is_initialized():
        receipt['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
    (out/'smoke_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
