"""Prepare isolated, size-selectable Stage A GPU worker; never launch here."""
from pathlib import Path
import ast

P=Path(__file__).resolve().parent
s=(P/'run_kitti_v2.py').read_text(encoding='utf-8')
s=s.replace("a=p.parse_args()", "p.add_argument('--frames', type=int, choices=[40,200], required=True);a=p.parse_args()")
s=s.replace("source=ROOT/'source_v2'", "source=ROOT/'source_v3_depthguard'")
s=s.replace("assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)", "assert torch.cuda.is_available()\n    assert torch.cuda.get_device_capability() in [(8,6),(12,0)]")
s=s.replace("n_img=200", "n_img=a.frames").replace("'frames':200", "'frames':a.frames")
s=s.replace("data,rows,input_meta=load_r1_window();meta.update(input_meta)", "data,rows,input_meta=load_r1_window();meta.update(input_meta)\n        data=data[:a.frames];rows=rows[:a.frames]\n        meta.update(intervention='Stage A depth guard only', depth_guard_preparation_sha256=sha256(ROOT/'stage_a/preparation.json'), cuda_capability=torch.cuda.get_device_capability(), runtime_frames=a.frames)")
s=s.replace("len(pixel_hash_to_id)==200", "len(pixel_hash_to_id)==a.frames")
s=s.replace("trajectory.shape==(200,7)", "trajectory.shape==(a.frames,7)")
needle="                if name=='mapping':print(json.dumps({'event':'mapping_completed','keyframe':int(args[0]),'gaussians':int(mapper.gaussians.get_xyz().shape[0])}),flush=True)"
assert needle in s
s=s.replace(needle, """                if name=='mapping':
                    xyz=mapper.gaussians.get_xyz().detach()
                    logscale=mapper.gaussians.get_scaling().detach()
                    camera=args[3][:3,3].to(xyz.device)
                    stats={'event':'mapping_completed','keyframe':int(args[0]),'gaussians':int(xyz.shape[0]),
                        'xyz_finite':bool(torch.isfinite(xyz).all().item()),
                        'scale_finite':bool(torch.isfinite(logscale).all().item()),
                        'max_distance_current_camera_m':float(torch.linalg.vector_norm(xyz-camera,dim=-1).max().item()),
                        'max_activated_scale_m':float(torch.exp(logscale).max().item())}
                    with (output/'map_range_history.jsonl').open('a') as stream:stream.write(json.dumps(stats)+'\\n')
                    print(json.dumps(stats),flush=True)""")
ast.parse(s)
(P/'run_stage_a.py').write_text(s,encoding='utf-8')
print('Prepared run_stage_a.py')
