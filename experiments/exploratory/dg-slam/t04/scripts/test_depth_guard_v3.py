"""CPU checks of actual patched functions, without CUDA/FAISS imports."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch

root = Path(__file__).resolve().parents[1] / '02_Gaussian适配审计/GS_SLAM借鉴_20261012/stage_a_patch'
def function(path, name, scope=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = tree.body if scope is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == scope).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env, env[name]

env, valid = function(root / 'dg_slam/gaussian/common.py', 'valid_t04_depth')
depth = torch.tensor([float('nan'), float('inf'), -1., 0., .49, .5, 8., 39.99, 40., 80., 1e7])
assert valid(depth).tolist() == [False, False, False, False, False, True, True, True, False, False, False]

# Actual model admission must return safely before any FAISS/CUDA work.
env, add = function(root / 'dg_slam/gaussian_model.py', 'add_neural_points', 'GaussianModel')
env['valid_t04_depth'] = valid
owner = SimpleNamespace(t04_keyframe=7)
assert add(owner, torch.zeros(3, 3), torch.ones(3, 3), torch.tensor([float('inf'), 40., 1e7]), torch.ones(3, 3), dynamic_radius=torch.ones(3)) == 0
assert add(owner, torch.empty(0, 3), torch.empty(0, 3), torch.empty(0), torch.empty(0, 3)) == 0
assert add(owner, torch.full((1, 3), float('nan')), torch.ones(1, 3), torch.ones(1), torch.ones(1, 3)) == 0

# Each real sampler returns only valid depth and preserves pixel/color alignment.
for name in ['get_samples', 'get_samples_with_pixel_grad', 'get_samples_point_add']:
    env, sample = function(root / 'dg_slam/gaussian/common.py', name)
    import numpy as np
    env.update(valid_t04_depth=valid, np=np)
    d = depth.reshape(1, -1)
    c = torch.arange(depth.numel()*3).reshape(1, -1, 3).float()
    env['get_rays_from_uv'] = lambda i,j,*args: (torch.zeros(len(i),3), torch.ones(len(i),3))
    if name == 'get_samples':
        env['get_sample_uv'] = lambda *args, **kwargs: (torch.arange(len(depth)).float(),torch.zeros(len(depth)),depth,c[0],torch.ones(len(depth),dtype=torch.bool))
        out = sample(0,1,0,len(depth),len(depth),1,1,0,0,torch.eye(4),d,c,'cpu',depth_filter=True,return_index=True)
    elif name == 'get_samples_with_pixel_grad':
        env['get_sample_uv_with_grad'] = lambda *args: np.arange(len(depth))
        out = sample(0,1,0,len(depth),len(depth),1,len(depth),1,1,0,0,torch.eye(4),d,c,'cpu')
    else:
        env['get_sample_uv_with_depth_res'] = lambda *args: np.arange(len(depth))
        env['get_sample_uv_with_opacity_acc'] = lambda *args: np.arange(len(depth))
        out = sample(torch.ones_like(d),c,torch.ones_like(d),len(depth),.1,0,1,0,len(depth),1,len(depth),1,1,0,0,torch.eye(4),d,c,'cpu',seg_mask=torch.ones_like(d,dtype=torch.bool))
    assert torch.equal(out[2], depth[valid(depth)]), name
    assert torch.equal(out[3], c[0][valid(depth)]), name
    assert out[4].tolist() == [5,6,7], name
result = {'status':'CPU_PASS_GPU_NOT_RUN', 'checks':['finite/range boundary mask', 'all-invalid/empty/nonfinite-rays admission exits before FAISS', 'three actual sampler functions preserve depth/color/pixel alignment'], 'limits':['not a SLAM run', 'post-optimization map extents not yet validated']}
(root / 'cpu_validation.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
print(json.dumps(result))
