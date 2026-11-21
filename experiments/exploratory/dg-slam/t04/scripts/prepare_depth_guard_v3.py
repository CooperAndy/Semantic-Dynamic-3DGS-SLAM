"""Build isolated Stage A sources from downloaded, exact V2 parent files.

No GPU run or remote mutation. Stage B remains a separate intervention.
"""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / '02_Gaussian适配审计/GS_SLAM借鉴_20261012'
OUT = AUDIT / 'stage_a_patch'
OUT.mkdir(exist_ok=True)

def replace_once(text, before, after):
    assert text.count(before) == 1, (before, text.count(before))
    return text.replace(before, after)

common = (AUDIT / 'parent_v2_common.py').read_text(encoding='utf-8')
helper = '''
def valid_t04_depth(depth):
    # Camera z depth in metres, not Euclidean distance to world origin.
    return torch.isfinite(depth) & (depth >= 0.5) & (depth < 40.0)

'''
common = replace_once(common, '\ndef get_samples(H0,', helper + '\ndef get_samples(H0,')
common = replace_once(common, 'mask = (sample_depth > 0) & (sample_depth < 8.0)', 'mask = valid_t04_depth(sample_depth)')
common = replace_once(common, 'mask = sample_depth > 0', 'mask = valid_t04_depth(sample_depth)')
common = replace_once(common, 'mask = (sample_depth > 0.1)', 'mask = valid_t04_depth(sample_depth)')
common = replace_once(common, 'depth_res = torch.abs(depth - depth_render) * (depth > 0) * (depth < 8.0)',
    'depth_res = torch.where(valid_t04_depth(depth) & torch.isfinite(depth_render), torch.abs(depth - depth_render), torch.zeros_like(depth))')

model = (AUDIT / 'parent_v2_gaussian_model.py').read_text(encoding='utf-8')
model = replace_once(model, 'from dg_slam.gaussian.common import setup_seed, clone_kf_dict',
    'from dg_slam.gaussian.common import setup_seed, clone_kf_dict, valid_t04_depth')
model = replace_once(model, '            mask = batch_gt_depth > 0', '''            mask = valid_t04_depth(batch_gt_depth)
            mask = mask & torch.isfinite(batch_rays_o).all(dim=-1) & torch.isfinite(batch_rays_d).all(dim=-1)
            mask = mask & torch.isfinite(batch_gt_color).all(dim=-1)
            print('T04_ADD_GUARD', {'keyframe': getattr(self, 't04_keyframe', None),
                  'input': int(mask.numel()), 'accepted': int(mask.sum().item()),
                  'rejected': int((~mask).sum().item()), 'is_pts_grad': bool(is_pts_grad)}, flush=True)
            if not mask.any():
                return 0
            if dynamic_radius is not None:
                dynamic_radius = dynamic_radius[mask]''')
# FAISS must not receive an empty array when every candidate already exists.
model = replace_once(model, '            self._cloud_pos += pts.tolist()',
    '            if pts.shape[0] == 0:\n                return 0\n\n            self._cloud_pos += pts.tolist()')

mapper = (AUDIT / 'parent_v2_mapper.py').read_text(encoding='utf-8')
needle = '    def mapping(self, idx, gt_color, gt_depth, estimated_c2w, seg_mask, init = False):'
assert mapper.count(needle) == 1
pos = mapper.index('\n', mapper.index(needle)) + 1
mapper = mapper[:pos] + '''        # Diagnostic only: do not replace front-end or input pseudo-depth.
        self.gaussians.t04_keyframe = int(idx)
        with torch.no_grad():
            finite = torch.isfinite(gt_depth)
            positive = finite & (gt_depth > 0)
            valid = positive & (gt_depth >= 0.5) & (gt_depth < 40.0)
            values = gt_depth[positive]
            print('T04_MAPPER_DEPTH', {'keyframe': int(idx),
                  'source_tstamp': float(self.video.tstamp[int(idx)].item()),
                  'pixels': int(gt_depth.numel()), 'valid_0p5_40': int(valid.sum().item()),
                  'nonfinite': int((~finite).sum().item()),
                  'positive_max_m': float(values.max().item()) if values.numel() else None}, flush=True)
''' + mapper[pos:]

files = {'dg_slam/gaussian/common.py': common,
         'dg_slam/gaussian_model.py': model,
         'dg_slam/gs_tracking_mapping.py': mapper}
for name, text in files.items():
    ast.parse(text)
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {'status': 'PREPARED_NOT_RUN', 'depth_metres': '[0.5,40)',
    'parent_sha256': {n: sha(AUDIT / ('parent_v2_' + s + '.py')) for n, s in
        [('dg_slam/gaussian/common.py', 'common'), ('dg_slam/gaussian_model.py', 'gaussian_model'), ('dg_slam/gs_tracking_mapping.py', 'mapper')]},
    'patched_sha256': {n: sha(OUT / n) for n in files},
    'changes': ['all three addition sample paths use finite z depth [0.5,40)',
                'uniform sample and residual proposal historical 8m limit expanded to 40m; support distribution changes explicitly',
                'model final admission guards rays/colors/depth and synchronizes dynamic_radius',
                'empty admission/duplicate sets return before FAISS',
                'per-keyframe mapper depth and per-admission rejection logs'],
    'unchanged': ['DROID input and settings', 'tracking loss/mask/optimizer', 'GS-SLAM selector disabled', '20m experiment deferred'],
    'limits': ['cannot prove post-optimization scale/position outliers eliminated before GPU validation',
               'first-frame too-few-FAISS-points behavior requires runtime validation']}
(OUT / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
