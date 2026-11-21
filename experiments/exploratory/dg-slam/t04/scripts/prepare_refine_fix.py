"""Build the refine-objective-fix mapper patch and the C-arm worker.

Local only: no GPU, no network. Derives the patched mapper from the frozen Stage A+B
mapper by exact string replacement with occurrence assertions, then derives the worker
from the frozen Stage B worker the same way. Everything is ast-parsed before it is
written, and the preparation receipt is never overwritten.

Counterpart protocol:
    00_任务边界与协议/精修目标函数修复_C_20261013.md
"""
from pathlib import Path
import ast
import hashlib
import json

R = Path(__file__).resolve().parents[1]
P = R / 'scripts'
AUDIT = R / '02_Gaussian适配审计'
BASE_MAPPER = AUDIT / 'GS_SLAM借鉴_20261012/stage_b_patch/gs_tracking_mapping.py'
BASE_MAPPER_SHA = '77da7b3bc3b065639908aa837cc35f7e8bf656f6924517314dfd750acc9097b5'
OUT = AUDIT / '精修目标函数修复_20261013'
PATCHED = OUT / 'patched/dg_slam/gs_tracking_mapping.py'

SOURCE_TREE = 'source_v5_fixB'
RUN_STEM = 'KITTI00_200_C_on_20261013_01'
SMOKE_STEM = 'KITTI00_200_C_smoke40_20261013_01'

THRESHOLDS = {
    'min_frozen_support_px': 128,
    'step_cap_translation_m': 0.005,
    'step_cap_rotation_deg': 0.05,
}

MAPPER_EDITS = [
    (
        'imports',
        'from depth_consistent_selector import InsufficientTrackingSupport\nimport os\n',
        'from depth_consistent_selector import InsufficientTrackingSupport\nimport json\nimport math\nimport os\n',
    ),
    (
        'init_state',
        '        self.scale_factor = 1.0\n',
        '        self.scale_factor = 1.0\n'
        '        self._t04_frozen_mask = None\n'
        '        self._t04_frozen_support = 1\n'
        '        self._t04_keyframe = -1\n'
        '        self._t04_clamped_steps = 0\n',
    ),
    (
        'freeze_and_normalise',
        '        geo_loss = torch.clamp((torch.abs(gt_depth-depth)), min=0.0, max=1e3)[mask].sum()\n'
        '        loss = self.w_geo_loss_tracking * geo_loss\n'
        '        color_loss = torch.abs(gt_color - image)[mask].sum()\n'
        '        if self.use_color_in_tracking:\n'
        '            loss += self.w_color_loss_tracking*color_loss\n',
        '        support_scale = 1\n'
        "        if self.cfg.get('t04_freeze_support', False):\n"
        '            # Fix 1: freeze the evaluated pixel set at the coarse pose.\n'
        '            # The original objective summed the residual over a mask that itself\n'
        '            # depends on the rendered opacity, so a pose could lower the loss simply\n'
        '            # by shrinking the set it is scored on. A frozen set removes that route.\n'
        '            if self._t04_frozen_mask is None:\n'
        '                self._t04_frozen_mask = mask.detach().clone()\n'
        '                self._t04_frozen_support = int(self._t04_frozen_mask.sum().item())\n'
        "                if self._t04_frozen_support < self.cfg['t04_min_frozen_support']:\n"
        '                    # Fix 2: minimum coverage guard. Raised as the same exception the\n'
        '                    # Stage B selector already catches, so the caller retains the\n'
        '                    # DROID coarse pose instead of refining on too few pixels.\n'
        '                    raise InsufficientTrackingSupport(\n'
        "                        'frozen_support_below_floor:' + str(self._t04_frozen_support))\n"
        "                print('T04_REFINE_FIX ' + json.dumps({'event': 'support_frozen',\n"
        "                    'keyframe': self._t04_keyframe,\n"
        "                    'frozen_support': self._t04_frozen_support,\n"
        "                    'pre_freeze_support': int(mask.sum().item())}), flush=True)\n"
        '            else:\n'
        '                mask = self._t04_frozen_mask\n'
        '            support_scale = self._t04_frozen_support\n'
        '        geo_loss = torch.clamp((torch.abs(gt_depth-depth)), min=0.0, max=1e3)[mask].sum()\n'
        '        # Fix 1 continued: express the objective per evaluated pixel. Adam is invariant\n'
        '        # to a constant loss rescaling, so with a frozen support_scale this changes the\n'
        '        # printed loss but not the update magnitude.\n'
        '        loss = self.w_geo_loss_tracking * geo_loss / support_scale\n'
        '        color_loss = torch.abs(gt_color - image)[mask].sum()\n'
        '        if self.use_color_in_tracking:\n'
        '            loss += self.w_color_loss_tracking*color_loss / support_scale\n',
    ),
    (
        'step_cap_call',
        '        loss.backward()\n'
        '        optimizer.step()\n'
        '        optimizer.zero_grad()\n'
        '        self.opacity_mask = opacity_mask\n',
        '        pose_before = (self.quad.detach().clone(), self.T.detach().clone())'
        ' if self.separate_LR else None\n'
        '        loss.backward()\n'
        '        optimizer.step()\n'
        '        optimizer.zero_grad()\n'
        "        if self.separate_LR and self.cfg.get('t04_step_cap', False):\n"
        '            # Fix 3: clamp one refinement step to the registered budget.\n'
        '            self._t04_apply_step_cap(pose_before)\n'
        '        self.opacity_mask = opacity_mask\n',
    ),
    (
        'step_cap_method',
        '    def convert_relative_pose(self, idx):\n',
        '    def _t04_apply_step_cap(self, pose_before):\n'
        '        """Clamp one refinement step to the registered translation/rotation budget."""\n'
        '        quad_before, translation_before = pose_before\n'
        '        with torch.no_grad():\n'
        '            quad_after = self.quad.detach()\n'
        '            translation_after = self.T.detach()\n'
        '            translation_delta = float(torch.linalg.vector_norm(\n'
        '                translation_after - translation_before).item())\n'
        '            dot = float(torch.clamp(torch.dot(quad_before, quad_after),\n'
        '                                    min=-1.0, max=1.0).item())\n'
        '            rotation_delta_deg = math.degrees(2.0 * math.acos(abs(dot)))\n'
        "            translation_cap = float(self.cfg['t04_step_cap_translation_m'])\n"
        "            rotation_cap = float(self.cfg['t04_step_cap_rotation_deg'])\n"
        '            ratio = 1.0\n'
        '            if translation_delta > translation_cap:\n'
        '                ratio = min(ratio, translation_cap / translation_delta)\n'
        '            if rotation_delta_deg > rotation_cap:\n'
        '                ratio = min(ratio, rotation_cap / rotation_delta_deg)\n'
        '            if ratio < 1.0:\n'
        '                aligned = quad_after if dot >= 0 else -quad_after\n'
        '                blended = quad_before + (aligned - quad_before) * ratio\n'
        '                self.quad.data.copy_(blended / torch.linalg.vector_norm(blended))\n'
        '                self.T.data.copy_(translation_before\n'
        '                                  + (translation_after - translation_before) * ratio)\n'
        '                self._t04_clamped_steps += 1\n'
        "                print('T04_REFINE_FIX ' + json.dumps({'event': 'step_clamped',\n"
        "                    'keyframe': self._t04_keyframe,\n"
        "                    'translation_m': translation_delta,\n"
        "                    'rotation_deg': rotation_delta_deg,\n"
        "                    'ratio': ratio}), flush=True)\n"
        '\n'
        '    def convert_relative_pose(self, idx):\n',
    ),
    (
        'reset_per_keyframe',
        '            candidate_cam_tensor = None\n'
        '            current_min_loss = float(1e20)\n',
        '            self._t04_frozen_mask = None\n'
        '            self._t04_frozen_support = 1\n'
        '            self._t04_keyframe = int(idx)\n'
        '            candidate_cam_tensor = None\n'
        '            current_min_loss = float(1e20)\n',
    ),
]

NEW_CFG_KEYS = (
    "    cfg['gs_pose_refine']=a.gs_refine=='on'\n"
    '    cfg.update(t04_freeze_support=True, t04_min_frozen_support=%d, t04_step_cap=True,\n'
    '               t04_step_cap_translation_m=%s, t04_step_cap_rotation_deg=%s)\n'
    % (THRESHOLDS['min_frozen_support_px'],
       THRESHOLDS['step_cap_translation_m'],
       THRESHOLDS['step_cap_rotation_deg'])
)

WORKER_EDITS = [
    ('source_tree',
     "    source=ROOT/'source_v4_depthselection'\n",
     "    source=ROOT/'%s'\n" % SOURCE_TREE),
    ('cfg_keys',
     "    cfg['gs_pose_refine']=a.gs_refine=='on'\n",
     NEW_CFG_KEYS),
    ('intervention',
     "intervention='Stage A plus B observed-depth Gaussian selection'",
     "intervention='Stage A + B selection plus refine objective fix: "
     "frozen support, minimum frozen support guard, per-step pose cap'"),
    ('protocol_meta',
     "            'selector_sha256':sha256(Path(__file__).with_name('depth_consistent_selector.py'))}",
     "            'selector_sha256':sha256(Path(__file__).with_name('depth_consistent_selector.py')),\n"
     "            'refine_objective_fix':{'frozen_support':True,\n"
     "                'min_frozen_support':%d,'step_cap':True,\n"
     "                'step_cap_translation_m':%s,'step_cap_rotation_deg':%s,\n"
     "                'preparation_sha256':sha256(ROOT/'stage_refinefix/preparation.json'),\n"
     "                'basis':'one refinement step may move at most ~0.7 percent of the "
     "inter-frame translation (0.72 m) and ~9 percent of the observed adjacent-keyframe "
     "rotation (0.537 deg)'}}"
     % (THRESHOLDS['min_frozen_support_px'],
        THRESHOLDS['step_cap_translation_m'],
        THRESHOLDS['step_cap_rotation_deg'])),
]


def apply(text, edits, label):
    for name, old, new in edits:
        count = text.count(old)
        assert count == 1, '%s: anchor %s matched %d times' % (label, name, count)
        text = text.replace(old, new)
    return text


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    receipt_path = OUT / 'preparation.json'
    if receipt_path.exists():
        raise SystemExit('Refusing to overwrite existing preparation receipt: %s' % receipt_path)

    base = BASE_MAPPER.read_text(encoding='utf-8')
    assert sha(BASE_MAPPER) == BASE_MAPPER_SHA, 'Stage B mapper base hash changed'
    patched = apply(base, MAPPER_EDITS, 'mapper')
    ast.parse(patched)
    PATCHED.parent.mkdir(parents=True, exist_ok=True)
    PATCHED.write_text(patched, encoding='utf-8')

    worker_src = (P / 'run_stage_b.py').read_text(encoding='utf-8')
    worker = apply(worker_src, WORKER_EDITS, 'worker')
    ast.parse(worker)
    (P / 'run_stage_c.py').write_text(worker, encoding='utf-8')

    receipt = {
        'status': 'PREPARED_NOT_RUN',
        'date_client': '2026-10-13',
        'scope': 'CHY KITTI00 0-199 refine objective fix arm C; refine ON only',
        'design': 'single-factor counterpart to B_on: same Stage A+B source, only the refinement objective changes',
        'source_tree': SOURCE_TREE,
        'parent_sha256': {'dg_slam/gs_tracking_mapping.py': BASE_MAPPER_SHA},
        'patched_sha256': {'dg_slam/gs_tracking_mapping.py': sha(PATCHED)},
        'worker_sha256': sha(P / 'run_stage_c.py'),
        'thresholds': THRESHOLDS,
        'changes': [
            'Fix 1 fused support: the evaluated pixel set is frozen at the coarse pose for the whole refinement and reused for all iterations',
            'Fix 1 logging: the objective is also expressed per evaluated pixel over the frozen support; Adam is invariant to a constant loss rescaling so the update magnitude is unchanged',
            'Fix 2 minimum coverage guard: frozen support below %d px raises InsufficientTrackingSupport, which the Stage B selector already catches and converts into a fallback to the DROID coarse pose' % THRESHOLDS['min_frozen_support_px'],
            'Fix 3 per-step pose cap: one refinement step may not exceed %s m translation or %s deg rotation; an overshooting step is interpolated back and counted' % (THRESHOLDS['step_cap_translation_m'], THRESHOLDS['step_cap_rotation_deg']),
            'per-keyframe state reset plus T04_REFINE_FIX json lines on stdout for freezing and clamping',
        ],
        'unchanged': [
            'Stage A depth admission guard', 'Stage B observed-depth selector and its fallback',
            'DROID front end', '50 tracking iterations', 'learning rates', 'BA false',
            'mapping depth support 40 m', 'seed 1219', '640x192 inputs and byte-identical source frames',
        ],
        'comparison': 'C_on versus B_on (identical source tree, refinement objective not fixed) and versus the three refine-off floor arms',
        'acceptance': {
            'exit_code': 0,
            'manifest_status': 'COMPLETED',
            'input_frames': 200,
            'gs_pose_optimization_calls_gt': 0,
            'blocked_gt_accesses': 0,
            'map_finite_all_keyframes': True,
            'frozen_support_constant_within_a_keyframe': 'pixel_support_min == pixel_support_max in selection_history.jsonl',
        },
        'limits': [
            'single exposed DEV window and single seed; DEV diagnostic only',
            'the step cap is a post-step projection, so the Adam state is not reset after a clamp',
            'freezing the support also freezes the pixels that invalidate after a large pose change; an aggressive cap can therefore bind',
            'the thresholds are registered engineering values, not tuned on this window',
        ],
        'deploy': 'scripts/deploy_refine_fix.py on the server builds %s from source_v4_depthselection' % SOURCE_TREE,
        'launcher': 'scripts/launch_refine_fix_pipeline.py',
        'run_names': {'smoke40': SMOKE_STEM, 'full200': RUN_STEM},
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Prepared refine objective fix arm C')
    print('  patched mapper sha256: %s' % receipt['patched_sha256']['dg_slam/gs_tracking_mapping.py'])
    print('  worker sha256        : %s' % receipt['worker_sha256'])
    print('  source tree          : %s' % SOURCE_TREE)
    print('  runs                 : %s / %s' % (SMOKE_STEM, RUN_STEM))
    print('  receipt              : %s' % receipt_path)


if __name__ == '__main__':
    main()
