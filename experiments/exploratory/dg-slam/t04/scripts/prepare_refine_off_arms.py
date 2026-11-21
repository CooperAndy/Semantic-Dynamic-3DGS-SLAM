"""Pre-register the refine-off control arms A_off and B_off.

Local only: no GPU, no network, no server connection. Verifies that the already
frozen A/B workers really expose the refine switch together with its counter
assertion, then writes a preparation receipt carrying the worker hashes and the
frozen condition table. Refuses to overwrite an existing receipt so that any
rerun is a deliberate act rather than an accident.

Counterpart protocol:
    00_任务边界与协议/精修开关对照_A_off_B_off_预登记_20261013.md
"""
from pathlib import Path
import hashlib
import json

R = Path(__file__).resolve().parents[1]
P = R / 'scripts'
OUT = R / '02_Gaussian适配审计' / '精修开关对照_20261013'

DATE = '20261013'

ARMS = {
    'A_off': {
        'worker': 'run_stage_a.py',
        'source': 'source_v3_depthguard',
        'intervention': 'Stage A depth guard only',
        'paired_on_run': 'KITTI00_200_A_on_20261012_01',
        'smoke_run': 'KITTI00_200_A_off_smoke40_%s_01' % DATE,
        'full_run': 'KITTI00_200_A_off_%s_01' % DATE,
        'installs_selector': False,
    },
    'B_off': {
        'worker': 'run_stage_b.py',
        'source': 'source_v4_depthselection',
        'intervention': 'Stage A plus B observed-depth Gaussian selection',
        'paired_on_run': 'KITTI00_200_B_on_20261012_01',
        'smoke_run': 'KITTI00_200_B_off_smoke40_%s_01' % DATE,
        'full_run': 'KITTI00_200_B_off_%s_01' % DATE,
        'installs_selector': True,
    },
}

REQUIRED_WORKER_MARKERS = [
    "--gs-refine",
    "choices=['on','off']",
    "cfg['gs_pose_refine']=a.gs_refine=='on'",
    "(counts['gs_pose_optimization_calls']>0) == cfg['gs_pose_refine']",
]

HELD_CONSTANT = [
    'KITTI00 source frames 0-199, 640x192, P2/P3 pixels, pixel-centre intrinsics',
    'original 200 timestamps, no semantic mask',
    'SGBM metric pseudo-depth cache with [0.5, 40) m admission eligibility',
    'seed 1219, 50 tracking iterations, pixel budgets, learning rates',
    'BA false, mapping depth support 40 m',
    'RTX 5090 / sm120 compatibility natives and LD_PRELOAD configuration',
    'same post-run CPU evaluator, P2->P0, SE(3), scale correction disabled',
]

ACCEPTANCE = {
    'exit_code': 0,
    'manifest_status': 'COMPLETED',
    'input_frames': 200,
    'gs_pose_optimization_calls': 0,
    'gs_mapping_calls_gt': 0,
    'gs_render_calls_gt': 0,
    'blocked_gt_accesses': 0,
    'map_finite_all_keyframes': True,
    'max_distance_current_camera_m_lt': 1000,
    'max_activated_scale_m_lt': 100,
    'smoke_keyframes_gt': 2,
}

FORBIDDEN = [
    'tuning thresholds, seeds, windows or iteration budgets to pass the criteria',
    'GT or ORB pose initialisation, injection or retro-correction',
    'substituting Sim(3) numbers for the formal SE(3) results',
    'dropping failed frames or reporting only successful segments',
    'starting stage C (20/40 m distance ablation) in this arm set',
    'touching teammate DG-SLAM sources, outputs or attribution',
]

LIMITS = [
    'KITTI00 0-199 is an exposed window; single window and single seed, DEV diagnostic only',
    'A_off and B_off are different source builds from V2_off, which is an off-axis historical anchor only',
    'off arms intentionally produce no selection_history.jsonl; the evaluator must not require it',
    'independent online runs are not bitwise paired maps, even with identical seeds',
]

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    existing = OUT / 'preparation.json'
    if existing.exists():
        raise SystemExit('Refusing to overwrite existing preparation receipt: %s' % existing)
    OUT.mkdir(parents=True, exist_ok=True)

    arms = {}
    for arm, spec in ARMS.items():
        worker_path = P / spec['worker']
        text = worker_path.read_text(encoding='utf-8')
        for marker in REQUIRED_WORKER_MARKERS:
            assert marker in text, 'worker %s lacks marker: %s' % (spec['worker'], marker)
        source_marker = "source=ROOT/'%s'" % spec['source']
        assert source_marker in text, 'worker %s does not use %s' % (spec['worker'], spec['source'])
        arms[arm] = {
            'worker': spec['worker'],
            'worker_sha256': sha(worker_path),
            'source_tree': spec['source'],
            'intervention': spec['intervention'],
            'installs_selector': spec['installs_selector'],
            'smoke_run_name': spec['smoke_run'],
            'full_run_name': spec['full_run'],
            'paired_on_run_name': spec['paired_on_run'],
            'smoke_command': 'python scripts/%s --run-name %s --frames 40 --gs-refine off'
                             % (spec['worker'], spec['smoke_run']),
            'full_command': 'python scripts/%s --run-name %s --frames 200 --gs-refine off'
                            % (spec['worker'], spec['full_run']),
        }

    matrices = {
        'V2': {'off': {'run': 'matched_v2_off', 'status': 'COMPLETED_EVALUATED'},
               'on': {'run': 'matched_v2_on', 'status': 'COMPLETED_EVALUATED'}},
        'A': {'off': {'run': arms['A_off']['full_run_name'], 'status': 'THIS_PREPARATION'},
              'on': {'run': 'KITTI00_200_A_on_20261012_01', 'status': 'COMPLETED_EVALUATED'}},
        'B': {'off': {'run': arms['B_off']['full_run_name'], 'status': 'THIS_PREPARATION'},
              'on': {'run': 'KITTI00_200_B_on_20261012_01', 'status': 'COMPLETED_EVALUATED'}},
    }

    receipt = {
        'status': 'PREPARED_NOT_RUN',
        'date_client': '2026-10-13',
        'scope': 'CHY KITTI00 0-199 refine-off control arms; single flag flip, no source change',
        'design': '3 depth-treatment variants x 2 refine switch; separates the refinement effect from the depth-patch effect',
        'matrix': matrices,
        'arms': arms,
        'only_change': "run with --gs-refine off, i.e. cfg['gs_pose_refine']=False",
        'worker_self_assertion': "assert (counts['gs_pose_optimization_calls']>0) == cfg['gs_pose_refine']",
        'held_constant': HELD_CONSTANT,
        'acceptance': ACCEPTANCE,
        'forbidden': FORBIDDEN,
        'limits': LIMITS,
        'server_root': '/root/autodl-tmp/t04_chy_kitti00_200_20261012',
        'receipt_dir': '/root/autodl-tmp/t04_chy_kitti00_200_20261012/stage_refoff',
        'deploy_requirement': 'upload scripts/run_stage_a.py, scripts/run_stage_b.py and this preparation.json; the launcher verifies SHA256 and refuses to start on mismatch',
        'launcher': 'scripts/launch_refine_off_pipeline.py',
        'evaluator': 'scripts/evaluate_refine_off.py',
        'basis': 'motivated by the completed matched V2 off/on pair, where refinement tripled the 1-frame translation RPE; no off arm has been run for A or B yet',
    }
    existing.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Prepared refine-off arms')
    for arm, spec in arms.items():
        print('  %-6s worker=%s  source=%s' % (arm, spec['worker'], spec['source_tree']))
        print('         smoke: %s' % spec['smoke_run_name'])
        print('         full : %s' % spec['full_run_name'])
    print('Receipt: %s' % existing)


if __name__ == '__main__':
    main()
