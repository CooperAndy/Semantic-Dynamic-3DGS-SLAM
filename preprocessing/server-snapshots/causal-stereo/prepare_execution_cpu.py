"""Prepare per-window causal confidence contexts and commands, without GPU execution."""
from common import *
import ast, shutil, subprocess, sys, time
import numpy as np

started = time.monotonic()
reg = json.loads((R / 'registration_cpu.json').read_text())
assert reg['status'] == 'PASS_CPU_STEREO_PIPELINE_SOURCE_WEIGHT_AND4000_IMAGE_LOCKS_PENDING_GPU'
assert not (R / 'execution_cpu_registration.json').exists()
for rel, digest in reg['source_files'].items():
    assert sha(R / rel) == digest, rel
assert sha(R / 'runtime/raftstereo-middlebury.pth') == reg['checkpoint_sha256']
commands, contracts, contexts = [], [], []
for window in reg['windows']:
    wid = window['id']
    frontend = I / 'windows' / wid
    W = R / 'windows' / wid
    assert not W.exists()
    for folder in ['scripts', 'provenance', 'receipts']:
        (W / folder).mkdir(parents=True, exist_ok=True)
    poses = np.loadtxt(window['primary_pose_path'])
    assert poses.shape == (1000, 12) and np.isfinite(poses).all()
    assert sha(window['primary_pose_path']) == window['primary_pose_sha256']
    shutil.copy2(frontend / 'input_view/calib.txt', W / 'calib.txt')
    shutil.copy2(R / 'source/confidence.py', W / 'provenance/legacy_confidence.py')
    assert sha(W / 'provenance/legacy_confidence.py') == sha(P14 / 'pipeline/confidence.py')
    for name in ['producer.py', 'contract.py']:
        shutil.copy2(R / 'templates' / name, W / 'scripts' / name)
        assert sha(W / 'scripts' / name) == sha(R / 'templates' / name)
    # Reuse the parent's audit, replacing only the task and source-frame identity.
    audit = (R / 'templates/audit_parent.py').read_text()
    assert audit.count('source_frame_id=i+3000') == 1
    audit = audit.replace('source_frame_id=i+3000', "source_frame_id=i+reg['source_start']")
    audit = audit.replace("task_id='T3-G19C-PAST-INPUT'", "task_id='T3-G21D-PAST-INPUT'")
    (W / 'scripts/audit.py').write_text(audit)
    common = (Path(__file__).parent / 'common.py').read_text()
    common += '\nP=P14\nF=I/"windows"/' + repr(wid) + '\nR=Path(' + repr(str(W)) + ')\n'
    (W / 'scripts/common.py').write_text(common)
    context = dict(window=window, calib_sha256=sha(W / 'calib.txt'),
                   cpu_registration_sha256=sha(R / 'registration_cpu.json'),
                   confidence_source_sha256=sha(W / 'provenance/legacy_confidence.py'),
                   formula_sha256=sha(W / 'scripts/producer.py'), GT_scoring=False)
    dump(W / 'cpu_context.json', context)
    for p in (W / 'scripts').glob('*.py'):
        ast.parse(p.read_bytes())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    with (W / 'receipts/contract.log').open('x') as log:
        subprocess.run([str(CPU), str(W / 'scripts/contract.py')], env=env,
                       cwd=W, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    contract = json.loads((W / 'receipts/contract.json').read_text())
    assert contract['status'] == 'PASS_KNOWN_MOVING_PLANE_FIRST_FRAME_AND_INVALID_RANGE'
    contracts.append(dict(window=wid, receipt_sha256=sha(W / 'receipts/contract.json')))
    tracebase = ['strace', '-f', '-ttt', '-yy', '-s', '256', '-e', 'trace=openat,close,write']
    for side in ['left', 'right']:
        argv = tracebase + ['-o', str(W / f'{side}_file_activity.log'), str(RAFTPY),
            str(R / f'source/{side}.py'), '--raft-root', str(R / 'source/RAFT-Stereo'),
            '--checkpoint', str(R / 'runtime/raftstereo-middlebury.pth'),
            '--sequence-dir', window['input_view'], '--output-dir', str(W / f'raw_{side}'),
            '--num-frames', '1000', '--valid-iters', '32', '--min-depth', '0.5', '--max-depth', '80']
        if side == 'left':
            argv += ['--preview-indices']
        commands.append(dict(window=wid, step=side, argv=argv, executed=False))
    contexts.append(dict(window=wid, path=str(W), files={
        p.relative_to(W).as_posix(): sha(p) for p in W.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}))
dump(R / 'execution_cpu_registration.json', dict(
    status='PASS_TWO_WINDOW_CPU_CONTEXTS_AND_MOVING_PLANE_CONTRACTS_GPU_NOT_EXECUTED',
    at=now(), cpu_registration_sha256=sha(R / 'registration_cpu.json'),
    contexts=contexts, contracts=contracts, planned_stereo_commands=commands,
    unchanged_producer_and_contract_bytes=True,
    independent_full_stereo_trace_audit_and_runtime_controller_still_required=True,
    GPU_calls=0, GT_scoring=False, quality_adoption=False, seconds=time.monotonic()-started))
print(json.dumps(dict(status='PASS_TWO_WINDOW_CPU_CONTEXTS', windows=len(contexts),
                     GPU_calls=0, seconds=time.monotonic()-started)), flush=True)
