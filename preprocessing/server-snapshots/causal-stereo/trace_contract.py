"""Exercise the causal trace auditor against real strace output, including negative cases."""
from common import *
from audit_stereo_trace import audit_trace
import subprocess, sys, time

start=time.monotonic();D=R/'trace_contract';assert not D.exists();D.mkdir()
inputs={side:[] for side in ['left','right']}
for side in inputs:
    (D/side).mkdir()
    for i in range(3):
        p=D/side/f'{i:06d}.png';p.write_bytes(b'synthetic image fixture');inputs[side].append(p)
(D/'poses').mkdir();(D/'poses/08.txt').write_text('synthetic forbidden fixture, not dataset GT')
probe=D/'probe.py'
probe.write_text('''from pathlib import Path
import sys
D=Path(__file__).parent;case=sys.argv[1];out=D/case;out.mkdir()
for i in range(3):
 for side in ['left','right']:(D/side/f'{i:06d}.png').read_bytes()
 if i==0 and case=='future':(D/'left/000002.png').read_bytes()
 if i==0 and case=='forbidden':(D/'poses/08.txt').read_text()
 (out/f'{i:06d}_depth.npy').write_bytes(b'completed output fixture')
''')
proofs=[]
for case in ['valid','future','forbidden']:
    trace=D/(case+'.strace')
    subprocess.run(['strace','-f','-ttt','-yy','-s','256','-e','trace=openat,close,write','-o',str(trace),sys.executable,str(probe),case],check=True,timeout=30)
    try:
        result=audit_trace(trace,inputs,D/case,3)
    except AssertionError as err:
        assert case!='valid',repr(err)
        result=dict(rejected=True,reason=repr(err))
    else:
        assert case=='valid',('Negative case escaped',case)
    proofs.append(dict(case=case,result=result,trace_sha256=sha(trace)))
# Explicitly cover openat entry/resume across an output seal: reject at entry.
fixture=D/'unfinished_future.strace'
fixture.write_text(f'123 1.0 openat(AT_FDCWD, "{inputs["left"][2]}", O_RDONLY|O_CLOEXEC <unfinished ...>\n')
try:audit_trace(fixture,inputs,D/'valid',3)
except AssertionError as err:
    assert 'Future image open' in str(err)
else:raise AssertionError('Unfinished future-open fixture escaped')
dump(R/'trace_contract_receipt.json',dict(status='PASS_REAL_STRACE_VALID_FUTURE_GT_AND_UNFINISHED_OPEN_CONTRACTS',
    at=now(),cases=proofs,unfinished_future_rejected_at_entry=True,synthetic_only=True,
    actual_dataset_GT_reads=0,GPU_calls=0,seconds=time.monotonic()-start))
print('PASS_REAL_STRACE_CONTRACTS',flush=True)
