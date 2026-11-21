"""Exercise NumPy's real tofile/duplicate-FD path in the strace sealing audit."""
from common import *
from audit_stereo_trace import audit_trace
import subprocess, sys
import numpy as np

D=R/'trace_numpy_contract';assert not D.exists();D.mkdir();probe=D/'probe.py'
probe.write_text('''from pathlib import Path
import numpy as np
D=Path(__file__).parent;I=D.parent/'trace_contract'
for i in range(3):
 for side in ['left','right']:(I/side/f'{i:06d}.png').read_bytes()
 np.save(D/f'{i:06d}_depth.npy',np.arange(128*1024,dtype=np.float32).reshape(128,1024),allow_pickle=False)
''')
trace=D/'numpy.strace'
subprocess.run(['strace','-f','-ttt','-yy','-s','256','-e','trace=openat,close,write','-o',str(trace),sys.executable,str(probe)],check=True,timeout=30)
inputs={side:[R/'trace_contract'/side/f'{i:06d}.png' for i in range(3)] for side in ['left','right']}
proof=audit_trace(trace,inputs,D,3)
for i in range(3):
    a=np.load(D/f'{i:06d}_depth.npy',allow_pickle=False)
    assert a.dtype==np.float32 and a.shape==(128,1024) and np.array_equal(a.ravel(),np.arange(128*1024,dtype=np.float32))
dump(R/'trace_numpy_contract_receipt.json',dict(status='PASS_REAL_NUMPY_TOFILE_WRITE_CLOSE_TRACE_AND_ARRAY_BYTES',
    at=now(),proof=proof,dtype='float32',shape=[128,1024],synthetic_only=True,
    GPU_calls=0,actual_dataset_GT_reads=0,script_sha256=sha(__file__),trace_sha256=sha(trace)))
print('PASS_REAL_NUMPY_TOFILE_TRACE',flush=True)
