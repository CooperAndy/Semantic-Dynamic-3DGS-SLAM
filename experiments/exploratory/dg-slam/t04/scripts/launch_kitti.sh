#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/t04_chy_kitti00_200_20261012
BASE=/root/autodl-tmp/dg_slam_chy_20261011
export PATH="$BASE/01_source_env/venv_dg/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$R"
python - <<'PY'
import json,subprocess,shutil,os
from pathlib import Path
import torch,numpy as np
from scipy.spatial.transform import Rotation
r=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
assert shutil.disk_usage(r).free>5*1024**3
active='\n'.join(v.strip() for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines() if v.strip() and v.strip()!=str(os.getpid()))
assert not active, 'Other GPU compute processes active; do not overlap teammate run: '+active
p=json.loads((r/'runtime200/protocol.json').read_text());K=np.array(p['K']);assert p['gt_used'] is False
# Pixel projection equivalence after explicit independent axis scaling.
xyz=np.array([[1.,2.,10.],[-2.,-.5,30.],[0.,0.,5.]])
original=xyz[:,:2]/xyz[:,2,None]*718.856+np.array([607.1928,185.2157])
resized=xyz[:,:2]/xyz[:,2,None]*K[:2]+K[2:]
assert np.allclose(resized,original*np.array([640/1241,192/376]),atol=1e-10)
ndc=xyz[:,:2]/xyz[:,2,None]*2*K[:2]/np.array([640,192])+(2*K[2:]/np.array([640,192])-1)
assert np.allclose((ndc+1)*np.array([640,192])/2,resized,atol=1e-10)
import sys
sys.path.insert(0,str(r/'scripts'))
from r1_pose_io import matrix_to_droid_pose
from lietorch import SE3
m=np.eye(4);m[:3,:3]=Rotation.from_euler('xyz',[.1,.2,-.3]).as_matrix();m[:3,3]=[1,2,3]
assert np.allclose(SE3(matrix_to_droid_pose(m)).matrix().numpy(),m,atol=1e-6)
(r/'geometry_preflight.json').write_text(json.dumps({'status':'PASS','projection_scaling':'PASS','offcenter_projection':'PASS','xyzw_pose_roundtrip':'PASS','gpu':torch.cuda.get_device_name(),'shared_gpu_processes_before_run':active},indent=2))
PY
name="KITTI00_200_$(date -u +%Y%m%d_%H%M%S)"
printf '%s\n' "$name" > "$R/current.run_name"
set +e
timeout --signal=TERM --kill-after=30s 30m python "$R/scripts/run_kitti.py" --run-name "$name" > "$R/tracking.log" 2>&1
status=$?
printf '%s\n' "$status" > "$R/tracking.exit"
exit "$status"
