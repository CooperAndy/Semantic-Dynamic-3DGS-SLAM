"""Validate pinned wheels and record GPU code before installing (no GPU)."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

R=Path('/root/autodl-tmp/dg_slam_chy_20261011')
W=R/'01_source_env/dependencies/wheels'
EXPECTED={'torch_scatter-2.1.2+pt28cu128-cp312-cp312-linux_x86_64.whl':'372005c58c8ab7a96c8e971abe008bfbe5933f5c526d5d3419a48482381d1d47'}
records=[]
for name,digest in EXPECTED.items():
    wheel=W/name
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()==digest
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.endswith('_cuda.so'):
                z.extract(name,W/'inspect')
                so=W/'inspect'/name
                outputs=[]
                for flag in ('--list-elf','--list-ptx'):
                    result=subprocess.run(['/usr/local/cuda/bin/cuobjdump',flag,str(so)],capture_output=True,text=True)
                    if result.returncode:
                        assert so.name=='_version_cuda.so' and result.returncode==255 and 'does not contain device code' in result.stderr, result.stderr
                    outputs.append(result.stdout)
                elf,ptx=outputs
                if elf.strip():
                    assert ptx.strip(),f'No native sm_120 or forward-compatible PTX: {name}'
                records.append({'library':name,'elf':elf.splitlines(),'ptx':ptx.splitlines(),'runtime_validation':'GPU_PENDING'})
(R/'05_reports/cpu_prepare/scatter_wheel_architecture.json').write_text(json.dumps({'wheel_sha256':digest,'libraries':records,'compatibility_basis':'NVIDIA Blackwell Compatibility Guide: ordinary PTX is forward compatible','gpu_execution_verified':False},indent=2))
print('PINNED_SCATTER_WHEEL_HASH_AND_PTX_PASS')
