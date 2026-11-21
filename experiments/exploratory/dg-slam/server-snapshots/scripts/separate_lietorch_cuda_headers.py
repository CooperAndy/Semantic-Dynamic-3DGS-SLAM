"""Keep Torch tensor wrappers in C++; compile unchanged CUDA kernels without Torch headers."""
from pathlib import Path
import re
import json
import hashlib
import difflib

R=Path('/root/autodl-tmp/dg_slam_chy_20261011')
root=R/'01_source_env/dependencies/DROID-SLAM/thirdparty/lietorch'
src=root/'lietorch/src'
original=(src/'lietorch_gpu.cu').read_text()
matches=list(re.finditer(r'^(?:torch::Tensor|std::vector<torch::Tensor>)\s+(\w+)\(',original,re.M))
assert len(matches)==19
kernel_prefix=original[:matches[0].start()]
signatures={m.group(1):m.group(2) for m in re.finditer(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)',kernel_prefix)}
assert len(signatures)==19
device_prefix=kernel_prefix.replace('#include "lietorch_gpu.h"','#include <cuda_runtime.h>').replace('#include "dispatch.h"','// Torch dispatch is compiled in separate host translation units.')
assert 'torch/' not in device_prefix and 'torch::' not in device_prefix
(src/'gpu_device_kernels.h').write_text('#pragma once\n'+device_prefix)
declarations=['#pragma once','#include "common.h"','#include "so3.h"','#include "rxso3.h"','#include "se3.h"','#include "sim3.h"']
for name,signature in signatures.items():declarations.append(f'template <typename Group, typename scalar_t> void launch_{name}({signature});')
(src/'gpu_launchers.h').write_text('\n'.join(declarations)+'\n')
manifest=[]
for i,m in enumerate(matches):
    name=m.group(1)
    block=original[m.start():matches[i+1].start() if i+1<len(matches) else len(original)]
    pattern=r'(\w+_kernel)<group_t,\s*scalar_t>\s*<<<NUM_BLOCKS\(batch_size\),\s*NUM_THREADS>>>\s*\('
    calls=re.findall(pattern,block)
    assert len(calls)==1
    kernel=calls[0]
    host,n=re.subn(pattern,lambda match:f'launch_{match.group(1)}<group_t, scalar_t>(',block)
    assert n==1
    old=src/f'split_gpu/{name}.cu'
    if old.exists():old.rename(old.with_suffix('.cu.before_host_split'))
    hostfile=src/f'split_gpu/{name}.cpp'
    hostfile.write_text('#include "lietorch_gpu.h"\n#include "dispatch.h"\n#include "../gpu_launchers.h"\n'+host)
    signature=signatures[kernel]
    argument_names=[re.search(r'(\w+)\s*$',part.strip()).group(1) for part in signature.split(',')]
    launcher=f'template <typename Group, typename scalar_t>\nvoid launch_{kernel}({signature}) {{\n    {kernel}<Group,scalar_t><<<NUM_BLOCKS({argument_names[-1]}), NUM_THREADS>>>({", ".join(argument_names)});\n}}\n'
    for group in ['SO3','RxSO3','SE3','Sim3']:
        for dtype in ['float','double']:
            launcher+=f'template void launch_{kernel}<{group}<{dtype}>,{dtype}>({signature.replace("scalar_t",dtype)});\n'
    cudafile=src/f'split_gpu/{name}_launch.cu'
    cudafile.write_text('#include "../gpu_device_kernels.h"\n#include "../gpu_launchers.h"\n'+launcher)
    manifest.append({'operation':name,'kernel':kernel,'original_host_wrapper_sha256':hashlib.sha256(block.encode()).hexdigest(),'replacement':'Identical launch arguments / grid / block; tensor allocation and dispatch remain in host C++','groups':['SO3','RxSO3','SE3','Sim3'],'dtypes':['float','double']})
setup=root/'setup.py'
before=setup.read_text()
after=before.replace("*sorted(glob.glob('lietorch/src/split_gpu/*.cu'))","*sorted(glob.glob('lietorch/src/split_gpu/*.cpp')),\n                *sorted(glob.glob('lietorch/src/split_gpu/*.cu'))")
assert after!=before
setup.write_text(after)
(R/'01_source_env/patches/lietorch_cuda_host_separation.json').write_text(json.dumps({'original_kernel_prefix_sha256':hashlib.sha256(kernel_prefix.encode()).hexdigest(),'device_kernel_bodies_unchanged':True,'operations':manifest},indent=2))
(R/'01_source_env/patches/lietorch_cuda_host_setup.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='lietorch/setup.py.before_host_split',tofile='lietorch/setup.py')))
print('19 CUDA launchers; all kernel bodies / arguments / grid / block preserved; no Torch headers in device units')
