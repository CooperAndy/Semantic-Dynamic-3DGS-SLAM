"""Build a private rasterizer with FP64 accumulation of FP32 contributions.

The public Python ABI, forward pass, per-pixel derivative expressions,
Gaussian model and optimizer are unchanged. This is a registered numerical
candidate, not a waiver of either failed repeatability gate.
"""
import hashlib,json,shutil,subprocess,os
from pathlib import Path
from experiment import REMOTE,PARENT,PYTHON

src=PARENT/'raster_sm86'
dst=REMOTE/'raster_fp64_accum_v1'
assert not dst.exists()
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('build','*.so','*.egg-info','__pycache__','.git'))
before={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and p.suffix in ['.cu','.h','.cpp','.py'] and 'build' not in p.parts}

def change(name,old,new,count=1):
    p=dst/name;s=p.read_text();assert s.count(old)==count,(name,old,s.count(old))
    p.write_text(s.replace(old,new))

# Scratch is owned by Torch's caching allocator and scoped to this backward.
change('rasterize_points.cu','  if(P != 0)\n  {  \n\t  CudaRasterizer::Rasterizer::backward',
       '  auto accumulation = torch::zeros({P, 8 + NUM_CHANNELS}, means3D.options().dtype(torch::kFloat64));\n  if(P != 0)\n  {  \n\t  CudaRasterizer::Rasterizer::backward')
change('rasterize_points.cu','dL_drotations.contiguous().data<float>());','dL_drotations.contiguous().data<float>(),\n      accumulation.data_ptr<double>());')
change('cuda_rasterizer/rasterizer.h','float* dL_drot);','float* dL_drot, double* accumulation);')
change('cuda_rasterizer/rasterizer_impl.cu','float* dL_drot)\n{','float* dL_drot, double* accumulation)\n{')
change('cuda_rasterizer/rasterizer_impl.cu','\tBACKWARD::render(\n\t\ttile_grid,','\tBACKWARD::render(\n        P, accumulation,\n\t\ttile_grid,')
change('cuda_rasterizer/backward.h','\tvoid render(\n','\tvoid render(\n        int P, double* accumulation,\n')
change('cuda_rasterizer/backward.cu','void BACKWARD::render(\n','void BACKWARD::render(\n    int P, double* accumulation,\n')

# Only renderCUDA's accumulator arguments become double; preprocess stays FP32.
p=dst/'cuda_rasterizer/backward.cu';s=p.read_text()
a=s.index('renderCUDA(');b=s.index('void BACKWARD::preprocess(')
body=s[a:b]
body=body.replace('float3* __restrict__ dL_dmean2D','double* __restrict__ dL_dmean2D')
body=body.replace('float4* __restrict__ dL_dconic2D','double* __restrict__ dL_dconic2D')
body=body.replace('float* __restrict__ dL_dopacity','double* __restrict__ dL_dopacity')
body=body.replace('float* __restrict__ dL_dcolors','double* __restrict__ dL_dcolors')
for field,component in [('x',0),('y',1)]:
    body=body.replace('dL_dmean2D[global_id].'+field,'dL_dmean2D[global_id * 3 + '+str(component)+']')
for field,component in [('x',0),('y',1),('w',3)]:
    body=body.replace('dL_dconic2D[global_id].'+field,'dL_dconic2D[global_id * 4 + '+str(component)+']')
assert body.count('atomicAdd(')==7
s=s[:a]+body+s[b:]
at=s.index('void BACKWARD::render(')
converter='''// FP32 output ABI; one independently owned result per Gaussian.
__global__ void castAccumulators(int P, const double* scratch,
    float3* mean, float4* conic, float* opacity, float* color) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= P) return;
    mean[i] = make_float3(float(scratch[i*3]), float(scratch[i*3+1]), float(scratch[i*3+2]));
    const double* c = scratch + P*3;
    conic[i] = make_float4(float(c[i*4]), float(c[i*4+1]), float(c[i*4+2]), float(c[i*4+3]));
    opacity[i] = float(scratch[P*7+i]);
    for(int j=0; j<NUM_CHANNELS; ++j) color[i*NUM_CHANNELS+j] = float(scratch[P*8+i*NUM_CHANNELS+j]);
}

'''
s=s[:at]+converter+s[at:]
old='''\t\tdL_dmean2D,
\t\tdL_dconic2D,
\t\tdL_dopacity,
\t\tdL_dcolors
\t\t);'''
assert s.count(old)==1
s=s.replace(old,'''        accumulation,
        accumulation + P*3,
        accumulation + P*7,
        accumulation + P*8
        );
    castAccumulators<<<(P+255)/256, 256>>>(P, accumulation,
        dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors);''')
p.write_text(s)
registration=dict(task='S23_FP64_GRADIENT_ACCUMULATION_CONTROL',status='REGISTERED_BUILD',
    source=str(src),candidate=str(dst),original_sources_sha256=before,
    changed_factor='FP64 atomic accumulation of unchanged FP32 pixel derivative contributions, cast once to FP32',
    additional_scratch_bytes_per_gaussian=8*(8+3),forward_unchanged=True,optimizer_unchanged=True,
    thresholds_changed=False,GT_opened=False,
    controls_required=['fixed-map forward equality','gradient agreement with original FP32 implementation',
        'repeated backward hashes','six fresh online 25-frame runs under original E0 limits'],
    caveat='FP64 atomics are not claimed mathematically order independent; verify actual repeatability')
(dst/'registration.json').write_text(json.dumps(registration,indent=2))
env=dict(os.environ,CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='2')
with (dst/'build.log').open('w') as log:
    proc=subprocess.run([PYTHON,'setup.py','build_ext','--inplace'],cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT)
registration.update(status='BUILD_COMPLETE' if proc.returncode==0 else 'BUILD_FAILED',returncode=proc.returncode,
    modified_sources_sha256={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*') if p.is_file() and p.suffix in ['.cu','.h','.cpp','.py'] and 'build' not in p.parts})
(dst/'registration.json').write_text(json.dumps(registration,indent=2))
assert proc.returncode==0
print(json.dumps(dict(status=registration['status'],candidate=str(dst))))
