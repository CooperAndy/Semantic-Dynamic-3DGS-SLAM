"""Build native code without keeping the 500MB torch Python builder resident."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import json

PATCH = '''
# CPU_PREP_NINJA_ONLY: emit build rules, then release Python/torch RAM.
if os.environ.get('DG_GENERATE_NINJA_ONLY') == '1':
    import json as _json
    import subprocess as _subprocess
    from pathlib import Path as _Path
    import torch.utils.cpp_extension as _ce
    _original_ninja = _ce._run_ninja_build
    def _generate_only(build_directory, verbose, error_prefix):
        probe = _subprocess.run(['ninja', '-C', build_directory, '-n', '-j1'], capture_output=True, text=True)
        if probe.returncode or 'no work to do' not in probe.stdout:
            _Path(os.environ['DG_NINJA_MARKER']).write_text(_json.dumps({'directory':build_directory,'probe':probe.stdout[-4000:]}))
            raise RuntimeError('DG_NINJA_EMITTED_RUN_STANDALONE')
        return _original_ninja(build_directory, verbose, error_prefix)
    _ce._run_ninja_build = _generate_only
'''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('--probe-gpu-first',action='store_true')
    p.add_argument('--only-probe',action='store_true')
    a=p.parse_args()
    source=Path(a.source).resolve()
    setup=source/'setup.py'
    before=setup.read_text()
    if 'CPU_PREP_NINJA_ONLY' not in before:
        import difflib
        assert '\nsetup(' in before
        after=before.replace('\nsetup(', '\nimport os\n'+PATCH+'\nsetup(',1)
        setup.write_text(after)
        patchdir=Path('/root/autodl-tmp/dg_slam_chy_20261011/01_source_env/patches')
        (patchdir/f'{source.name}_standalone_setup.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(setup)+'.before_standalone',tofile=str(setup))))
    marker=source/'ninja_pending.json'
    env=os.environ.copy()
    env.update(DG_GENERATE_NINJA_ONLY='1', DG_NINJA_MARKER=str(marker), MAX_JOBS='1')
    log=source/'standalone_setup.log'
    for attempt in range(6):
        if marker.exists(): marker.unlink()
        with log.open('a') as f:
            result=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode==0:
            print('STANDALONE_BUILD_COMPLETE',source,flush=True)
            subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps',str(source)],check=True)
            return
        if not marker.exists():
            print(log.read_text()[-12000:]);raise RuntimeError('setup failed before generating standalone build rules')
        info=json.loads(marker.read_text())
        builddir=Path(info['directory'])
        print('STANDALONE_COMPILE',source,builddir,flush=True)
        if a.probe_gpu_first and attempt==0:
            ninja=(builddir/'build.ninja').read_text()
            needle='act4_backward_gpu_launch.o:' if 'act4_backward_gpu_launch.o:' in ninja else 'act4_backward_gpu.o:'
            target=next(line.split(':',1)[0][6:] for line in ninja.splitlines() if line.startswith('build ') and needle in line)
            subprocess.run(['ninja','-C',str(builddir),'-j1',target],check=True)
            print('FIRST_CUDA_OBJECT_PASSED_WITHOUT_TORCH_PARENT',flush=True)
            if a.only_probe: return
        subprocess.run(['ninja','-C',str(builddir),'-j1'],check=True)
    raise RuntimeError('Too many extension build stages')

if __name__=='__main__':main()
