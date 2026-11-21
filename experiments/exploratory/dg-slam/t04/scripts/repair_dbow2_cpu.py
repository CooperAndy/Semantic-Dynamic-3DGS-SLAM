from pathlib import Path
import shutil,subprocess,json,hashlib
R=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012');O=R/'orb_matched_v2';S=Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2/ORB_SLAM3_4452a3c4/Thirdparty/DBoW2');D=O/'DBoW2_x86_64'
assert (O/'tracking.exit').read_text().strip()=='132'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
shutil.copytree(S,D,ignore=shutil.ignore_patterns('build','lib','.git'))
p=D/'CMakeLists.txt';p.write_text(p.read_text().replace('-march=native','-march=x86-64'))
with (O/'dbow2_build.log').open('w') as f:
    subprocess.run(['cmake','-S',str(D),'-B',str(D/'build')],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['cmake','--build',str(D/'build'),'-j','4'],stdout=f,stderr=subprocess.STDOUT,check=True)
for n in ['tracking.log','tracking.exit']:
    dst=O/('attempt02_sigill_'+n);assert not dst.exists();(O/n).rename(dst)
p=O/'run.sh';s=p.read_text();s=s.replace('export LD_LIBRARY_PATH="','export LD_LIBRARY_PATH="'+str(D/'lib')+':',1);p.write_text(s)
p=O/'preparation.json';m=json.loads(p.read_text());m['compatibility_repair']={'library':str(D/'lib/libDBoW2.so'),'sha256':hashlib.sha256((D/'lib/libDBoW2.so').read_bytes()).hexdigest(),'source':'same DBoW2 source, only -march=native -> -march=x86-64','reason':'SIGILL captured at DBoW2::FORB::distance vpopcntd; CPU lacks avx512_vpopcntdq','shared_files_modified':False};p.write_text(json.dumps(m,indent=2));print('DBoW2 CPU repair built, isolated LD_LIBRARY_PATH updated')
