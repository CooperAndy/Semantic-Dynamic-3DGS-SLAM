from common import *
import shutil,subprocess,shlex,csv,collections,struct
original=json.loads((P/'input_registration.json').read_text());root=Path(original['vocabulary']).parent.parent
for name in ['runtime/lib','inputs','provenance/api','build','bin','output']:(R/name).mkdir(parents=True)
dump(R/'preparation_registration.json',dict(at=now(),task='T3-G19-INPUT',window=original['window'],protocol_sha256=sha(R/'PROTOCOL.md'),cpp_sha256=sha(R/'causal_stereo.cc'),GT_allowed=False,parameters='ReuseG14 exact settings/core/vocabulary;O0',budget=dict(build_seconds=600,run_seconds=1800,new_bytes=2**30,free_bytes=6*2**30)))
shutil.copy2(P/'orb/settings.yaml',R/'inputs/settings.yaml');shutil.copy2(P/'input_view/times.txt',R/'inputs/times.txt');shutil.copy2(P/'input_view/calib.txt',R/'inputs/calib.txt');shutil.copy2(P/'orb/frame_features.csv',R/'inputs/old_frame_features.csv');shutil.copy2(root/'LICENSE',R/'provenance/ORB_LICENSE');shutil.copy2(original['vocabulary'],R/'runtime/ORBvoc.txt');assert sha(R/'runtime/ORBvoc.txt')==original['vocabulary_sha256']
for relative in ['include/System.h','include/Tracking.h','src/System.cc']:
 target=R/'provenance/api'/Path(relative).name;shutil.copy2(root/relative,target)
old=list(csv.DictReader((R/'inputs/old_frame_features.csv').open()));assert len(old)==1000 and [int(x['frame_id']) for x in old]==list(range(1000));dump(R/'old_frontend_ledger.json',dict(status='RECOVERED_EXISTING_INPUT_FRONTEND_STATES',rows=1000,states=dict(collections.Counter(x['tracking_state'] for x in old)),source_sha256=sha(P/'orb/frame_features.csv'),enum_OK=2,scope='Old shared frontend state ledger;does not turn finalBA poses into causal input'))
linked=subprocess.run(['ldd',str(P/'pipeline/stereo_kitti_headless')],env=dict(os.environ,LD_LIBRARY_PATH=':'.join(original['library_paths'])),capture_output=True,text=True,check=True);(R/'provenance/old_binary_ldd.txt').write_text(linked.stdout);libraries=[]
for line in linked.stdout.splitlines():
 if ' => ' not in line:continue
 name=line.split(' => ',1)[1].split(' (',1)[0].strip();p=Path(name)
 if p.is_file() and p.is_relative_to(R.parent):
  target=R/'runtime/lib'/p.name;shutil.copy2(p,target);libraries.append(dict(original=str(p),path=target.relative_to(R).as_posix(),sha256=sha(target)))
assert any(x['path'].endswith('libORB_SLAM3.so') for x in libraries)
inputs=[]
for row in original['rows']:
 i=row['frame_id'];assert row['source_frame_id']==3000+i
 for cam in ['image_2','image_3']:
  p=P/'input_view'/cam/f'{i:06}.png';assert sha(p)==row[cam]['sha256']
  with p.open('rb') as f:head=f.read(24)
  assert head[:8]==bytes([137,80,78,71,13,10,26,10]);w,h=struct.unpack('>II',head[16:24]);assert (h,w)==(370,1226)
  inputs.append(dict(path=str(p),source=row[cam]['source'],sha256=sha(p),bytes=p.stat().st_size))
cflags=shlex.split(subprocess.run(['pkg-config','--cflags','opencv4'],capture_output=True,text=True,check=True).stdout);libs=shlex.split(subprocess.run(['pkg-config','--libs','opencv4'],capture_output=True,text=True,check=True).stdout)
args=['g++','-std=c++17','-O2','-pthread','-MMD','-MF',str(R/'build/dependencies.d')]
for path in [root,root/'include',root/'include/CameraModels',root/'Thirdparty/Sophus',Path('/usr/include/eigen3'),root.parent/'pangolin_install/include']:args+=['-I'+str(path)]
args+=cflags+[str(R/'causal_stereo.cc'),'-L'+str(R/'runtime/lib'),'-lORB_SLAM3','-Wl,-rpath,$ORIGIN/../runtime/lib','-Wl,-rpath-link,'+str(R/'runtime/lib')]+libs+['-lboost_serialization','-lcrypto','-o',str(R/'bin/causal_stereo')]
dump(R/'build/command.json',dict(argv=args))
with (E/'compile.log').open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env(),timeout=600,check=True)
dep=(R/'build/dependencies.d').read_text().replace('\\\n',' ');headers=[]
for text in shlex.split(dep.split(':',1)[1]):
 p=Path(text)
 if p.is_file() and p.is_relative_to(root):
  target=R/'provenance/headers'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);headers.append(dict(path=str(p),sha256=sha(p)))
cppinfo=subprocess.run(['g++','--version'],capture_output=True,text=True,check=True).stdout
reg=dict(at=now(),status='REGISTERED_CAUSAL_INPUT_NOT_EXECUTED',window=original['window'],inputs=inputs,libraries=libraries,source_headers=headers,compiler=cppinfo,OpenCV_flags=cflags+libs,settings_sha256=sha(R/'inputs/settings.yaml'),vocabulary_sha256=sha(R/'runtime/ORBvoc.txt'),binary_sha256=sha(R/'bin/causal_stereo'),source_files={p.relative_to(R).as_posix():sha(p) for p in R.glob('*') if p.is_file()},GT_allowed=False,quality_adoption=False)
dump(R/'registration.json',reg);print('PASS_BUILD_AND_REGISTER2000_STEREO_INPUTS',flush=True)
