"""Build only an isolated headless entry point against existing ORB library."""
from pathlib import Path
import json,subprocess,hashlib,os
R=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
B=Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2')
S=B/'ORB_SLAM3_4452a3c4';O=R/'orb_matched_v2';O.mkdir(exist_ok=True);assert not (O/"stereo_kitti_headless").exists() and not (O/"tracking.log").exists()
p=json.loads((R/'runtime200_v2/protocol.json').read_text());fx,fy,cx,cy=p['K']
source=(S/'Examples/Stereo/stereo_kitti.cc').read_text();old='ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::STEREO,true);';assert source.count(old)==1
source=source.replace(old,old.replace(',true)',',false)'))
(O/'stereo_kitti_headless.cc').write_text(source)
for alias,target in [('image_0','image_2'),('image_1','image_3')]:
    link=R/'matched_stereo'/alias
    if not link.exists():link.symlink_to(R/'matched_stereo'/target,target_is_directory=True)
cfg=f'''%YAML:1.0
File.version: "1.0"
Camera.type: "Rectified"
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}
Camera.width: 640
Camera.height: 192
Camera.fps: 10
Camera.RGB: 0
Stereo.b: 0.532331857841
Stereo.ThDepth: 35.0
ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.6
Viewer.KeyFrameLineWidth: 2.0
Viewer.GraphLineWidth: 1.0
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.7
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -100.0
Viewer.ViewpointZ: -0.1
Viewer.ViewpointF: 2000.0
'''
(O/'settings.yaml').write_text(cfg)
libs=[B/'p4_prebuilt',B/'pangolin_install/lib',S/'Thirdparty/DBoW2/lib',S/'Thirdparty/g2o/lib']
env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(map(str,libs))+':'+env.get('LD_LIBRARY_PATH','')
flags=subprocess.check_output(['pkg-config','--cflags','--libs','opencv4'],text=True).split()
cmd=['g++','-std=c++14','-DCOMPILEDWITHC11','-DHAVE_EIGEN','-DHAVE_EPOXY','-D_LINUX_','-O2',str(O/'stereo_kitti_headless.cc'),'-o',str(O/'stereo_kitti_headless'),'-I'+str(S),'-I'+str(B/'pangolin_install/include'),'-I'+str(S/'include'),'-I'+str(S/'include/CameraModels'),'-I'+str(S/'Thirdparty/Sophus'),'-I/usr/include/eigen3']+['-L'+str(x) for x in libs]+['-lORB_SLAM3','-lpango_display','-lpango_vars','-lpango_opengl','-lpango_core','-lDBoW2','-lg2o','-lboost_serialization','-lcrypto','-lpthread','-lepoxy']+flags
build=subprocess.run(cmd,env=env,capture_output=True,text=True)
(O/'build.log').write_text(build.stdout+build.stderr);assert build.returncode==0,build.stderr[-5000:]
files=[O/'stereo_kitti_headless.cc',O/'stereo_kitti_headless',O/'settings.yaml',B/'p4_prebuilt/libORB_SLAM3.so',S/'Vocabulary/ORBvoc.txt']
receipt={'status':'BUILT','source_library_commit_label':'4452a3c4 recovered, differs from historical baseline provenance','input':'same 200 P2/P3 RGB pixel bytes at 640x192; BGR imread and Camera.RGB=0','no_masks':True,'entry_change':'viewer true -> false only','build_command':cmd,'hashes':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}}
(O/'preparation.json').write_text(json.dumps(receipt,indent=2))
run='#!/usr/bin/env bash\nset -euo pipefail\ncd "'+str(O)+'"\nexport LD_LIBRARY_PATH="'+env['LD_LIBRARY_PATH']+'"\nexport OMP_NUM_THREADS=4\nset +e\ntimeout 10m ./stereo_kitti_headless "'+str(S/'Vocabulary/ORBvoc.txt')+'" settings.yaml "'+str(R/'matched_stereo')+'" > tracking.log 2>&1\nstatus=$?\nprintf "%s\\n" "$status" > tracking.exit\nexit "$status"\n'
(O/'run.sh').write_text(run)
print('ORB MATCHED BUILT')
