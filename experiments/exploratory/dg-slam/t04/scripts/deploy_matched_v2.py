from pathlib import Path
import json,shutil,hashlib,py_compile
R=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
assert not (R/'source_v2').exists(), 'Preserve existing revision, no overwrite'
shutil.copytree(R/'source',R/'source_v2',symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
g=R/'source_v2/dg_slam/gs_tracking_mapping.py';s=g.read_text()
for old,new in [('2.0*self.cx/self.W - 1.0','(2.0*self.cx + 1.0)/self.W - 1.0'),('2.0*self.cy/self.H - 1.0','(2.0*self.cy + 1.0)/self.H - 1.0'),('if idx <= 1:','if idx <= 1 or not self.cfg.get("gs_pose_refine", True):')]:
    assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
g.write_text(s);py_compile.compile(str(g),doraise=True)
v=R/'runtime200_v2';v.mkdir()
for f in (R/'runtime200').iterdir():
    if f.name!='protocol.json':(v/f.name).symlink_to(f)
shutil.copy2(R/'matched_v2/protocol.json',v/'protocol.json')
shutil.copy2(R/'matched_v2/frozen_config_v2.json',R/'frozen_config_v2.json')
p=json.loads((v/'protocol.json').read_text())
for name,digest in p['input_hashes'].items():assert hashlib.sha256((v/name).read_bytes()).hexdigest()==digest
baseline=R/'matched_stereo';baseline.mkdir();(baseline/'image_2').symlink_to(v,target_is_directory=True);(baseline/'image_3').symlink_to(R/'matched_v2/image_3',target_is_directory=True)
shutil.copy2(R/'matched_v2/times.txt',baseline/'times.txt')
(R/'v2_preparation.json').write_text(json.dumps({'status':'PREPARED','source':str(g),'source_sha256':hashlib.sha256(g.read_bytes()).hexdigest(),'K':p['K'],'raw_rgb_depth_unchanged':True,'gt_or_orb_pose_input':False,'changes':['OpenCV resize pixel-center K','CUDA raster ndc2Pix projection +1','optional GS pose refinement bypass; same mapping budget'],'comparison_limits':['RGBD pseudo-depth vs ORB stereo input remains a method difference','OFF/ON independent single runs; inspect coarse-pose correspondence before causal claims']},indent=2))
print('V2 DEPLOYED')
