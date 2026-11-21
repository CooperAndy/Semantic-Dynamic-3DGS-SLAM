"""Run only on the authorized server: isolate and patch R1.1, freeze worker."""
from pathlib import Path
import hashlib,json,shutil,ast
ROOT=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
BASE=Path('/root/autodl-tmp/dg_slam_chy_20261011')
source=ROOT/'source';assert not source.exists()
parent=BASE/'01_source_env/DG-SLAM_R11_no_gt_timestamps'
shutil.copytree(parent,source,ignore=shutil.ignore_patterns('__pycache__','.git','assets','checkpoints','submodules'))
path=source/'dg_slam/gs_tracking_mapping.py';old=path.read_text();new=old
assert new.count('self.scale_factor = 0.8')==1
new=new.replace('self.scale_factor = 0.8','self.scale_factor = 1.0')
for token in ['gt_depth < 8.', 'gt_depths < 8.']:
    new=new.replace(token,token.replace('8.','40.'))
needle='self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy).transpose(0,1).cuda()'
assert needle in new
new=new.replace(needle,needle+'\n        self.projection_matrix[2, 0] = 2.0*self.cx/self.W - 1.0\n        self.projection_matrix[2, 1] = 2.0*self.cy/self.H - 1.0')
ast.parse(new);path.write_text(new)
manifest=json.loads((BASE/'04_reproduction/R11_200_20261011_175732/run_manifest.json').read_text())
p=json.loads((ROOT/'runtime200/protocol.json').read_text());cfg=manifest['cfg'];args=manifest['args']
cfg['cam'].update(H=192,W=640,fx=p['K'][0],fy=p['K'][1],cx=p['K'][2],cy=p['K'][3],png_depth_scale=1.)
cfg['scene']='KITTI00_000000_000199';cfg['project_name']='T04_KITTI_ENGINEERING';cfg['tracking']['handle_dynamic']=False
cfg['mapping']['color_refine']=False
args.update(image_size=[192,640],config='frozen_config.json')
(ROOT/'frozen_config.json').write_text(json.dumps({'cfg':cfg,'args':args},indent=2))
worker=(BASE/'scripts/r1/run_r11.py').read_text()
worker=worker.replace('from r1_inputs import','from kitti_inputs import')
worker=worker.replace("a.run_name.startswith('R11_200_')","a.run_name.startswith('KITTI00_200_')")
start=worker.index("    source=ROOT/'01_source_env/DG-SLAM_R11_no_gt_timestamps'")
end=worker.index('    os.chdir(source)',start)
worker=worker[:start]+"    source=ROOT/'source'\n"+worker[end:]
start=worker.index("    cfg_path='configs/TUM_RGBD/")
end=worker.index("    setup_seed",start)
worker=worker[:start]+"    frozen=json.loads((ROOT/'frozen_config.json').read_text());cfg=frozen['cfg'];args=argparse.Namespace(**frozen['args'])\n    cfg['data'].update(n_img=200,output=str(output),exp_name='KITTI00_NO_GT_NO_SEMANTIC');cfg['wandb']=False\n"+worker[end:]
worker=worker.replace("'inputs_code_sha256':sha256(Path(__file__).with_name('r1_inputs.py')),'source_preparation_sha256':sha256(ROOT/'05_reports/r11_200_20261011/source_preparation.json')","'inputs_code_sha256':sha256(Path(__file__).with_name('kitti_inputs.py')),'frozen_config_sha256':sha256(ROOT/'frozen_config.json')")
start=worker.index("    pose_check=ROOT/");end=worker.index('    started=time.time()',start)
worker=worker[:start]+"    meta['variant']='KITTI_STEREO_DERIVED_RGBD_NO_SEMANTIC_ENGINEERING'\n    meta['notes']=['SGBM pseudo-depth, no GT or ORB pose input','No semantic exclusion; not original OneFormer method','640x192 engineering adaptation; not formal quality acceptance']\n"+worker[end:]
worker=worker.replace("        assert meta['runtime_index_sha256']==prep['runtime_index_sha256']",'')
ast.parse(worker)
(ROOT/'scripts/run_kitti.py').write_text(worker)
shutil.copy2(BASE/'scripts/r1/r1_pose_io.py',ROOT/'scripts/r1_pose_io.py')
assert 'cfg_path' not in worker and "prep['" not in worker
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'parent_source':str(parent),'parent_mapper_sha256':hashlib.sha256(old.encode()).hexdigest(),'patched_mapper_sha256':sha(path),'worker_sha256':sha(ROOT/'scripts/run_kitti.py'),'config_sha256':sha(ROOT/'frozen_config.json'),'scope':'independent KITTI engineering adaptation; no teammate experiment rerun'}
(ROOT/'source_preparation.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
