"""Freeze corrected geometry and instrument independent KITTI controls; preserve V1."""
from pathlib import Path
import json, hashlib, tarfile, py_compile
import cv2, numpy as np
R=Path(__file__).resolve().parents[1]
out=R/'03_数据索引/matched_v2';out.mkdir(exist_ok=True)
data=R.parents[1]/'slam_project/data/kitti/sequences/00'
p=json.loads((R/'03_数据索引/runtime200/protocol.json').read_text())
sx,sy=640/1241,192/376
p['K']=[718.856*sx,718.856*sy,(607.1928+.5)*sx-.5,(185.2157+.5)*sy-.5]
p['geometry_revision']='V2 OpenCV half-pixel resize and CUDA ndc2Pix center correction'
right=out/'image_3';right.mkdir(exist_ok=True)
for i in range(200):
    a=cv2.imdecode(np.frombuffer((data/'image_3'/f'{i:06d}.png').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    assert a.shape==(376,1241,3)
    ok,b=cv2.imencode('.png',cv2.resize(a,(640,192),interpolation=cv2.INTER_AREA));assert ok
    (right/f'{i:06d}.png').write_bytes(b.tobytes())
np.savetxt(out/'times.txt',np.loadtxt(data/'times.txt')[:200],fmt='%.12e')
(out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
f=json.loads((R/'04_后续运行/最终运行产物/frozen_config.json').read_text())
f['cfg']['cam'].update(dict(zip(['fx','fy','cx','cy'],p['K'])))
(out/'frozen_config_v2.json').write_text(json.dumps(f,indent=2),encoding='utf-8')
inputs=(R/'scripts/kitti_inputs.py').read_text().replace("ROOT/'runtime200'","ROOT/'runtime200_v2'")
(R/'scripts/kitti_inputs_v2.py').write_text(inputs,encoding='utf-8')
w=(R/'04_后续运行/首次GPU回执/scripts__run_kitti.py').read_text()
w=w.replace('from kitti_inputs import','from kitti_inputs_v2 import').replace("source=ROOT/'source'","source=ROOT/'source_v2'")
w=w.replace("a=p.parse_args()","p.add_argument('--gs-refine',choices=['on','off'],required=True);a=p.parse_args()")
w=w.replace("ROOT/'frozen_config.json'","ROOT/'frozen_config_v2.json'").replace("with_name('kitti_inputs.py')","with_name('kitti_inputs_v2.py')")
w=w.replace("setup_seed(cfg['setup_seed'])","cfg['gs_pose_refine']=a.gs_refine=='on'\n    setup_seed(cfg['setup_seed'])")
w=w.replace("frame_audit=[]","frame_audit=[];coarse=[];refined=[]")
w=w.replace('return original_dispatch(idx,*args,**kwargs)',"coarse.append(args[2].detach().cpu().numpy().copy())\n            result=original_dispatch(idx,*args,**kwargs)\n            refined.append(mapper.estimate_c2w_list[key].cpu().numpy().copy())\n            np.savez(output/'pose_stages.npz',coarse=np.array(coarse),refined=np.array(refined),frame_ids=[x['actual_rgb_input_frame'] for x in frame_audit])\n            return result")
w=w.replace("assert all(counts[k]>0 for k in ['gs_pose_optimization_calls','gs_mapping_calls','gs_render_calls'])","assert all(counts[k]>0 for k in ['gs_mapping_calls','gs_render_calls'])\n        assert (counts['gs_pose_optimization_calls']>0) == cfg['gs_pose_refine']")
(R/'scripts/run_kitti_v2.py').write_text(w,encoding='utf-8')
for n in ['kitti_inputs_v2.py','run_kitti_v2.py']:py_compile.compile(str(R/'scripts'/n),doraise=True)
manifest={str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest() for x in out.rglob('*') if x.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(out.parent/'matched_v2.tar.gz','w:gz') as t:t.add(out,arcname='matched_v2')
print('V2 prepared:',p['K'],len(manifest),'files; no GT or ORB poses')
