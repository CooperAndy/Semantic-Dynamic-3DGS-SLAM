from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parent.parent.parent;HERE=Path(__file__).resolve().parent;OUT=HERE/'seq03_m3f_20261116';BASE=ROOT/'阶段2/动态语义Demo_设计/T07_动态语义独立副本/scene-data/KITTI03_M3F_801';SRC=ROOT/'特调地图/seq03_m3f_801_delivery'
assert (OUT/'prepared/object_inputs/manifest.json').is_file() and not (OUT/'semantic').exists()
(OUT/'semantic').mkdir();shutil.copy2(BASE/'Semantic.ply',OUT/'semantic/Semantic.ply');shutil.copy2(BASE/'semantic_manifest.json',OUT/'semantic/semantic_manifest.json');shutil.copy2(SRC/'semantic_gaussian_labels.npz',OUT/'semantic/semantic_gaussian_labels.npz')
p=json.loads((HERE/'reuse/seq10_tmd_profile.json').read_text(encoding='utf8'));p.update(name='seq03_m3f_object_trial',sequence='03',scene_id='kitti03_m3f_000000_000800',frame_range=[0,800],scope='Seq03 M3F fixed-map semantic regions, same-frame Mask2Former masks, RAFT metric depth and causal Pi3MOS predictions.')
p['paths'].update(semantic_dir=str(OUT/'semantic'),raw_frames=str(OUT/'prepared/frames'),camera_poses=str(BASE/'camera_poses.json'),instance_inputs=str(OUT/'prepared/object_inputs'),observation_run=str(OUT/'observations'))
(OUT/'profile.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
