from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parent.parent.parent;HERE=Path(__file__).resolve().parent;OUT=HERE/'seq13_o3f_20261116';BASE=ROOT/'阶段2/动态语义Demo_设计/T07_动态语义独立副本/scene-data/KITTI13_O3F_260';SRC=ROOT/'特调地图/seq13_o3f_260_delivery'
assert (OUT/'prepared/object_inputs/manifest.json').is_file() and not (OUT/'semantic').exists()
(OUT/'semantic').mkdir();shutil.copy2(BASE/'Semantic.ply',OUT/'semantic/Semantic.ply');shutil.copy2(BASE/'semantic_manifest.json',OUT/'semantic/semantic_manifest.json');shutil.copy2(SRC/'semantic_gaussian_labels.npz',OUT/'semantic/semantic_gaussian_labels.npz')
p=json.loads((HERE/'reuse/seq10_tmd_profile.json').read_text(encoding='utf8'));p.update(name='seq13_o3f_object_trial',sequence='13',scene_id='kitti13_o3f_780_1039',frame_range=[780,1039],scope='Seq13 O3F fixed-map semantic regions, same-frame Mask2Former masks, RAFT metric depth and causal Pi3MOS predictions.')
p['paths'].update(semantic_dir=str(OUT/'semantic'),raw_frames=str(OUT/'prepared/frames'),camera_poses=str(BASE/'camera_poses.json'),instance_inputs=str(OUT/'prepared/object_inputs'),observation_run=str(OUT/'observations'))
(OUT/'profile.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
