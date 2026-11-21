"""Rebind existing image-level seq10 observations to the O3F camera poses."""
from pathlib import Path
import json, shutil, sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEMO = ROOT / '阶段2/动态语义Demo_设计/T07_动态语义独立副本'
SRC = HERE / 'seq10_tmd_20261115/fullraw_prepared/seq10_prepared'
D = ROOT / '特调地图/seq10_o3f_1000_delivery'
OUT = HERE / 'seq10_o3f_20261116'
MAP = DEMO / 'scene-data/KITTI10_O3F_1000'
sys.path.insert(0, str(HERE))
from prepare_seq10_tmd_motion import anchor, sha, save

def main():
    manifest = json.loads((SRC/'object_inputs/manifest.json').read_text(encoding='utf8'))
    data = json.loads((D/'input_manifest.json').read_text(encoding='utf8'))
    poses = json.loads((MAP/'camera_poses.json').read_text(encoding='utf8'))
    key = '09f8afdbe059d61d50feb92c9529276dd3947f3578de1e0c1d047009f92e2a7f'
    assert sha(D/'params.npz') == key and len(poses)==len(data['rows'])==len(manifest['frames'])==1000
    assert np.allclose(data['K'], manifest['intrinsics'], atol=0.0001)
    assert all(a['frame_id']==b['source_frame']==i and a['hashes']['rgb']==b['rgb_sha256'] for i,(a,b) in enumerate(zip(data['rows'],manifest['frames'])))
    OUT.mkdir(exist_ok=False); (OUT/'frames').mkdir(); (OUT/'semantic').mkdir()
    shutil.copytree(SRC/'object_inputs', OUT/'object_inputs')
    manifest['source_map_sha256']=key
    manifest['rebind_note']='Same 1000 RGB hashes and intrinsics; masks and metric depth reused, O3F camera poses used for anchors.'
    save(OUT/'object_inputs/manifest.json',manifest)
    for n, pose in enumerate(poses):
        assert pose['source_frame']==n
        snap=json.loads((SRC/f'frames/{n:06d}.json').read_text(encoding='utf8'))
        with np.load(OUT/f'object_inputs/instances/{n:06d}_masks.npz') as z: masks=z['masks']
        with np.load(OUT/f'object_inputs/masked_depth/{n:06d}.npz') as z: depth=z['depth']
        for o in snap['objects']:
            o.update(anchor(masks[o['mask_index']].astype(bool),depth,pose,np.asarray(data['K'])))
        snap.update(scene_id='kitti10_o3f_000000_000999',source_map_sha256=key,depth='Existing metric RAFT depth, O3F poses; 0.5-40m')
        save(OUT/f'frames/{n:06d}.json',snap)
    shutil.copy2(D/'semantic_labels.npz', OUT/'semantic/semantic_gaussian_labels.npz')
    shutil.copy2(MAP/'Semantic.ply', OUT/'semantic/Semantic.ply')
    save(OUT/'semantic/semantic_manifest.json',dict(source_params_sha256=key,semantic_ply_sha256=sha(OUT/'semantic/Semantic.ply')))
    profile=json.loads((HERE/'reuse/seq10_tmd_profile.json').read_text(encoding='utf8'))
    profile.update(name='seq10_o3f_object_trial',scene_id='kitti10_o3f_000000_000999',scope='O3F semantic regions; same-image original Pi3MOS scores and RAFT depth; recomputed O3F pose anchors.')
    profile['paths'].update(semantic_dir=str(OUT/'semantic'),raw_frames=str(OUT/'frames'),camera_poses=str(MAP/'camera_poses.json'),instance_inputs=str(OUT/'object_inputs'),observation_run=str(OUT/'observations'))
    save(OUT/'profile.json',profile)
    print('Prepared 1000 matched frames and O3F anchors',flush=True)

if __name__=='__main__':main()
