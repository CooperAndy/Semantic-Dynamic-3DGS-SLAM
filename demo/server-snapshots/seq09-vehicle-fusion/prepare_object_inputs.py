"""Export compact original instance masks and mask-covered RAFT depth for association."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); source,out=Path(args.source),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'instances').mkdir();(out/'masked_depth').mkdir()
    provenance=[]
    for local in range(200):
        n=local+300
        mask_path=source/f'annotations_20261107/instances/{local:06d}_masks.npz'
        instance_path=source/f'annotations_20261107/instances/{local:06d}.json'
        depth_path=source/f'raft_left/{local:06d}_depth.npy'
        motion_path=source/f'annotations_20261107/motion/{local:06d}.json'
        with np.load(mask_path,allow_pickle=False) as f:masks=f['masks']
        depth=np.load(depth_path,allow_pickle=False)
        metadata=json.loads(instance_path.read_text()); motion=json.loads(motion_path.read_text())
        assert masks.shape[1:]==depth.shape==(370,1226)
        by_index={o['mask_index']:o for o in metadata['instances']}
        for o in motion['objects']:
            instance=by_index[o['mask_index']]
            assert instance['track_id']==o['track_id'] and instance['mask_area']==o['mask_area']
        union=masks.any(axis=0) if len(masks) else np.zeros(depth.shape,dtype=bool)
        cropped=np.where(union&np.isfinite(depth)&(depth>=.5)&(depth<=40),depth,0).astype('float32')
        np.savez_compressed(out/f'masked_depth/{n:06d}.npz',depth=cropped)
        shutil.copy2(mask_path,out/f'instances/{n:06d}_masks.npz')
        shutil.copy2(instance_path,out/f'instances/{n:06d}.json')
        provenance.append(dict(source_frame=n,mask_sha256=sha(mask_path),instance_sha256=sha(instance_path),
                               source_depth_sha256=sha(depth_path),motion_sha256=sha(motion_path),
                               masked_depth_sha256=sha(out/f'masked_depth/{n:06d}.npz')))
    params=source/'runs/F1_appearance800/params.npz'
    with np.load(params,allow_pickle=False) as f:intrinsics=f['intrinsics'].tolist()
    manifest=dict(source=str(source),frames=provenance,intrinsics=intrinsics,source_map_sha256=sha(params),
                  depth_policy='Original RAFT-Stereo float32 values inside existing instance masks; 0.5-40 m; zero outside masks. No new depth or labels inferred.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=out.parent/'object_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='object_inputs')
    (out.parent/'object_inputs.sha256').write_text(sha(archive))
    print(json.dumps(dict(frames=200,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))))


if __name__=='__main__':main()
