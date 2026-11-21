"""Prepare metric instance/depth association inputs for any contiguous sequence window.

Run beside the original server inputs. Source frame numbers and local filenames
are explicit; no SSH password or server connection is stored here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['instances','depth','motion','params','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    p.add_argument('--local-start',type=int,default=0);p.add_argument('--depth-min',type=float,default=.5);p.add_argument('--depth-max',type=float,default=40.)
    p.add_argument('--depth-template',default='{local:06d}_depth.npy')
    a=p.parse_args()
    if a.start<0 or a.end<a.start or not .5<=a.depth_min<a.depth_max:raise ValueError('Invalid frame/depth range')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'instances').mkdir();(out/'masked_depth').mkdir()
    records=[];dimensions=set()
    for n in range(a.start,a.end+1):
        local=a.local_start+n-a.start
        mp=Path(a.instances)/f'{local:06d}_masks.npz';jp=Path(a.instances)/f'{local:06d}.json'
        dp=Path(a.depth)/a.depth_template.format(local=local,source=n);rp=Path(a.motion)/f'{local:06d}.json'
        with np.load(mp,allow_pickle=False) as z:masks=z['masks']
        depth=np.load(dp,allow_pickle=False);meta=json.loads(jp.read_text(encoding='utf-8-sig'));raw=json.loads(rp.read_text(encoding='utf-8-sig'))
        if masks.ndim!=3 or masks.shape[1:]!=depth.shape or depth.ndim!=2:raise ValueError(f'Mask/depth dimensions disagree at {n}')
        dimensions.add(tuple(depth.shape))
        by_index={o['mask_index']:o for o in meta['instances']}
        for o in raw['objects']:
            instance=by_index[o['mask_index']]
            if instance['track_id']!=o['track_id'] or instance['mask_area']!=o['mask_area']:raise ValueError(f'Instance/motion indexing mismatch at {n}')
            if 'confidence' not in instance:raise ValueError('Actual detection confidence missing; do not substitute motion probability')
        union=masks.any(axis=0) if len(masks) else np.zeros(depth.shape,dtype=bool)
        values=np.where(union&np.isfinite(depth)&(depth>=a.depth_min)&(depth<=a.depth_max),depth,0).astype('float32')
        output_depth=out/f'masked_depth/{n:06d}.npz';np.savez_compressed(output_depth,depth=values)
        shutil.copy2(mp,out/f'instances/{n:06d}_masks.npz');shutil.copy2(jp,out/f'instances/{n:06d}.json')
        records.append(dict(source_frame=n,local_frame=local,mask_sha256=sha(mp),instance_sha256=sha(jp),source_depth_sha256=sha(dp),motion_sha256=sha(rp),masked_depth_sha256=sha(output_depth)))
    with np.load(a.params,allow_pickle=False) as z:intrinsics=z['intrinsics']
    if intrinsics.shape!=(3,3) or not np.isfinite(intrinsics).all():raise ValueError('Expected finite 3x3 intrinsics from this exact map')
    manifest=dict(sequence=a.sequence,source_frame_range=[a.start,a.end],local_start=a.local_start,frames=records,intrinsics=intrinsics.tolist(),source_map_sha256=sha(a.params),depth_range_m=[a.depth_min,a.depth_max],image_shapes=[list(s) for s in sorted(dimensions)],depth_policy='Original float32 RAFT values inside original instance masks; zero outside. No new inference.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(frames=len(records),output=str(out),image_shapes=manifest['image_shapes'])))


if __name__=='__main__':main()
