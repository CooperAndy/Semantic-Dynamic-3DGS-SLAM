"""Stage2 dense classes plus delayed per-track motion; no categorical motion guess."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def export(R,sequence='08'):
    R=Path(R);runtime=json.loads((R/'runtime.json').read_text());out=R/'observations';out.mkdir(exist_ok=False)
    names=json.loads((R/'semantic/id2label.json').read_text());names=[names[str(i)] for i in range(len(names))]
    rows=[];counts={str(k):0 for k in [0,1,2,255]};locks=[]
    for row in runtime['rows']:
        i=row['frame_id'];frame=f'{i:06}';rgb=Path(row['rgb'])
        if sha(rgb)!=row['hashes']['rgb']:raise ValueError('G30 image identity mismatch')
        category=np.load(R/'semantic/labels'/f'{frame}_semantic.npy',allow_pickle=False).astype(np.int16)
        confidence=np.load(R/'semantic/confidence'/f'{frame}_confidence.npy',allow_pickle=False).astype(np.float32)
        shape=(runtime['H'],runtime['W']);ids=np.full(shape,-1,np.int32);state=np.full(shape,255,np.uint8)
        detection_file=R/'instances'/f'{frame}.json';mask_file=R/'instances'/f'{frame}_masks.npz'
        detection=json.loads(detection_file.read_text());assert detection['frame_id']==frame
        previous={}
        paths=[detection_file,mask_file,R/'semantic/labels'/f'{frame}_semantic.npy',R/'semantic/confidence'/f'{frame}_confidence.npy']
        if i:
            previous_file=R/'states'/f'{i-1:06}.json';pair=R/'motion'/f'{i-1:06}_{i:06}.json'
            source=json.loads(previous_file.read_text());evidence=json.loads(pair.read_text())
            assert evidence['frame_t']==f'{i-1:06}' and evidence['frame_next']==frame
            previous={int(x['track_id']):int(x['state']) for x in source['instances']}
            paths += [previous_file,pair]
        with np.load(mask_file,allow_pickle=False) as z:masks=z['masks']
        assert masks.shape[1:]==shape and category.shape==confidence.shape==shape
        for item in sorted(detection['instances'],key=lambda x:(x['confidence'],x['track_id'])):
            occupied=masks[item['mask_index']].astype(bool);tid=int(item['track_id'])
            ids[occupied]=tid;state[occupied]=previous.get(tid,1)
        file=out/f'{frame}.npz';np.savez_compressed(file,category=category,confidence=confidence,instance_id=ids,state=state)
        for k in counts:counts[k]+=int((state==int(k)).sum())
        locks.extend(dict(path=str(p),sha256=sha(p)) for p in paths)
        rows.append(dict(frame_id=i,source_frame=row['source_frame_id'],sequence=sequence,camera='P2',rgb_sha256=sha(rgb),
                         shape=list(shape),evidence_last_frame=row['source_frame_id'],path=file.name,sha256=sha(file)))
    summary=json.loads((R/'semantic/generation_summary.json').read_text())
    m=dict(schema='s23_observations_v1',dataset='KITTI_Odometry',sequence=sequence,camera='P2',GT_runtime=False,
           motion_states_available=True,instance_tracks_available=True,class_names=names,
           encoding=dict(Static=0,Uncertain=1,Dynamic=2,Ignore=255),
           producer=dict(semantic_checkpoint=summary['checkpoint'],source_locks=locks,pose_source='G19 causal ORB P2'),rows=rows)
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
    (out/'receipt.json').write_text(json.dumps(dict(status='COMPLETE_TRUE_STAGE2_DELAYED_OBSERVATIONS',frames=len(rows),counts=counts,GT_runtime=False),indent=2))
    return m

if __name__=='__main__':export(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '08')
