"""Install the independently computed O3F semantic object states."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import json, shutil
from PIL import Image, ImageDraw, ImageFont
from aggregate_vehicle_states import sha, save_json

ROOT=Path(__file__).resolve().parent.parent.parent
DEMO=ROOT/'阶段2/动态语义Demo_设计/T07_动态语义独立副本'
RUN=Path(__file__).resolve().parent/'seq13_o3f_20261116'
BASE=DEMO/'scene-data/KITTI13_O3F_260'
ASSET=BASE/'ObjectStates'
BACK=DEMO/'修改记录/20261116_seq13_o3f_object_states'
CFG=DEMO/'源码/scene-config.kitti13-o3f-260.json'
CAT=DEMO/'源码/scene-catalog.json'
RGB=ROOT/'slam_project/data/kitti/sequences/13/image_2'
KEY='6cebfe6efc70888706c17c443e5774e3894bd74e0d0baa6f495b51a33bb97c20'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    assert not ASSET.exists() and not BACK.exists()
    cfg=read(CFG); summary=read(RUN/'object_trial/summary.json'); objs=read(RUN/'object_trial/objects.json')
    assert cfg['sceneId']=='kitti13_o3f_780_1039' and summary['source_map_sha256']==KEY==read(BASE/'manifest.json')['params_sha256']
    for p,h in summary['source_hashes'].items(): assert sha(p)==h, p
    assert summary['original_anchor_reprojection_all_pass'] and any(o['final_state']!='uncertain' for o in objs)
    states={o['object_id']:o['final_state'] for o in objs}; by_frame=defaultdict(list)
    for row in read(RUN/'object_trial/deduplicated_observations.json'):
        if states[row['object_id']]!='uncertain': by_frame[row['source_frame']].append(row)
    frame=max(by_frame,key=lambda n:(len(by_frame[n]),sum(r['mask_area'] for r in by_frame[n]),-abs(n-909)))
    inputs=read(RUN/'prepared/object_inputs/manifest.json');assert sha(RGB/f'{frame:06d}.png')==inputs['frames'][frame-780]['rgb_sha256']
    protected={p:sha(p) for p in (BASE/'RGB.ply',BASE/'Semantic.ply',BASE/'camera_poses.json')}
    BACK.mkdir(parents=True); shutil.copy2(CFG,BACK/CFG.name);shutil.copy2(CAT,BACK/CAT.name)
    ASSET.mkdir();(ASSET/'reference').mkdir();(ASSET/'evidence').mkdir()
    snapshot=deepcopy(read(RUN/f'prepared/frames/{frame:06d}.json')); display=[]
    for o in objs:
        a=o['world_anchor'];display.append(dict(track_id=o['region_index']+1,object_id=o['object_id'],category=o['category'],score=o['score'],state=o['state'],final_state=o['final_state'],final_reason=o['final_reason'],source_track_ids=o['source_track_ids'],display_estimate=True,world_xyz=a,viewer_xyz=[-a[0],-a[1],a[2]],fixed_state=True,anchor_verified=False,semantic_anchor_bound=True,anchor_source_gaussian_index=o['anchor_source_gaussian_index'],anchor_status='actual_gaussian_in_O3F_semantic_vehicle_region',gaussian_region_index=o['region_index'],region_gaussian_count=o['gaussian_count'],identity_status=o['identity_status'],aggregate_evidence=dict(first_frame=o['first_frame'],last_frame=o['last_frame'],valid_count=o['valid_observation_count'],unique_observation_frames=o['unique_observation_frames'],sampled_frames=o['sampled_frames'],final_reason=o['final_reason'],static_hold=o['static_hold'])))
    counts=dict(Counter(o['final_state'] for o in objs));prefix='scene-data/KITTI13_O3F_260/ObjectStates'
    snapshot.update(objects=display,semantic_object_labels=True,fixed_track_labels=True,label_mode='offline_semantic_vehicle_objects',motion_model='Pi3MOS + semantic vehicle objects',source_frames=[780,1039],window=[],aggregation_config='semantic-region-v1 + short4_z0.5-40_far20',reference_image=f'{prefix}/reference/{frame:06d}_objects.jpg',history_note='Fixed state per predicted vehicle region, aggregated across frames 780-1039; causal 8-frame Pi3MOS scores.',position_note='Anchor is a Gaussian belonging to its O3F semantic vehicle region.',object_counts=counts,positioned_object_count=len(display),track_counts=counts,positioned_track_counts=counts,positioned_track_count=len(display),unpositioned_track_count=0,association_manifest=f'{prefix}/manifest.json',reference_image_note='Illustrative frame selected after predictions, not ground truth.',accuracy_status='NOT_EVALUATED_NO_OBJECT_GROUND_TRUTH')
    snapshot.pop('display_2d_only',None);save_json(ASSET/'snapshot.json',snapshot)
    image=Image.open(RGB/f'{frame:06d}.png').convert('RGB');draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for r in by_frame[frame]:
        state=states[r['object_id']];color={'static':'#368cf5','dynamic':'#f28c28'}[state];box=r['bbox'];draw.rectangle(box,outline=color,width=2); label=f"{r['object_id']} | {state.upper()}";x,y=box[0],max(0,box[1]-18);w=draw.textbbox((0,0),label,font=font)[2]+6;draw.rectangle((x,y,min(image.width,x+w),y+18),fill='#132035');draw.text((x+2,y),label,font=font,fill=color)
    image.save(ASSET/f'reference/{frame:06d}_objects.jpg',quality=93)
    for p in (RUN/'object_trial').iterdir():
        if p.is_file(): shutil.copy2(p,ASSET/'evidence'/p.name)
    save_json(ASSET/'manifest.json',dict(method='predicted_semantic_3d_regions_with_mask_depth_association',source_map_sha256=KEY,source_frames=[780,1039],summary=summary,gaussian_count=cfg['gaussianCount'],gaussian_region_ids='evidence/gaussian_region_ids.npz',semantic_source='O3F fixed-map offline fusion of 260 Mask2Former frames',motion_source='Existing Pi3MOS causal8 float32 instance medians from identical RGB images',depth_source='Existing metric RAFT stereo depth on identical RGB images',motion_accuracy_verified=False,backup=str(BACK)))
    cfg.update(motionIndex=None,motionSnapshot=f'{prefix}/snapshot.json',motionReferenceImageTemplate=None,motionLabelMode='offline_semantic_vehicle_objects',motionStateManifest=f'{prefix}/manifest.json',motionStateConfig='semantic-region-v1 + short4_z0.5-40_far20',hideUncertainMotion=True,dynamicVehicleRemoval=None)
    save_json(CFG,cfg)
    catalog=read(CAT)
    for scene in catalog['scenes']:
        if scene['id']==cfg['sceneId']: scene['label']='Seq 13 · 000780–001039 · O3F · Object States'
    save_json(CAT,catalog)
    assert all(sha(p)==h for p,h in protected.items())
    receipt=dict(scene=cfg['sceneId'],state_counts=counts,shown=sum(o['final_state']!='uncertain' for o in objs),reference_frame=frame,map_sha256=KEY,protected_files_unchanged=True)
    save_json(ASSET/'installation_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
