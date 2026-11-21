"""Validate the transferred object-state scene without browser automation."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from aggregate_vehicle_states import sha,save_json
sys.path.insert(0,str(Path(__file__).parent/'reuse/core'))
from fuse_reference_seq09 import load_ply


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--demo-root',required=True);a=p.parse_args()
    root=Path(a.demo_root);code=root/'源码';base=root/'scene-data/KITTI09_C50S42';target=base/'KITTI09_ObjectStates'
    cfg=read(code/'scene-config.kitti09-c50-s42.json');s=read(target/'snapshot.json');manifest=read(target/'manifest.json')
    assert cfg['motionIndex'] is None and cfg['motionReferenceImageTemplate'] is None
    assert cfg['dynamicVehicleRemoval'] is None and cfg['hideUncertainMotion'] is True
    assert cfg['sceneId']==s['scene_id']=='kitti09_conf050_seed42'
    assert cfg['gaussianCount']==3166109 and s['source_map_sha256']==manifest['source_map_sha256']
    assert s['semantic_object_labels'] and s['fixed_track_labels'] and len(s['objects'])==13
    assert len({o['object_id'] for o in s['objects']})==13
    assert all(o['final_state']=='static' for o in s['objects'])
    assert (root/s['reference_image']).is_file()
    ply=load_ply(base/'KITTI09_F1_Semantic/KITTI09_F1_Semantic.ply')
    with np.load(target/'evidence/gaussian_region_ids.npz') as z:region=z['region_id']
    assert len(region)==len(ply)==cfg['gaussianCount']
    for o in s['objects']:
        ix=o['anchor_source_gaussian_index']
        assert region[ix]==o['gaussian_region_index']
        assert o['world_xyz']==[float(ply[k][ix]) for k in ('x','y','z')]
        assert o['viewer_xyz']==[-o['world_xyz'][0],-o['world_xyz'][1],o['world_xyz'][2]]
    assert len([o for o in s['objects'] if 77 in o['source_track_ids'] and 103 in o['source_track_ids']])==1
    backup=Path(manifest['backup']);protected=read(backup/'backup_manifest.json')['protected']
    assert all(sha(root/name)==h for name,h in protected.items())
    html=(code/'interactive_viewer.html').read_text(encoding='utf-8')
    body=next(b for b in re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>',html,re.S) if 'async function readMotionSnapshot' in b)
    reader=body[body.index('  async function readMotionSnapshot'):body.index('  async function initializeMotion')]
    loader=body[body.index('  async function initializeMotion'):body.index('  const requestedMode')]
    visible=re.search(r'  function isMotionLabelVisible\(object\) \{.*?\n  \}',body,re.S).group(0)
    js='''const vm=require('node:vm'),assert=require('node:assert/strict'),a=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const control={hidden:false},requests=[];
const c={config:a.cfg,poses:new Map([[399,{}]]),motionSnapshot:null,motionIndex:null,console,resolveAsset:p=>p,document:{getElementById:()=>control},fetch:async p=>{requests.push(p);return {ok:true,json:async()=>a.snapshot}}};
vm.createContext(c);vm.runInContext(a.functions,c);
(async()=>{await c.initializeMotion();assert(c.motionSnapshot.semantic_object_labels);assert.equal(c.motionSnapshot.scene_id,'kitti09_conf050_seed42');assert.equal(control.hidden,true);assert.deepEqual(requests,[a.cfg.motionSnapshot]);assert.equal(c.motionSnapshot.objects.filter(c.isMotionLabelVisible).length,13);console.log('C50S42 production loader PASS: one object snapshot, 13 labels, no frame selector.');})().catch(e=>{console.error(e);process.exitCode=1});'''
    r=subprocess.run(['node','-e',js],input=json.dumps(dict(cfg=cfg,snapshot=s,functions=reader+loader+visible)),capture_output=True,text=True,encoding='utf-8')
    assert r.returncode==0,r.stderr
    report=dict(status='PASS',scene=s['scene_id'],target_map_gaussian_count=3166109,
                target_map_hash=s['source_map_sha256'],source_anchor_membership='13/13 in target C50S42 regions',
                production_loader='PASS',shown_static_labels=13,state_frame_selector_hidden=True,
                legacy_cleanup_disabled=True,original_raw_predictions_unchanged=True,
                old_successful_scene_and_original_map_files_unchanged=True,
                visual_review='Not performed: user will inspect')
    save_json(target/'integration_check.json',report);print(r.stdout.strip());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
