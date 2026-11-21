"""Validate the original seq10 object-state data/production-reader contract."""
import argparse
from collections import Counter
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
    root=Path(a.demo_root);code=root/'源码';base=root/'scene-data/KITTI10_TMD_TSDD_seed2'
    cfg=read(code/'scene-config.kitti10-tmd-tsdd-s2.json');asset=(root/cfg['motionSnapshot']).parent
    assert asset.resolve().is_relative_to(base.resolve())
    s=read(asset/'snapshot.json');manifest=read(asset/'manifest.json')
    calculated=read(asset/'evidence/objects.json');expected=dict(Counter(o['final_state'] for o in calculated))
    assert cfg['motionIndex'] is None and cfg['hideUncertainMotion'] is True
    assert cfg['sceneId']==s['scene_id']=='kitti10_tmd_tsdd_s2' and s['source_frames']==[0,999]
    assert cfg['gaussianCount']==2484177 and len(s['objects'])==16
    assert s['object_counts']==expected==manifest['summary']['object_state_counts']
    assert {o['object_id']:o['final_state'] for o in s['objects']}=={o['object_id']:o['final_state'] for o in calculated}
    if manifest.get('compound_review'):
        review=read(asset/manifest['compound_review'])
        parent=Path(manifest['summary']['parent_calculation'])
        prior={o['object_id']:o for o in read(parent/'objects.json')}
        assert sha(asset/'evidence/gaussian_region_ids.npz')==sha(parent/'gaussian_region_ids.npz')
        for o in calculated:
            if o['object_id'] not in review['changed_objects']:assert o==prior[o['object_id']]
            else:
                assert o['compound_review']['resolution']=='isolated_alarm_explained_by_later_whole_instance'
                assert all(e['explained'] for e in o['compound_review']['explanations'])
    assert len({o['object_id'] for o in s['objects']})==16
    ply=load_ply(base/'Semantic.ply')
    with np.load(asset/'evidence/gaussian_region_ids.npz') as z:region=z['region_id']
    assert len(region)==len(ply)==cfg['gaussianCount']
    for o in s['objects']:
        index=o['anchor_source_gaussian_index'];assert region[index]==o['gaussian_region_index']
        assert o['world_xyz']==[float(ply[k][index]) for k in ('x','y','z')]
        assert o['viewer_xyz']==[-o['world_xyz'][0],-o['world_xyz'][1],o['world_xyz'][2]]
    assert (root/s['reference_image']).is_file()
    backup=Path(manifest['backup']);protected=read(backup/'backup_manifest.json')['protected']
    assert all(sha(root/name)==h for name,h in protected.items())
    html=(code/'interactive_viewer.html').read_text(encoding='utf-8')
    body=next(b for b in re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>',html,re.S) if 'async function readMotionSnapshot' in b)
    reader=body[body.index('  async function readMotionSnapshot'):body.index('  async function initializeMotion')]
    loader=body[body.index('  async function initializeMotion'):body.index('  const requestedMode')]
    visible=re.search(r'  function isMotionLabelVisible\(object\) \{.*?\n  \}',body,re.S).group(0)
    poses=read(base/'camera_poses.json');assert len(poses)==1000
    js='''const vm=require('node:vm'),assert=require('node:assert/strict'),a=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const control={hidden:false},c={config:a.config,poses:new Map(a.frames.map(n=>[n,{}])),motionSnapshot:null,motionIndex:null,console,resolveAsset:p=>p,document:{getElementById:()=>control},fetch:async()=>({ok:true,json:async()=>a.snapshot})};vm.createContext(c);vm.runInContext(a.functions,c);
(async()=>{await c.initializeMotion();assert.equal(control.hidden,true);const shown=c.motionSnapshot.objects.filter(c.isMotionLabelVisible);assert.equal(shown.length,(a.expected.static||0)+(a.expected.dynamic||0));assert.equal(shown.filter(o=>o.final_state==='static').length,a.expected.static||0);assert.equal(shown.filter(o=>o.final_state==='dynamic').length,a.expected.dynamic||0);for(const id of a.changed)assert(shown.some(o=>o.object_id===id));console.log('Production seq10 loader PASS: '+JSON.stringify(a.expected)+'; uncertain hidden; no frame selector.');})().catch(e=>{console.error(e);process.exitCode=1});'''
    r=subprocess.run(['node','-e',js],input=json.dumps(dict(config=cfg,snapshot=s,expected=expected,changed=manifest['summary'].get('compound_review_changed_objects',[]),frames=[p['source_frame'] for p in poses],functions=reader+loader+visible)),capture_output=True,text=True,encoding='utf-8');assert r.returncode==0,r.stderr
    report=dict(status='PASS',scene=s['scene_id'],source_frames=[0,999],map_gaussians=len(ply),
                semantic_anchor_region_membership='16/16',production_reader='PASS',shown_static=expected.get('static',0),shown_dynamic=expected.get('dynamic',0),hidden_uncertain=expected.get('uncertain',0),
                original_maps_and_successful_seq09_files_unchanged=True,new_seq10_not_edited=True,
                temporal_scope='1000 source frames; original semantic votes integrated 950 mapper frames; Pi3MOS predictions cover full1000.',
                visual_review='User will inspect; no browser operated',motion_accuracy_verified=False)
    save_json(asset/'integration_check.json',report);print(r.stdout.strip());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
