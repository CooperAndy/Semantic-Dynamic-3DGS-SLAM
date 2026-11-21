"""Check the installed data/loader contract without controlling a browser."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--demo-root', required=True)
    a = p.parse_args(); root = Path(a.demo_root)
    code, asset = root/'源码', root/'scene-data/KITTI09_FixedStates'
    cfg = read(code/'scene-config.kitti09-orb-f200.json')
    snapshot = read(asset/'snapshot.json'); manifest = read(asset/'manifest.json')
    expected = read(asset/'tracks.json')
    backup = read(Path(manifest['backup'])/'backup_manifest.json')
    assert cfg['motionIndex'] is None
    assert cfg['motionSnapshot'] == 'scene-data/KITTI09_FixedStates/snapshot.json'
    assert snapshot['scene_id'] == cfg['sceneId']
    assert snapshot['source_map_sha256'] == manifest['source_map_sha256']
    assert snapshot['fixed_track_labels'] and snapshot['source_frames'] == [300,499]
    assert len(snapshot['objects']) == len(expected) == 125
    assert len({o['track_id'] for o in snapshot['objects']}) == 125
    raw = root/'scene-data/KITTI09_MotionDemo/frames'
    assert all(sha(raw/f['name']) == f['sha256'] for f in manifest['source_snapshot_hashes'])
    assert all(sha(root/name) == h for name,h in backup['protected'].items())
    for o in snapshot['objects']:
        e = expected[o['track_segment_id']]
        assert o['final_state'] == e['E_state']
        assert o['state'] == {'static':'静态','dynamic':'动态','uncertain':'不确定'}[o['final_state']]
        if o['display_estimate']:
            original = read(raw/f"{o['representative_source_frame']:06d}.json")
            source = next(v for v in original['objects'] if v['track_id'] == o['track_id'])
            assert o['viewer_xyz'] == source['viewer_xyz'] and o['world_xyz'] == source['world_xyz']
            assert o['anchor_verified'] is False
    reference = code/snapshot['reference_image']
    # Asset URLs resolve from the source directory via the Demo's scene-data route.
    assert (root/snapshot['reference_image']).is_file()
    html = (code/'interactive_viewer.html').read_text(encoding='utf-8')
    blocks = [s for s in re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>', html, re.S) if s.strip()]
    for s in blocks:
        r = subprocess.run(['node','--check'],input=s,text=True,capture_output=True,encoding='utf-8')
        assert r.returncode == 0, r.stderr
    r = subprocess.run(['node','--check',str(code/'t07.js')],text=True,capture_output=True,encoding='utf-8')
    assert r.returncode == 0, r.stderr
    source = next(s for s in blocks if 'async function readMotionSnapshot' in s)
    reader = source[source.index('  async function readMotionSnapshot'):source.index('  async function initializeMotion')]
    loader = source[source.index('  async function initializeMotion'):source.index('  const requestedMode')]
    # Execute the actual production snapshot reader and fixed-mode branch with a
    # minimal document/fetch adapter. This is a code contract check, not UI use.
    js = '''const vm=require('node:vm'); const assert=require('node:assert/strict');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const control={hidden:false}; const paths=[];
const context={config:input.config,poses:new Map([[input.snapshot.reference_frame,{}]]),
 motionSnapshot:null,motionIndex:null,console,
 document:{getElementById(id){assert.equal(id,'motion-frame-control');return control}},
 resolveAsset:p=>p,fetch:async p=>{paths.push(p);return {ok:true,json:async()=>input.snapshot}}};
vm.createContext(context);
vm.runInContext(input.functions,context);
(async()=>{await context.initializeMotion();assert.equal(control.hidden,true);
assert.equal(paths.length,1);assert.equal(paths[0],input.config.motionSnapshot);
assert.equal(context.motionSnapshot.objects.length,125);
assert.equal(context.motionSnapshot.fixed_track_labels,true);
console.log('Production reader accepted the fixed snapshot; frame selector stays hidden.');})().catch(e=>{console.error(e);process.exitCode=1});'''
    r = subprocess.run(['node','-e',js],input=json.dumps(dict(config=cfg,snapshot=snapshot,functions=reader+loader)),text=True,capture_output=True,encoding='utf-8')
    assert r.returncode == 0, r.stderr
    print(r.stdout.strip())
    catalog = read(code/'scene-catalog.json')
    old_catalog = read(Path(manifest['backup'])/'源码/scene-catalog.json')
    assert len(catalog['scenes']) == len(old_catalog['scenes'])
    assert catalog['defaultSceneId'] == old_catalog['defaultSceneId']
    assert [s for s in catalog['scenes'] if s['id'] != cfg['sceneId']] == [s for s in old_catalog['scenes'] if s['id'] != cfg['sceneId']]
    report = dict(status='PASS', production_loader='PASS', javascript_syntax='PASS',
                  original_motion_hashes='200/200 unchanged',
                  new_scene_config_and_predictions='unchanged',
                  representative_anchors='All copied from original old-map observations',
                  catalog='Old seq09 label replaced; other scenes and default unchanged',
                  browser_visual_check='Not performed: user will inspect')
    (asset/'integration_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
