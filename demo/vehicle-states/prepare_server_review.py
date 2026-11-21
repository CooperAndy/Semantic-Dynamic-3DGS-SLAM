"""Verify prediction provenance and prepare unannotated RGB review material."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-build', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    build, output = Path(a.source_build), Path(a.output)
    output.mkdir(exist_ok=False)
    (output/'rgb').mkdir()
    manifest_path = build/'input_manifest.csv'
    manifest = list(csv.DictReader(manifest_path.open()))
    spec = json.loads((build/'input_spec.json').read_text())
    assert sha(manifest_path) == spec['input_manifest_sha256']
    assert len(manifest) == 200
    frames = build/'delivery/scene-data/KITTI09_MotionDemo/frames'
    comparisons, image_hashes = [], []
    for entry in manifest:
        local, source = int(entry['local_index']), int(entry['source_frame'])
        image = build/f'sequences/09/image_2/{local:06d}.png'
        assert sha(image) == entry['image_2_sha256'], image
        Image.open(image).convert('RGB').save(output/f'rgb/{source:06d}.jpg', quality=90)
        image_hashes.append(dict(source_frame=source, original_sha256=entry['image_2_sha256'], review_jpeg_sha256=sha(output/f'rgb/{source:06d}.jpg')))
        raw_path = build/f'annotations_20261107/motion/{local:06d}.json'
        raw = json.loads(raw_path.read_text())
        exported_path = frames/f'{source:06d}.json'
        exported = json.loads(exported_path.read_text())
        assert exported['reference_frame'] == source
        assert [v+source-local for v in raw['window']] == exported['window']
        assert len(raw['objects']) == len(exported['objects'])
        by_id = {o['track_id']: o for o in exported['objects']}
        for obj in raw['objects']:
            assert all(by_id[obj['track_id']][k] == v for k, v in obj.items()), (source, obj['track_id'])
        comparisons.append(dict(source_frame=source, raw_sha256=sha(raw_path), exported_sha256=sha(exported_path)))
    verification = json.loads((build/'field_verification.json').read_text())
    map_path = build/'runs/F1_appearance800/params.npz'
    map_sha = sha(map_path)
    assert map_sha == verification['F1_sha256']
    assert all(json.loads(p.read_text())['source_map_sha256'] == map_sha for p in frames.glob('*.json'))
    script_dir = build/'annotations_20261107/scripts'
    report = dict(image_alignment='200/200 original left image hashes match input_manifest',
                  prediction_alignment='200/200 exported objects and temporal windows match original Pi3MOS outputs',
                  source_map_sha256=map_sha, input_manifest_sha256=sha(manifest_path),
                  input_spec_sha256=sha(build/'input_spec.json'),
                  model_provenance='Existing Pi3MOS outputs reused; model weights are not reloaded or rehashed by this CPU experiment.',
                  source_scripts={p.name: sha(p) for p in script_dir.glob('*.py')},
                  predictions=comparisons, images=image_hashes)
    (output/'source_audit.json').write_text(json.dumps(report, indent=2))
    shutil.copy2(manifest_path, output/'source_input_manifest.csv')
    print(json.dumps({k: v for k, v in report.items() if k not in ('predictions', 'images', 'source_scripts')}, indent=2))


if __name__ == '__main__': main()
