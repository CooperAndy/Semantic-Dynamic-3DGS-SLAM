"""Auditable prefix loader: same preprocessing as DG-SLAM run_tum.py (R0)."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/dg_slam_chy_20261011')
DATA = ROOT/'02_data_masks/TUM/rgbd_dataset_freiburg3_walking_xyz'
MODEL = ROOT/'01_source_env/weights/oneformer_ade20k_swin_large'
ONEFORMER_SHA256 = 'c0b2fe11dfecee6f2f1f315f466946e96f4e94813f3f6d660ff3747b83c28cc9'

def oneformer_processor():
    # Transformers constructs ImageProcessor before applying from_pretrained
    # keyword overrides; supply local class metadata before construction.
    from transformers import OneFormerImageProcessor, OneFormerProcessor, CLIPTokenizer
    config=json.loads((MODEL/'preprocessor_config.json').read_text())
    config['repo_path']=str(ROOT/'01_source_env/weights')
    config['class_info_file']='ade20k_panoptic.json'
    config.pop('_max_size',None)  # Already represented in size.longest_edge.
    if 'reduce_labels' in config: config['do_reduce_labels']=config.pop('reduce_labels')
    processor=OneFormerImageProcessor(**config)
    tokenizer=CLIPTokenizer.from_pretrained(MODEL,local_files_only=True)
    return OneFormerProcessor(image_processor=processor,tokenizer=tokenizer)
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def rows_for(frames):
    name = {30:'smoke_first30.csv', 200:'dev_first200.csv'}[frames]
    index = ROOT/'02_data_masks'/name
    rows = list(csv.DictReader(index.open()))
    full = list(csv.DictReader((ROOT/'02_data_masks/official_loader_selected.csv').open()))
    assert len(rows) == frames and rows == full[:frames], 'index must equal author-selected prefix'
    for row in rows:
        for key in ['rgb_path', 'depth_path']:
            path = (DATA/row[key]).resolve()
            assert path.is_relative_to(DATA.resolve()) and path.is_file(), path
    return rows, index

def check_masks(rows, required=False):
    manifest = DATA/'seg_mask/manifest.json'
    present = []
    for row in rows:
        path = DATA/'seg_mask'/Path(row['rgb_path']).name
        if path.is_file():
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert mask is not None and mask.shape == (480, 640) and mask.dtype == np.uint8, path
            assert mask.max() < 150, 'Expected ADE20K class IDs (person=12), not binary masks'
            present.append(path.name)
    if required:
        assert len(present) == len(rows), f'Missing masks: {len(present)}/{len(rows)} present'
        assert manifest.is_file(), 'missing model/frame/checksum provenance'
        meta = json.loads(manifest.read_text())
        assert meta['model_revision'] == '4a5bac8e64f82681a12db2e151a4c2f4ce6092b2'
        assert meta['model_weights_sha256'] == ONEFORMER_SHA256 and meta['person_label'] == 12
        records = {x['file']: x for x in meta['masks']}
        for row in rows:
            name = Path(row['rgb_path']).name
            assert records[name]['sha256'] == sha256(DATA/'seg_mask'/name)
            assert records[name]['rgb_sha256'] == sha256(DATA/row['rgb_path'])
    return len(present)

def load_window(frames):
    rows, _ = rows_for(frames)
    check_masks(rows, required=True)
    gt = np.loadtxt(DATA/'groundtruth.txt')
    data = []
    for t, row in enumerate(rows):
        rgb = cv2.imread(str(DATA/row['rgb_path']))
        depth = cv2.imread(str(DATA/row['depth_path']), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(cv2.resize(rgb, (512, 384)), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb[None]).permute(0, 3, 1, 2)
        depth = torch.from_numpy(cv2.resize(depth, (512, 384)).astype(np.float32)/5000.)
        labels = cv2.imread(str(DATA/'seg_mask'/Path(row['rgb_path']).name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.dilate((labels == 12).astype(np.uint8), np.ones((8, 8), np.uint8), iterations=1)
        # R0 retains author's default INTER_LINEAR + uint8 conversion.
        mask = torch.from_numpy(cv2.resize(mask, (512, 384)).astype(np.uint8))
        intrinsics = .8*torch.tensor([535.4, 539.2, 320.1, 247.6])
        pose = torch.from_numpy(gt[int(row['gt_index']), 1:].astype(np.float32))
        data.append((t, image, depth, intrinsics, pose, mask))
    return data
