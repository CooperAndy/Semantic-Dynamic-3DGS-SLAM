"""Run only after GPU allocation; never creates placeholder masks."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('WANDB_MODE', 'disabled')
from tum_window import DATA, MODEL, ROOT, ONEFORMER_SHA256, rows_for, sha256, oneformer_processor

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, choices=[30, 200], required=True)
    args = p.parse_args()
    rows, index = rows_for(args.frames)
    import torch
    assert torch.cuda.is_available(), 'GPU required; no masks written'
    assert sha256(MODEL/'pytorch_model.bin') == ONEFORMER_SHA256
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    processor = oneformer_processor()
    model = OneFormerForUniversalSegmentation.from_pretrained(MODEL, local_files_only=True).eval().cuda()
    assert model.config.id2label[12] == 'person'
    output = DATA/'seg_mask'
    output.mkdir(exist_ok=True)
    manifest = output/'manifest.json'
    meta = json.loads(manifest.read_text()) if manifest.exists() else {'model_id':'shi-labs/oneformer_ade20k_swin_large','model_revision':'4a5bac8e64f82681a12db2e151a4c2f4ce6092b2','model_weights_sha256':sha256(MODEL/'pytorch_model.bin'),'person_label':12,'masks':[]}
    assert meta['model_revision'] == '4a5bac8e64f82681a12db2e151a4c2f4ce6092b2'
    records = {x['file']:x for x in meta['masks']}
    for row in tqdm(rows):
        path = DATA/row['rgb_path']
        destination = output/path.name
        rgb_hash = sha256(path)
        if destination.exists():
            assert path.name in records and records[path.name]['rgb_sha256'] == rgb_hash
            assert records[path.name]['sha256'] == sha256(destination)
            continue
        image = Image.open(path).convert('RGB')
        inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            outputs = model(**inputs)
        labels = processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height,image.width)])[0].cpu().numpy().astype(np.uint8)
        assert labels.shape == (480,640) and labels.max() < 150
        tmp = destination.with_suffix('.tmp.png')
        Image.fromarray(labels).save(tmp)
        tmp.replace(destination)
        records[path.name] = {'file':path.name,'sha256':sha256(destination),'rgb_sha256':rgb_hash,'person_fraction':float((labels==12).mean())}
        meta['masks'] = list(records.values())
        meta['last_index_sha256'] = sha256(index)
        temp_meta = manifest.with_suffix('.tmp.json')
        temp_meta.write_text(json.dumps(meta, indent=2))
        temp_meta.replace(manifest)
    print(f'MASKS_READY: {args.frames}')

if __name__ == '__main__': main()
