"""One real-frame inference for environment acceptance; separate from experiment masks."""
import json
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import OneFormerForUniversalSegmentation
from tum_window import ROOT,DATA,MODEL,rows_for,oneformer_processor,sha256,ONEFORMER_SHA256

assert torch.cuda.get_device_capability()==(8,6)
assert sha256(MODEL/'pytorch_model.bin')==ONEFORMER_SHA256
rows,_=rows_for(30)
image=Image.open(DATA/rows[0]['rgb_path']).convert('RGB')
processor=oneformer_processor()
model=OneFormerForUniversalSegmentation.from_pretrained(MODEL,local_files_only=True).eval().cuda()
inputs=processor(images=image,task_inputs=['semantic'],return_tensors='pt').to('cuda')
torch.cuda.synchronize();started=time.monotonic()
with torch.inference_mode():
    outputs=model(**inputs)
    assert torch.isfinite(outputs.class_queries_logits).all()
    assert torch.isfinite(outputs.masks_queries_logits).all()
    labels=processor.post_process_semantic_segmentation(outputs,target_sizes=[(480,640)])[0].cpu().numpy()
torch.cuda.synchronize()
assert labels.shape==(480,640) and labels.min()>=0 and labels.max()<150
out=ROOT/'05_reports/3080ti_prepare'
Image.fromarray(labels.astype(np.uint8)).save(out/'oneformer_firstframe_labels.png')
result={'status':'PASS','gpu':torch.cuda.get_device_name(),'actual_network_forward':True,
        'frames':1,'source_rgb':rows[0]['rgb_path'],'rgb_sha256':sha256(DATA/rows[0]['rgb_path']),
        'labels_shape':list(labels.shape),'labels_present':np.unique(labels).tolist(),
        'person_fraction':float((labels==12).mean()),'inference_seconds':time.monotonic()-started,
        'max_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'experiment_mask_directory_created':False,
        'weights_sha256':ONEFORMER_SHA256}
(out/'oneformer_gpu_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
