"""CPU checks; CUDA remains hidden even when physical GPU devices exist."""
import argparse
import ast
import csv
import importlib
import json
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['WANDB_MODE'] = 'disabled'
os.environ['OMP_NUM_THREADS'] = '1'
from tum_window import ROOT, DATA, MODEL, rows_for, check_masks, sha256, oneformer_processor

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['inputs','native','models','droid_weights','scatter'])
    args = p.parse_args()
    import numpy as np
    import torch
    source = ROOT/'01_source_env/DG-SLAM'
    sys.path[:0] = [str(source), str(source/'dg_slam')]
    report = {'stage':args.stage,'gpu_nodes':[str(x) for x in Path('/dev').glob('nvidia*')], 'cuda_initialized_before':torch.cuda.is_initialized()}
    assert not report['cuda_initialized_before'], 'CPU checks must not initialize CUDA'
    report['torch_version']=torch.__version__
    report['target_arch']=os.environ.get('TORCH_CUDA_ARCH_LIST','UNSET')
    if args.stage == 'inputs':
        import cv2
        # Execute only author's CPU loader functions, without importing its GPU model.
        tree = ast.parse((source/'run_tum.py').read_text())
        keep = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['parse_list','associate_frames','loadtum']]
        scope = {'np':np,'torch':torch,'os':os}
        exec(compile(ast.Module(body=keep,type_ignores=[]),str(source/'run_tum.py'),'exec'),scope)
        images,depths,poses,masks = scope['loadtum'](str(DATA),frame_rate=32)
        assert len(images)==828
        gt = np.loadtxt(DATA/'groundtruth.txt')
        report['author_selected_frames'] = len(images)
        report['windows'] = []
        for frames in [30,200]:
            rows,index = rows_for(frames)
            for i,row in enumerate(rows):
                assert Path(images[i]) == DATA/row['rgb_path'] and Path(depths[i]) == DATA/row['depth_path']
                assert np.allclose(poses[i].numpy(),gt[int(row['gt_index']),1:])
            report['windows'].append({'frames':frames,'index_sha256':sha256(index),'author_prefix_match':True,'masks_present':check_masks(rows)})
        raw_valid = []
        for row in rows:
            image = cv2.imread(str(DATA/row['rgb_path']))
            depth = cv2.imread(str(DATA/row['depth_path']),cv2.IMREAD_UNCHANGED)
            assert image.shape==(480,640,3) and depth.shape==(480,640) and depth.dtype==np.uint16
            resized = cv2.resize(depth,(512,384)).astype(np.float32)/5000.
            assert np.isfinite(resized).all() and (resized>=0).all()
            raw_valid.append(float((depth>0).mean()))
        report['decoded_rgb_depth_pairs']=len(rows)
        report['valid_depth_fraction_min_max']=[min(raw_valid),max(raw_valid)]
        report['depth_units_per_meter']=5000
        report['seg_mask_directory_exists']=(DATA/'seg_mask').exists()
    elif args.stage == 'scatter':
        from torch_scatter import scatter_sum
        x=torch.tensor([1.,2.,3.],requires_grad=True)
        y=scatter_sum(x,torch.tensor([0,1,0]))
        assert torch.equal(y,torch.tensor([4.,2.]))
        y.square().sum().backward()
        assert torch.equal(x.grad,torch.tensor([8.,4.,8.]))
        report.update(scatter_cpu_forward_backward='PASS',gpu_execution_verified=False)
    elif args.stage == 'droid_weights':
        from collections import OrderedDict
        from modules.extractor import BasicEncoder
        from modules.gru import ConvGRU
        from modules.clipping import GradientClip
        # Original network definitions; only construction / state loading.
        # Do not import or mock native operators; no network forward is run.
        tree=ast.parse((source/'dg_slam/droid_net.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        scope={'np':np,'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'OrderedDict':OrderedDict,'BasicEncoder':BasicEncoder,'ConvGRU':ConvGRU,'GradientClip':GradientClip}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/'dg_slam/droid_net.py'),'exec'),scope)
        state=torch.load(ROOT/'01_source_env/weights/droid.pth',map_location='cpu',weights_only=True)
        state={k.replace('module.',''):v for k,v in state.items()}
        for key in ['update.weight.2.weight','update.weight.2.bias','update.delta.2.weight','update.delta.2.bias']:state[key]=state[key][:2]
        result=scope['DroidNet']().load_state_dict(state,strict=True)
        assert all(torch.isfinite(v).all() for v in state.values())
        report.update(droid_weights_strict_load=str(result),parameter_tensors=len(state),network_forward_executed=False,native_operators_imported=False,droid_sha256=sha256(ROOT/'01_source_env/weights/droid.pth'))
    elif args.stage == 'native':
        names = ['lietorch','lietorch_backends','lietorch_extras','torch_scatter','droid_backends','diff_gaussian_rasterization_depth_pose','faiss','faiss.contrib.torch_utils','cv2','open3d','mathutils','skimage','wandb','dg_slam.dg_model']
        report['imports']={}
        for name in names:
            module = importlib.import_module(name)
            report['imports'][name] = str(getattr(module,'__file__',None))
        from lietorch import SE3
        from torch_scatter import scatter_sum
        x=torch.tensor([[.1,.2,.3,.0,.0,.1]],dtype=torch.float32,requires_grad=True)
        actual=SE3.exp(x).log()
        assert torch.allclose(actual,x,atol=1e-5)
        actual.sum().backward()
        assert torch.allclose(x.grad,torch.ones_like(x),atol=1e-4)
        # Check a non-cancelling SE3 matrix loss against central differences.
        v=torch.tensor([[.1,-.2,.3,.04,-.02,.07]],dtype=torch.float64,requires_grad=True)
        def objective(z):
            m=SE3.exp(z).matrix()
            return m[0,0,3]+.2*m[0,1,3]+.3*m[0,1,0]
        objective(v).backward()
        finite=[]
        for j in range(6):
            delta=torch.zeros_like(v);delta[0,j]=1e-6
            finite.append(float((objective(v.detach()+delta)-objective(v.detach()-delta))/2e-6))
        assert torch.allclose(v.grad.flatten(),torch.tensor(finite,dtype=torch.float64),atol=1e-5,rtol=1e-4)
        report['lie_matrix_gradient_max_error']=float((v.grad.flatten()-torch.tensor(finite,dtype=torch.float64)).abs().max())
        assert torch.equal(scatter_sum(torch.tensor([1.,2.,3.]),torch.tensor([0,1,0])),torch.tensor([4.,2.]))
        import faiss, droid_backends
        assert hasattr(faiss,'StandardGpuResources') and hasattr(faiss,'index_cpu_to_gpu')
        report['faiss_gpu_api_symbols']=True
        for name in ['ba','frame_distance','projmap','depth_filter','iproj','altcorr_forward','altcorr_backward','corr_index_forward','corr_index_backward']: assert hasattr(droid_backends,name)
        report['lie_cpu_forward_backward']='PASS'
        report['scatter_cpu']='PASS'
        from droid_net import DroidNet
        state=torch.load(ROOT/'01_source_env/weights/droid.pth',map_location='cpu',weights_only=True)
        state={k.replace('module.',''):v for k,v in state.items()}
        for key in ['update.weight.2.weight','update.weight.2.bias','update.delta.2.weight','update.delta.2.bias']: state[key]=state[key][:2]
        result=DroidNet().load_state_dict(state,strict=True)
        report['droid_weights_strict_load']=str(result)
    elif args.stage == 'models':
        from transformers import OneFormerProcessor,OneFormerConfig,OneFormerForUniversalSegmentation
        from PIL import Image
        processor=oneformer_processor()
        rows,_=rows_for(30)
        image=Image.open(DATA/rows[0]['rgb_path']).convert('RGB')
        inputs=processor(images=image,task_inputs=['semantic'],return_tensors='pt')
        report['processor_tensor_shapes']={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}
        config=OneFormerConfig.from_pretrained(MODEL,local_files_only=True)
        assert config.id2label[12]=='person'
        # Metadata-only construction + mmap avoids copying ~879MB parameters.
        from accelerate import init_empty_weights
        with init_empty_weights(): model=OneFormerForUniversalSegmentation(config)
        state=torch.load(MODEL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
        result=model.load_state_dict(state,strict=True,assign=True)
        report.update(person_label=12,person_label_name=config.id2label[12],oneformer_state_strict_load=str(result),oneformer_forward_executed=False,model_sha256=sha256(MODEL/'pytorch_model.bin'),droid_sha256=sha256(ROOT/'01_source_env/weights/droid.pth'),model_revision='4a5bac8e64f82681a12db2e151a4c2f4ce6092b2',class_metadata_revision='4d683bd5bf84e9c8b5537dce306230bde409fe89')
    report['cuda_initialized_after']=torch.cuda.is_initialized()
    assert not report['cuda_initialized_after']
    report['status']='PASS'
    destination=Path(os.environ.get('DG_REPORT_DIR',str(ROOT/'05_reports/cpu_prepare')))
    destination.mkdir(parents=True,exist_ok=True)
    (destination/f'cpu_check_{args.stage}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
