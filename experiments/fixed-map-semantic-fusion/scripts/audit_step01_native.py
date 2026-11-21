"""Step01 native CUDA audit. Synthetic semantic labels are engineering probes."""
import argparse,hashlib,json,sys,time,shutil,platform
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,native_render_variables,UNKNOWN,NO_MAP
from native_semantic_render import semantic_features,decode_mass,render_mass

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_hash(v):
    h=hashlib.sha256();h.update(str(v.dtype).encode());h.update(json.dumps(list(v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes());return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--splatam-root',required=True)
    args=parser.parse_args();root=Path(args.root);src=Path(args.splatam_root)
    out=root/'01_语义表示_初始化与渲染审计/native_audit_20261007_v1'
    if out.exists():raise FileExistsError('new run required')
    out.mkdir();sys.path.insert(0,str(src))
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import diff_gaussian_rasterization as dgr
    from utils.recon_helpers import setup_camera
    from utils.slam_helpers import transform_to_frame,transformed_params2rendervar
    # Freeze tolerances and engineering thresholds BEFORE evaluating outputs.
    config={'rgb_atol':1e-6,'rgb_rtol':1e-5,'alpha_atol':5e-5,'alpha_rtol':1e-4,
        'permutation_atol':1e-6,'alpha_min':.05,'unknown_fraction_max':.5,
        'local_frames':[0,99,199],'original_frames':[3000,3099,3199],
        'geometry_mode':'fixed final map; all Gaussians, no timestep pruning',
        'purpose':'ENGINEERING_ONLY; synthetic labels are not predictions or GT'}
    (out/'frozen_config.json').write_text(json.dumps(config,indent=2))
    sources=[src/'utils/recon_helpers.py',src/'utils/slam_helpers.py',Path(dgr.__file__),Path(dgr._C.__file__),
        Path(__file__),Path(__file__).with_name('native_semantic_render.py'),Path(__file__).with_name('semantic_cpu.py')]
    snap=out/'source_snapshots';snap.mkdir()
    source_manifest=[]
    for i,p in enumerate(sources):
        name=f'{i:02d}_{p.name}';shutil.copy2(p,snap/name)
        source_manifest.append({'source':str(p),'snapshot':str(snap/name),'sha256':sha(p)})
    (out/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2))
    result={'status':'RUNNING','config':config,'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'python':platform.python_version(),'controlled':[],'real_frames':[],'groups':[]}
    torch.cuda.reset_peak_memory_stats();start=time.time()
    def var(p,t,c):return native_render_variables(p,t,c,torch.zeros_like,lambda q:F.normalize(q,dim=1),torch.exp,torch.sigmoid)
    def direct(p,t,c,camera):return dgr.GaussianRasterizer(raster_settings=camera)(**var(p,t,c))[0].cpu().numpy()
    def assertclose(a,b,atol,rtol=0):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.allclose(a,b,atol=atol,rtol=rtol),float(np.max(np.abs(a-b)))
        return float(np.max(np.abs(a-b)))
    def image(a,name):
        a=np.clip(a,0,1);Image.fromarray((a.transpose(1,2,0)*255).astype(np.uint8)).save(out/name)
    with torch.no_grad():
        small=setup_camera(129,129,np.array([[100,0,64],[0,100,64],[0,0,1]],float),np.eye(4))
        p={'means3D':torch.tensor([[0,0,2],[0,0,4]],device='cuda',dtype=torch.float32),
           'unnorm_rotations':torch.tensor([[1,0,0,0],[1,0,0,0]],device='cuda',dtype=torch.float32),
           'log_scales':torch.tensor([[np.log(.12)],[np.log(.24)]],device='cuda',dtype=torch.float32),
           'logit_opacities':torch.logit(torch.tensor([[.99],[.99]],device='cuda'))}
        t={k:p[k] for k in ['means3D','unnorm_rotations']}
        features=np.zeros((2,20),np.float32);features[0,19]=1;features[1,0]=1
        mass=render_mass(p,t,features,small);decoded=decode_mass(mass)
        native_alpha=direct(p,t,torch.ones((2,3),device='cuda'),small)[0]
        err=assertclose(decoded['alpha'],native_alpha,config['alpha_atol'],config['alpha_rtol'])
        assert int(decoded['label'][64,64])==UNKNOWN
        # A negative control deliberately drops unknown opacity, reproducing old bug.
        buggy={k:v.clone() for k,v in p.items()};buggy['logit_opacities'][0]=-100
        leaking=decode_mass(render_mass(buggy,t,features,small))
        assert int(leaking['label'][64,64])==0
        bg=direct(p,t,torch.tensor([[0,0,0],[1,0,0]],device='cuda',dtype=torch.float32),small)[0,64,64]
        assert mass[0,64,64]<.02 and leaking['class_mass'][0,64,64]>.98
        result['controlled'].append({'test':'unknown_foreground_occludes_labelled_background','pass':True,
            'center_unknown_mass':float(mass[19,64,64]),'center_background_class_mass':float(bg),
            'negative_control_label':int(leaking['label'][64,64]),'alpha_max_error':err})
        perm=np.array([1,0]);pp={k:v[torch.tensor(perm,device='cuda')] for k,v in p.items()}
        tt={k:pp[k] for k in t}
        err=assertclose(render_mass(pp,tt,features[perm],small),mass,config['permutation_atol'])
        result['controlled'].append({'test':'array_order_does_not_replace_native_depth_sort','pass':True,'max_error':err})
        unknown_features=features.copy();unknown_features[:]=0;unknown_features[:,19]=1
        unknown=decode_mass(render_mass(p,t,unknown_features,small))
        assert (unknown['label'][unknown['alpha']>=.05]==UNKNOWN).all()
        low={k:v.clone() for k,v in p.items()};low['logit_opacities'][:]=torch.logit(torch.tensor(.01,device='cuda'))
        low_d=decode_mass(render_mass(low,t,features,small))
        assert (low_d['label']==NO_MAP).all()
        result['controlled'].append({'test':'all_unknown_and_low_support_are_distinct','pass':True})
        # Mixed evidence exercises every class chunk and sum of 20 channels.
        mixed=np.tile(np.arange(1,21,dtype=np.float32),(2,1));mixed/=mixed.sum(1,keepdims=True)
        mm=render_mass(p,t,mixed,small)
        err=assertclose(mm.sum(0),native_alpha,config['alpha_atol'],config['alpha_rtol'])
        for i in range(20):assertclose(mm[i],native_alpha*mixed[0,i],config['alpha_atol'],config['alpha_rtol'])
        result['controlled'].append({'test':'20_numeric_channels_match_native_alpha','pass':True,'max_error':err})
        np.savez_compressed(out/'controlled_arrays.npz',mass=mass,leaking_mass=leaking['class_mass'],native_alpha=native_alpha,unknown_label=unknown['label'],low_label=low_d['label'])
        image(np.stack([mass[19],mass[0],np.zeros_like(native_alpha)]),'controlled_unknown_red_background_green.png')
        base_file=root/'inputs/base_map/params.npz';expected=json.loads((root/'inputs/base_map_manifest.json').read_text())
        before=sha(base_file);assert before==expected['file_sha256']
        with np.load(base_file,allow_pickle=False) as npz:base={k:npz[k] for k in npz.files}
        assert all(array_hash(v)==expected['fields'][k]['array_sha256'] for k,v in base.items())
        ids=base['source_gaussian_index'];assert len(np.unique(ids))==len(ids)
        camera=setup_camera(int(base['org_width']),int(base['org_height']),base['intrinsics'],base['w2c'])
        keys=['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']
        baseline={k:torch.as_tensor(base[k],device='cuda',dtype=torch.float32) for k in keys}
        reference={}
        for f in config['local_frames']:
            transformed=transform_to_frame(baseline,f,False,False)
            reference[f]=dgr.GaussianRasterizer(raster_settings=camera)(**transformed_params2rendervar(baseline,transformed))[0].cpu().numpy()
            image(reference[f],f'{f:06d}_rgb_reference.png')
        runs=out/'initializations';runs.mkdir()
        for group in ['S0','S1','S-G','S2','S2-H']:
            # Create fresh semantic state; never import historic base semantic labels.
            state=Evidence(ids,19);state_path=runs/f'{group}_semantic_state.npz';state.save(state_path)
            restored=Evidence.load(state_path,ids)
            assert not restored.evidence.any() and not restored.support.any() and not restored.total_weight.any() and restored.processed=={}
            gp={k:v.copy() for k,v in base.items()}
            assert all(np.array_equal(gp[k],base[k],equal_nan=True) for k in base)
            params={k:torch.as_tensor(gp[k],device='cuda',dtype=torch.float32) for k in keys}
            errors=[]
            for f in config['local_frames']:
                transformed=transform_to_frame(params,f,False,False)
                rgb=direct(params,transformed,params['rgb_colors'],camera)
                errors.append(assertclose(rgb,reference[f],config['rgb_atol'],config['rgb_rtol']))
            result['groups'].append({'group':group,'zero_evidence_support_weight':True,'history':'disabled; no stored conflict field or imported history',
                'identity_roundtrip':True,'all_base_parameter_fields_equal':True,'state_sha256':sha(state_path),
                'rgb_max_abs_errors':errors,'pass':True})
            del params,gp,state,restored
        # Empty and deterministic artificial semantic states exercise real map channels.
        state=Evidence(ids,19);features=semantic_features(state)
        for f in config['local_frames']:
            transformed=transform_to_frame(baseline,f,False,False)
            mass=render_mass(baseline,transformed,features,camera);d=decode_mass(mass)
            alpha=direct(baseline,transformed,torch.ones_like(baseline['rgb_colors']),camera)[0]
            err=assertclose(d['alpha'],alpha,config['alpha_atol'],config['alpha_rtol'])
            assert (d['label'][d['alpha']>=.05]==UNKNOWN).all() and (d['class_mass']==0).all()
            # Synthetic labels, deliberately not copied from previous semantic map.
            diagnostic=np.zeros_like(features);diagnostic[np.arange(len(ids)),ids%20]=1
            diag=render_mass(baseline,transformed,diagnostic,camera)
            diagerr=assertclose(diag.sum(0),alpha,config['alpha_atol'],config['alpha_rtol'])
            np.savez_compressed(out/f'{f:06d}_semantic_empty.npz',**d)
            np.savez_compressed(out/f'{f:06d}_semantic_artificial_probe.npz',**decode_mass(diag))
            result['real_frames'].append({'local_frame':f,'original_frame':f+3000,'all_empty_supported_pixels_unknown':True,
                'supported_pixels':int((alpha>=.05).sum()),'no_support_pixels':int((alpha<.05).sum()),
                'alpha_max_error_empty':err,'alpha_max_error_artificial':diagerr,'pass':True})
        assert sha(base_file)==before
        result.update(status='PASS_STEP01_NATIVE_ENGINEERING',num_gaussians=len(ids),base_sha256=before,
            base_hash_before_after_equal=True,wall_seconds=time.time()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
            science_quality_evaluated=False,visible_association_cache='NOT_CREATED; fusion-stage dependency',
            scope_limit='Three real views plus controlled scene, not all 200 views or semantic accuracy acceptance')
    (out/'audit_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
