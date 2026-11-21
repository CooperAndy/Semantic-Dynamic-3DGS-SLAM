"""Native visibility-adjoint semantic fusion; exposed engineering DEV only."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,factor_weights,native_render_variables,score_semantics
from native_semantic_render import render_mass,semantic_features,decode_mass

GROUPS=['S0','S1','S-G','S2','S2-H']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--splatam-root',required=True)
    ap.add_argument('--mode',choices=['smoke','full'],required=True);a=ap.parse_args()
    root=Path(a.root);sys.path.insert(0,a.splatam_root)
    import torch
    import torch.nn.functional as F
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    from utils.recon_helpers import setup_camera
    from utils.slam_helpers import transform_to_frame
    out=root/'02_基础因子DEV与Oracle诊断'/f'dev_sgbm_native_{a.mode}_20261007_v1'
    if out.exists():raise FileExistsError('preserve prior run; choose another version')
    out.mkdir();cache=out/'observations';cache.mkdir();states_dir=out/'states';states_dir.mkdir();renders=out/'renders';renders.mkdir()
    full_fusion=[i for i in range(200) if i%10!=9];full_eval=[i for i in range(200) if i%10==9]
    fusion=full_fusion if a.mode=='full' else [0,1,2]
    evaluation=full_eval if a.mode=='full' else [9]
    config={'status':'FROZEN_ENGINEERING_DEV_BEFORE_RUN','owner':'CHY','run_mode':a.mode,
        'fusion_local_frames':fusion,'eval_local_frames':evaluation,'original_offset':3000,
        'input_version':'kitti08_source_locked_20261007_v1_WITH_SGBM_ENGINEERING_DEPTH',
        'depth':'SGBM candidate, meters; explicit new version, not historical RAFT replacement',
        'map_depth':'native alpha-weighted camera-z mean; all fixed map Gaussians, no timestep pruning',
        'visibility':'exact native color adjoint per-Gaussian alpha/transmittance contributions, not center sampling',
        'shared_render_support':'independent native all-one feature alpha for all groups; no per-group denominator drift',
        'pixel_valid':'valid class 0..18, finite confidence 0..1, observed depth 0.5..80 m, map alpha>=0.05 and map depth>0',
        'minimum_gaussian_frame_alpha_mass':.001,'alpha_min':.05,'unknown_fraction_max':.5,
        'geometry_weight':'1 / (1 + (abs(z_map_mean-z_SGBM)/1.0m)^2)',
        'geometry_tolerance_abs_m':1.0,'relative_depth_tolerance':0.0,
        'frame_aggregation':'alpha-weighted hard-label distribution, normalized within frame; unit frame budget',
        'semantic_weight':'alpha-weighted mean of selected-class 2D confidence within same valid footprint',
        'geometric_weight':'alpha-weighted mean geometric reliability within same valid footprint',
        'S2':'product of separately aggregated semantic and geometric weights',
        'S2_H':'unit weight when product>=0.5, else zero; engineering initial threshold',
        'history':'OFF','support_unit':'distinct effective source frame per Gaussian',
        'checkpoint_every_fusion_frames':30,'base_exposure':'all 200 views historically exposed',
        'evaluation':'held-out-from-this-fusion teacher agreement only; independent GT unavailable',
        'O_Sem':'SKIPPED_NO_INDEPENDENT_GT','GT_subgroups':'NOT_AVAILABLE; no prediction masks substitute for GT',
        'quality_ranking':'NOT_AUTHORIZED_BY_EVIDENCE','online':'NO; offline fixed final map'}
    dump(out/'config.json',config)
    basepath=root/'inputs/base_map/params.npz';basehash=sha(basepath)
    assert basehash==json.loads((root/'inputs/base_map_manifest.json').read_text())['file_sha256']
    inputroot=root/'inputs/kitti08_source_locked_20261007_v1'
    inventory=list(csv.DictReader((root/'inputs/input_inventory_20261007.csv').open()))
    locks={(r['kind'],int(r['local_frame'])):r for r in inventory}
    for f in sorted(set(fusion+evaluation)):
        for kind in (['labels','confidence','depth_sgbm_candidate'] if f in fusion else ['labels']):
            p=Path(locks[kind,f]['stage3_reference']);assert sha(p)==locks[kind,f]['sha256']
    dump(out/'input_locks.json',[locks[k,f] for f in sorted(set(fusion+evaluation)) for k in (['labels','confidence','depth_sgbm_candidate'] if f in fusion else ['labels'])])
    for name in ['run_step02_dev.py','semantic_cpu.py','native_semantic_render.py']:
        p=Path(__file__).with_name(name);(out/name).write_bytes(p.read_bytes())
    # Verify installed renderer equals Step01-tested binary and wrappers.
    source_manifest=json.loads((root/'01_语义表示_初始化与渲染审计/native_audit_20261007_v1/source_manifest.json').read_text())
    for record in source_manifest[:4]:assert sha(record['source'])==record['sha256']
    dump(out/'native_dependencies.json',source_manifest[:4])
    with np.load(basepath,allow_pickle=False) as raw:base={k:raw[k] for k in raw.files}
    ids=base['source_gaussian_index'];n=len(ids)
    params={k:torch.as_tensor(base[k],device='cuda',dtype=torch.float32) for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']}
    camera=setup_camera(int(base['org_width']),int(base['org_height']),base['intrinsics'],base['w2c'])
    def variables(transformed,colors):
        return native_render_variables(params,transformed,colors,torch.zeros_like,lambda q:F.normalize(q,dim=1),torch.exp,torch.sigmoid)
    def forward(t,c):return Renderer(raster_settings=camera)(**variables(t,c))[0]
    def adjoint(t,masks):
        # Colors are independent feature values: d(sum mask*render)/d(color_g)
        # is the exact native alpha-weighted visible footprint sum for Gaussian g.
        cols=torch.zeros((n,3),device='cuda',requires_grad=True)
        rendered=forward(t,cols)
        return torch.autograd.grad(rendered,cols,grad_outputs=masks)[0].detach()
    torch.cuda.reset_peak_memory_stats();started=time.time();frame_stats=[]
    states={g:Evidence(ids,19) for g in GROUPS}
    def checkpoints(tag):
        for g,s in states.items():s.save(states_dir/f'{g}_{tag}.npz')
    # Real-scene adjoint identity checked BEFORE cache generation.
    with torch.no_grad():t=transform_to_frame(params,fusion[0],False,False)
    rng=torch.Generator(device='cuda');rng.manual_seed(42)
    masks=torch.rand((3,int(base['org_height']),int(base['org_width'])),generator=rng,device='cuda')
    weights=adjoint(t,masks)
    with torch.no_grad():
        cols=torch.rand((n,3),generator=rng,device='cuda')
        lhs=float((forward(t,cols)*masks).sum(dtype=torch.float64).cpu())
        rhs=float((cols*weights).sum(dtype=torch.float64).cpu())
    rel=abs(lhs-rhs)/max(abs(lhs),1);assert rel<1e-5,rel
    dump(out/'adjoint_validation.json',{'pass':True,'native_integral':lhs,'gaussian_adjoint_integral':rhs,'relative_error':rel,'tolerance':1e-5,'seed':42})
    del masks,weights,cols
    for pos,f in enumerate(fusion):
        frame_start=time.time()
        with torch.no_grad():
            transformed=transform_to_frame(params,f,False,False)
            z=transformed['means3D'][:,2]
            moments=forward(transformed,torch.stack([z,torch.ones_like(z),z*z],dim=1))
            alpha=moments[1];mapdepth=torch.where(alpha>0,moments[0]/alpha.clamp_min(1e-8),0)
            labels_np=np.load(Path(locks['labels',f]['stage3_reference']),allow_pickle=False)
            conf=torch.as_tensor(np.load(Path(locks['confidence',f]['stage3_reference']),allow_pickle=False),device='cuda',dtype=torch.float32)
            dep=torch.as_tensor(np.load(Path(locks['depth_sgbm_candidate',f]['stage3_reference']),allow_pickle=False),device='cuda',dtype=torch.float32)
            labels=torch.as_tensor(labels_np.astype(np.int64),device='cuda')
            valid=(labels>=0)&(labels<19)&torch.isfinite(conf)&(conf>=0)&(conf<=1)&torch.isfinite(dep)&(dep>=.5)&(dep<=80)&(alpha>=.05)&torch.isfinite(mapdepth)&(mapdepth>0)
            geo=1/(1+((mapdepth-dep).abs()/config['geometry_tolerance_abs_m'])**2)
        grads=[]
        # 19 votes + confidence mass + geometry mass + shared support mass.
        for start in range(0,22,3):
            mm=torch.zeros((3,*labels.shape),device='cuda')
            for j,index in enumerate(range(start,min(start+3,22))):
                signal=(labels==index).float() if index<19 else conf if index==19 else geo if index==20 else torch.ones_like(conf)
                mm[j]=valid.float()*signal
            grads.append(adjoint(transformed,mm).cpu().numpy())
        combined=np.concatenate(grads,axis=1)[:,:22]
        assert np.isfinite(combined).all() and combined.min()>=-1e-5
        combined=np.maximum(combined,0)
        denominator=combined[:,:19].sum(1);support_mass=combined[:,21]
        globalmass=float((alpha*valid).sum(dtype=torch.float64).cpu())
        sum_mass=float(denominator.sum(dtype=np.float64))
        assert abs(sum_mass-globalmass)/max(globalmass,1)<1e-4
        assert np.allclose(denominator,support_mass,atol=.002,rtol=1e-4)
        active=np.flatnonzero(denominator>=.001)
        distribution=combined[active,:19]/denominator[active,None]
        sem=np.clip(combined[active,19]/denominator[active],0,1)
        geometric=np.clip(combined[active,20]/denominator[active],0,1)
        version=hashlib.sha256(json.dumps([locks[k,f]['sha256'] for k in ['labels','confidence','depth_sgbm_candidate']]+[basehash],sort_keys=True).encode()).hexdigest()
        np.savez_compressed(cache/f'{f:06d}.npz',gaussian_id=ids[active],gaussian_index=active,distribution=distribution,
            semantic_reliability=sem,geometry_reliability=geometric,visible_alpha_mass=denominator[active],input_version=np.array(version))
        d=np.zeros((n,19),np.float32);d[active]=distribution
        observation={'dataset':'KITTI','sequence':'08','camera':'image_2','original_frame':f+3000}
        row={'local_frame':f,'original_frame':f+3000,'valid_pixels':int(valid.sum()),'associated_gaussians':len(active),
            'visible_mass_min':float(denominator[active].min()) if len(active) else None,'visible_mass_max':float(denominator[active].max()) if len(active) else None,
            'native_adjoint_total_relative_error':abs(sum_mass-globalmass)/max(globalmass,1)}
        for g,s in states.items():
            w=np.zeros(n,np.float64);w[active]=factor_weights(g,np.ones(len(active),bool),sem,geometric,.5)
            assert s.update_frame(observation,version,d,w)
            # Same source/version replay cannot increase any evidence or support.
            assert not s.update_frame(observation,version,d,w)
            row[g+'_accepted_gaussians']=int((w>0).sum())
        row['seconds']=time.time()-frame_start;frame_stats.append(row)
        del d,combined,grads,mm
        if (pos+1)%30==0:checkpoints(f'checkpoint_{pos+1:03d}')
        if (pos+1)%10==0 or pos+1==len(fusion):print(json.dumps({'phase':'fusion','complete':pos+1,'total':len(fusion),'seconds':time.time()-started}),flush=True)
    checkpoints('final')
    for g,s in states.items():
        loaded=Evidence.load(states_dir/f'{g}_final.npz',ids)
        assert all(np.array_equal(getattr(s,k),getattr(loaded,k)) for k in ['evidence','support','total_weight']) and s.processed==loaded.processed
        assert len(s.processed)==len(fusion) and int(s.support.max())<=len(fusion)
    dump(out/'frame_stats.json',frame_stats)
    evaluation_rows=[];curves=[]
    for g,s in states.items():
        features=semantic_features(s);folder=renders/g;folder.mkdir()
        for f in evaluation:
            with torch.no_grad():
                transformed=transform_to_frame(params,f,False,False)
                shared_alpha=forward(transformed,torch.ones_like(params['rgb_colors']))[0].cpu().numpy()
                decoded=decode_mass(render_mass(params,transformed,features,camera),reference_alpha=shared_alpha)
            teacher=np.load(Path(locks['labels',f]['stage3_reference']),allow_pickle=False)
            labelled=decoded['label']>=0;supported=decoded['alpha']>=.05
            compare=score_semantics(teacher,decoded['label'],19)
            row={'group':g,'local_frame':f,'original_frame':f+3000,'pixels':teacher.size,'labelled_pixels':int(labelled.sum()),
                'unknown_pixels':int((decoded['label']==-1).sum()),'no_map_pixels':int((decoded['label']==-2).sum()),
                'coverage_full_image':float(labelled.mean()),'coverage_map_support':float(labelled.sum()/max(supported.sum(),1)),
                'teacher_agree_pixels':int((decoded['label']==teacher).sum()),
                'conditional_teacher_agreement':float((decoded['label'][labelled]==teacher[labelled]).mean()) if labelled.any() else None,
                'teacher_IoU_diagnostic':compare['miou_gt_present_classes'],'teacher_confusion':compare['classes'],
                'score_definition':'uncalibrated max class mass / total map alpha',
                'metric_scope':'PREDICTION_TEACHER_DIAGNOSTIC_NOT_GT_ACCURACY'}
            evaluation_rows.append(row)
            for threshold in np.linspace(0,1,21):
                chosen=labelled&(decoded['score']>=threshold)
                curves.append({'group':g,'local_frame':f,'threshold':float(threshold),'accepted':int(chosen.sum()),
                    'coverage':float(chosen.mean()),'teacher_disagreements':int((decoded['label'][chosen]!=teacher[chosen]).sum()),
                    'scope':'fixed engineering thresholds, ties accepted together; teacher disagreement, not GT risk'})
            # Compact outputs retain decisions, scores and support; full numeric mass only at representative view 99.
            arrays={k:v for k,v in decoded.items() if k!='class_mass' or f==99}
            np.savez_compressed(folder/f'{f:06d}.npz',**arrays)
        print(json.dumps({'phase':'evaluation','group':g,'views':len(evaluation)}),flush=True)
    summary=[]
    for g,s in states.items():
        rows=[r for r in evaluation_rows if r['group']==g];pix=sum(r['pixels'] for r in rows);lab=sum(r['labelled_pixels'] for r in rows)
        rejected=sum(r['associated_gaussians']-r[g+'_accepted_gaussians'] for r in frame_stats);assoc=sum(r['associated_gaussians'] for r in frame_stats)
        summary.append({'group':g,'gaussians_with_evidence':int((s.evidence.sum(1)>0).sum()),'fraction_gaussians_with_evidence':float((s.evidence.sum(1)>0).mean()),
            'support_max':int(s.support.max()),'processed_unique_source_frames':len(s.processed),'distinct_effective_observations':int(s.support.sum(dtype=np.uint64)),
            'rejected_observation_fraction':rejected/max(assoc,1),'eval_pixel_coverage':lab/pix,
            'eval_conditional_teacher_agreement':sum(r['teacher_agree_pixels'] for r in rows)/lab if lab else None,
            'eval_unknown_fraction':sum(r['unknown_pixels'] for r in rows)/pix,'eval_no_map_fraction':sum(r['no_map_pixels'] for r in rows)/pix,
            'final_state_bytes':(states_dir/f'{g}_final.npz').stat().st_size,'state_sha256':sha(states_dir/f'{g}_final.npz')})
    assert sha(basepath)==basehash
    dump(out/'evaluation_rows.json',evaluation_rows);dump(out/'teacher_disagreement_coverage.json',curves)
    result={'status':'COMPLETE_ENGINEERING_DEV_NOT_SCIENTIFIC_ACCEPTANCE','mode':a.mode,'groups':summary,
        'fusion_frames':len(fusion),'evaluation_views':len(evaluation),'base_sha256':basehash,'base_unchanged':True,
        'all_source_frames_unique_and_replay_noop':True,'all_final_states_identity_roundtrip':True,
        'wall_seconds':time.time()-started,'peak_torch_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
        'association_cache_bytes':sum(p.stat().st_size for p in cache.glob('*.npz')),'independent_GT':'UNAVAILABLE',
        'O_Sem':'SKIPPED','GT_risk_coverage':'NOT_EVALUATED','GT_subgroups':'NOT_EVALUATED',
        'depth_version':'SGBM_ENGINEERING_NEW_VERSION','method_quality_ranking':'NOT_ESTABLISHED'}
    dump(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
