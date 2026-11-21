"""Seal engineering methods and independently recount exposed DEV outputs.

This audit never promotes teacher predictions to GT or consumed DEV to Test.
No torch/CUDA imports. --verify-only reruns from private frozen teacher copies.
"""
import argparse,csv,hashlib,json,shutil,time,importlib.util
from pathlib import Path
import numpy as np

GROUPS=['S0','S1','S-G','S2','S2-H']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def array_sha(v):
    h=hashlib.sha256();h.update(str(v.dtype).encode());h.update(json.dumps(list(v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes());return h.hexdigest()
def write_csv(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--verify-only',action='store_true');ap.add_argument('--replay-cache',action='store_true');a=ap.parse_args()
    root=Path(a.root);step=root/'03_方法冻结与隔离评价';freeze=step/'engineering_freeze_20261007_v1'
    run=root/'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1'
    if not a.verify_only:
        if freeze.exists():raise FileExistsError('preserve sealed freeze, use another version')
        freeze.mkdir();core=freeze/'core';core.mkdir();copies=core/'teacher_inputs';copies.mkdir();sources=core/'source_snapshots';sources.mkdir()
        config=read(run/'config.json');result=read(run/'result.json')
        assert result['status']=='COMPLETE_ENGINEERING_DEV_NOT_SCIENTIFIC_ACCEPTANCE'
        # Production batch code and later audited public CLI are separate versions.
        for name in ['run_step02_dev.py','semantic_cpu.py','native_semantic_render.py']:
            shutil.copy2(run/name,sources/('batch_'+name))
        cli=root/'02_基础因子DEV与Oracle诊断/cli_shared_alpha_check_20261007_v1/native_semantic_render.py'
        shutil.copy2(cli,sources/'audited_public_cli.py');shutil.copy2(Path(__file__),sources/'freeze_step03_cpu.py')
        for source,name in [(root/'inputs/base_map_manifest.json','base_map_manifest.json'),
            (root/'00_输入与基图_GT曝光审计/reaudit_20261007_v1/id2label.json','id2label.json'),
            (run/'native_dependencies.json','native_dependencies.json'),(run/'adjoint_validation.json','adjoint_validation.json')]:
            shutil.copy2(source,core/name)
        targets=[];locks=read(run/'input_locks.json');eval_frames=config['eval_local_frames']
        for f in eval_frames:
            record=next(r for r in locks if r['kind']=='labels' and r['local_frame']==str(f))
            source=Path(record['stage3_reference']);assert sha(source)==record['sha256']
            dest=copies/f'{f:06d}.npy';shutil.copy2(source,dest)
            targets.append({'local_frame':f,'original_frame':f+3000,'source_file':str(source),
                'frozen_file':dest.relative_to(core).as_posix(),'sha256':sha(dest),'role':'PREDICTION_TEACHER_NOT_GT'})
        dump(core/'teacher_targets.json',targets)
        dump(core/'frozen_method_config.json',config)
        dump(core/'frame_roles.json',{'geometry_rgb_prior_semantic_exposure_original':list(range(3000,3200)),
            'fusion_local':config['fusion_local_frames'],'fusion_original':[3000+i for i in config['fusion_local_frames']],
            'consumed_DEV_local':eval_frames,'consumed_DEV_original':[3000+i for i in eval_frames],
            'isolated_final_local':[],'role':'HISTORICALLY_EXPOSED_DEV; final evaluation NOT allocated',
            'development_scene_count':1,'development_window_count':1,
            'descriptive_blocks':[[eval_frames[i] for i in range(b*5,(b+1)*5)] for b in range(4)],
            'independence_note':'blocks are correlated parts of one trajectory; no independent-scene CI'})
        dump(core/'state_and_render_locks.json',{
            'base_file':'inputs/base_map/params.npz','base_sha256':result['base_sha256'],'base_gaussians':907488,
            'states':{g:{'file':(run/'states'/f'{g}_final.npz').relative_to(root).as_posix(),'sha256':sha(run/'states'/f'{g}_final.npz')} for g in GROUPS},
            'observations':[{'local_frame':f,'file':(run/'observations'/f'{f:06d}.npz').relative_to(root).as_posix(),'sha256':sha(run/'observations'/f'{f:06d}.npz')} for f in config['fusion_local_frames']],
            'renders':[{ 'group':g,'local_frame':f,'file':(run/'renders'/g/f'{f:06d}.npz').relative_to(root).as_posix(),
                'sha256':sha(run/'renders'/g/f'{f:06d}.npz')} for g in GROUPS for f in eval_frames]})
        dump(core/'scientific_gate.json',{'status':'NOT_ELIGIBLE_FOR_SCIENTIFIC_FREEZE_OR_FINAL_EVALUATION',
            'independent_semantic_GT':'UNAVAILABLE','isolated_final_split':'UNALLOCATED; existing 20 views already consumed as DEV',
            'quality_winner':None,'default_reference':'S0 for baseline reproducibility, not performance winner',
            'all_five_methods_retained':True,'O_Sem':'NOT_RUN','GT_mIoU':'NOT_RUN','GT_risk_coverage':'NOT_RUN','GT_subgroups':'NOT_DEFINED',
            'minimum_quality_gain':None,'maximum_coverage_loss':None,'critical_subgroup_constraints':None,
            'reason_unset':'No independent GT/DEV support; do not pick favorable limits after seeing teacher diagnostics',
            'next_allowed':'engineering packaging and display; science needs new GT/protocol and final split audit',
            'scope':'fixed final map, offline; not complete online SLAM, dynamic reconstruction or tracking gain'})
        core_manifest={p.relative_to(core).as_posix():sha(p) for p in core.rglob('*') if p.is_file()}
        dump(freeze/'core_SHA256.json',core_manifest)
        dump(freeze/'freeze_record.json',{'date':'2026-10-07','owner':'CHY','status':'SEALED_ENGINEERING_CONFIGURATION_ALL_FIVE',
            'core_manifest_sha256':sha(freeze/'core_SHA256.json'),'scientific_method_selection':False,
            'protocol_changed_after_DEV':False,'note':'Existing engineering configuration sealed; quality criteria remain unset'})
    core=freeze/'core';core_manifest=read(freeze/'core_SHA256.json')
    assert all(sha(core/n)==h for n,h in core_manifest.items())
    assert sha(freeze/'core_SHA256.json')==read(freeze/'freeze_record.json')['core_manifest_sha256']
    locks=read(core/'state_and_render_locks.json');config=read(core/'frozen_method_config.json');roles=read(core/'frame_roles.json')
    assert not set(roles['fusion_local'])&set(roles['consumed_DEV_local'])
    assert roles['isolated_final_local']==[] and len(roles['fusion_local'])==180 and len(roles['consumed_DEV_local'])==20
    assert sha(root/locks['base_file'])==locks['base_sha256']
    base_manifest=read(core/'base_map_manifest.json')
    with np.load(root/locks['base_file'],allow_pickle=False) as p:
        ids=p['source_gaussian_index'];assert len(np.unique(ids))==907488
        assert all(array_sha(p[k])==base_manifest['fields'][k]['array_sha256'] for k in p.files)
    state_checks=[]
    for g in GROUPS:
        record=locks['states'][g];assert sha(root/record['file'])==record['sha256']
        with np.load(root/record['file'],allow_pickle=False) as s:
            assert np.array_equal(s['gaussian_id'],ids)
            e=s['evidence'];w=s['total_weight'];support=s['support_count'];processed=read_json_string(s['processed_json'])
            frames=[json.loads(k)['original_frame'] for k in processed]
            assert set(frames)==set(roles['fusion_original']) and len(frames)==180
            assert np.isfinite(e).all() and (e>=0).all() and np.isfinite(w).all() and (w>=0).all()
            assert np.allclose(e.sum(1,dtype=np.float64),w,atol=1e-4,rtol=1e-5)
            assert np.array_equal(support==0,w==0) and support.max()<=180
        state_checks.append({'group':g,'identity_match':True,'exact_fusion_frame_ledger':True,'evidence_weight_conservation':True,'support_max':int(support.max())})
    if a.replay_cache:
        replay_started=time.time()
        spec=importlib.util.spec_from_file_location('frozen_semantic',core/'source_snapshots/batch_semantic_cpu.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        states={g:module.Evidence(ids,19) for g in GROUPS}
        for pos,record in enumerate(locks['observations']):
            assert sha(root/record['file'])==record['sha256']
            with np.load(root/record['file'],allow_pickle=False) as c:
                active=c['gaussian_index'];assert np.array_equal(ids[active],c['gaussian_id'])
                distributions=np.zeros((len(ids),19),np.float32);distributions[active]=c['distribution']
                version=str(c['input_version'])
                for g,s in states.items():
                    weights=np.zeros(len(ids),np.float64)
                    weights[active]=module.factor_weights(g,np.ones(len(active),bool),c['semantic_reliability'],c['geometry_reliability'],.5)
                    assert s.update_frame({'dataset':'KITTI','sequence':'08','camera':'image_2','original_frame':record['local_frame']+3000},version,distributions,weights)
            if (pos+1)%30==0:print(json.dumps({'phase':'CPU_cached_observation_replay','frames':pos+1,'total':180}),flush=True)
        replay_checks=[]
        for g,s in states.items():
            old=module.Evidence.load(root/locks['states'][g]['file'],ids)
            equality={k:bool(np.array_equal(getattr(s,k),getattr(old,k))) for k in ['evidence','support','total_weight']}
            assert all(equality.values()) and s.processed==old.processed
            replay_checks.append({'group':g,'arrays_equal':equality,'ledger_equal':True})
        replay={'status':'PASS_FULL_180_FRAME_REPLAY','checks':replay_checks,'frames':180,'groups':5,'seconds':time.time()-replay_started,
            'gpu_used':False,'scope':'replay of frozen native observations; not regenerating visibility or scientific evaluation'}
        if not a.verify_only:dump(freeze/'full_cache_replay.json',replay)
        del states,old,distributions,weights
    targets={t['local_frame']:t for t in read(core/'teacher_targets.json')}
    for t in targets.values():assert sha(core/t['frozen_file'])==t['sha256']
    old_rows={(r['group'],r['local_frame']):r for r in read(run/'evaluation_rows.json')}
    original_result=read(run/'result.json');original_groups={r['group']:r for r in original_result['groups']}
    recounted=[];blocks=[];combined={g:np.zeros((19,21),np.int64) for g in GROUPS};reference_masks={}
    started=time.time()
    for g in GROUPS:
        for f in roles['consumed_DEV_local']:
            record=next(r for r in locks['renders'] if r['group']==g and r['local_frame']==f)
            assert sha(root/record['file'])==record['sha256']
            with np.load(root/record['file'],allow_pickle=False) as p:
                label=p['label'];alpha=p['alpha'];score=p['score'];unknown_mass=p['unknown_mass']
                assert label.shape==alpha.shape==score.shape==(370,1226)
                assert np.isfinite(alpha).all() and np.isfinite(score).all() and np.isfinite(unknown_mass).all()
                assert label.min()>=-2 and label.max()<19 and (alpha>=0).all()
                support_mask=alpha>=.05;assert np.array_equal(label==-2,~support_mask)
                if f not in reference_masks:reference_masks[f]=support_mask.copy()
                else:assert np.array_equal(reference_masks[f],support_mask)
            teacher=np.load(core/targets[f]['frozen_file'],allow_pickle=False)
            assert teacher.shape==label.shape and teacher.min()>=0 and teacher.max()<19
            # Independent implementation: a single 19 x 21 integer confusion matrix.
            # Prediction columns: no-map, unknown, then classes 0..18. No GT claim.
            cm=np.bincount((teacher.astype(np.int64)*21+label.astype(np.int64)+2).ravel(),minlength=19*21).reshape(19,21)
            known=int(cm[:,2:].sum());agreed=int(sum(cm[c,c+2] for c in range(19)))
            ious=[]
            for c in range(19):
                overlap=int(cm[c,c+2]);teacher_count=int(cm[c].sum());prediction_count=int(cm[:,c+2].sum())
                if teacher_count:ious.append(overlap/(teacher_count+prediction_count-overlap))
            row={'group':g,'local_frame':f,'original_frame':f+3000,'pixels':int(cm.sum()),'labelled_pixels':known,
                'teacher_agree_pixels':agreed,'unknown_pixels':int(cm[:,1].sum()),'no_map_pixels':int(cm[:,0].sum()),
                'coverage_full_image':known/int(cm.sum()),'conditional_teacher_agreement':agreed/known if known else None,
                'teacher_IoU_diagnostic':float(np.mean(ious)) if ious else None,
                'scope':'RECOUNT_OF_ALREADY_CONSUMED_DEV; NOT_NEW_FINAL_EVALUATION'}
            old=old_rows[g,f]
            for k in ['pixels','labelled_pixels','teacher_agree_pixels','unknown_pixels','no_map_pixels']:assert row[k]==old[k]
            for k in ['coverage_full_image','conditional_teacher_agreement','teacher_IoU_diagnostic']:
                assert row[k]==old[k] if row[k] is None else abs(row[k]-old[k])<1e-12
            recounted.append(row);combined[g]+=cm
        for b,frames in enumerate(roles['descriptive_blocks']):
            rows=[r for r in recounted if r['group']==g and r['local_frame'] in frames];pix=sum(r['pixels'] for r in rows);known=sum(r['labelled_pixels'] for r in rows)
            blocks.append({'group':g,'block':b+1,'original_first':frames[0]+3000,'original_last':frames[-1]+3000,
                'views':5,'coverage':known/pix,'conditional_teacher_agreement':sum(r['teacher_agree_pixels'] for r in rows)/known if known else None,
                'scope':'CORRELATED_TRAJECTORY_BLOCK_DESCRIPTIVE_ONLY; not independent replicate or CI'})
    aggregate=[]
    for g,cm in combined.items():
        known=int(cm[:,2:].sum());agreed=int(sum(cm[c,c+2] for c in range(19)));pix=int(cm.sum());old=original_groups[g]
        coverage=known/pix;agreement=agreed/known
        assert abs(coverage-old['eval_pixel_coverage'])<1e-12 and abs(agreement-old['eval_conditional_teacher_agreement'])<1e-12
        aggregate.append({'group':g,'views':20,'pixels':pix,'labelled_pixels':known,'teacher_agree_pixels':agreed,
            'coverage':coverage,'conditional_teacher_agreement':agreement,'unknown_fraction':int(cm[:,1].sum())/pix,'no_map_fraction':int(cm[:,0].sum())/pix})
    if a.verify_only:
        original=read(freeze/'audit/aggregate_teacher_recount.json')
        assert original==aggregate
        result={'status':'PASS_CROSS_HOST_RECOUNT','core_files_verified':len(core_manifest),'renders_verified':100,'groups_verified':5,
            'local_original_groups_match':True,'scientific_final_evaluation':False,'gpu_used':False}
        print(json.dumps(result));return
    audit=freeze/'audit';audit.mkdir()
    dump(audit/'state_checks.json',state_checks);dump(audit/'per_view_teacher_recount.json',recounted)
    dump(audit/'aggregate_teacher_recount.json',aggregate);dump(audit/'block_descriptive.json',blocks)
    np.savez_compressed(audit/'teacher_confusion_matrices.npz',**combined)
    write_csv(audit/'aggregate_teacher_recount.csv',aggregate);write_csv(audit/'correlated_block_descriptive.csv',blocks)
    dump(audit/'audit_result.json',{'status':'PASS_ENGINEERING_FREEZE_AND_INDEPENDENT_RECOUNT','core_files_verified':len(core_manifest),
        'renders_verified':100,'per_view_counts_match':True,'aggregate_counts_match':True,'all_geometry_support_masks_equal':True,
        'state_and_base_hashes_verified':True,'exact_180_fusion_identities':True,'gpu_used':False,'recount_seconds':time.time()-started,
        'scientific_final_evaluation':False,'reason':'No independent GT and no isolated final split; teacher DEV already consumed'})
    print(json.dumps(read(audit/'audit_result.json')))

def read_json_string(v):return json.loads(str(v))
if __name__=='__main__':main()
