"""Four preregistered tables and paired-window checks; never choose a best camera."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from experiment import REMOTE,design


def validate_completed_cell(cell,evaluation,receipt,semantic,numeric):
    assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION'
    assert receipt['frames']==1000 and receipt['mapping_backward']==57000 and receipt['GT_blocked_opens']==0
    assert evaluation['status']=='COMPLETE60_FIXED_VIEWS_AND_FULL_TRAJECTORY'
    assert evaluation['group']==cell['group'] and evaluation['geometry_unchanged'] is True and evaluation['GT_prediction'] is False
    frames=list(range(19,200,20))+list(range(819,1000,20))
    expected={(mode,i) for mode in ['COMMON_GT','COMMON_ORB','OWN_SELECTED'] for i in frames}
    assert len(evaluation['rows'])==60 and {(r['mode'],r['frame']) for r in evaluation['rows']}==expected
    assert all(r['source_frame']==r['frame']+cell['source_start'] for r in evaluation['rows'])
    assert semantic['status']=='ONLINE_STATE_FINISHED'
    assert all(semantic[key]==cell[key] for key in ['T','M','D'])
    assert semantic['source_frames']==[i for i in range(1000) if i%20!=19]
    assert numeric['status']=='PASS_INDEPENDENT_BIRTH_POSE_AND_SEMANTIC_IDENTITIES'
    assert numeric['frames']==1000 and numeric['GT_used'] is False


def paired_table_statistics(tables):
    rows=[]
    for candidate,control in [('TM','B'),('TMD','D')]:
        for sequence in ['05','10']:
            for table,values in tables.items():
                index={(r['seed'],r['group']):r for r in values if r['stage']=='E3' and r['sequence']==sequence}
                pairs=[(index[(seed,candidate)],index[(seed,control)]) for seed in [0,1,2]]
                metrics=set(pairs[0][0])-{'seed','T','M','D'}
                for metric in sorted(metrics):
                    observed=[]
                    for seed,(a,b) in enumerate(pairs):
                        x,y=a.get(metric),b.get(metric)
                        if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool) and math.isfinite(x) and math.isfinite(y):
                            observed.append(dict(seed=seed,candidate=x,control=y,difference=x-y))
                    if not observed:continue
                    differences=[p['difference'] for p in observed]
                    rows.append(dict(table=table,sequence=sequence,candidate=candidate,control=control,metric=metric,
                        backend_pairs=len(observed),candidate_mean=float(np.mean([p['candidate'] for p in observed])),
                        control_mean=float(np.mean([p['control'] for p in observed])),difference_mean=float(np.mean(differences)),
                        difference_standard_deviation=float(np.std(differences,ddof=1)) if len(differences)>1 else None,
                        pairs=observed))
    return rows

def main():
    out=REMOTE/'final_report';out.mkdir(exist_ok=False)
    trajectory=[];mapping=[];semantic=[];cost=[];lookup={}
    allcells=design()['development']+design()['formal']
    index_path=REMOTE/'execution_root_index.json';root_index=json.loads(index_path.read_text()) if index_path.exists() else {}
    for cell in allcells:
        entry=root_index.get(f"{cell['stage']}:K{cell['sequence']}:S{cell['seed']}")
        root=REMOTE/(entry['active_root'] if entry else f"{cell['stage']}_K{cell['sequence']}_S{cell['seed']}");d=root/'runs'/cell['group']
        evaluation=json.loads((d/'evaluation/results.json').read_text());r=json.loads((d/'receipt.json').read_text());s=json.loads((d/'semantic_receipt.json').read_text())
        numeric=json.loads((d/'independent_numeric_audit.json').read_text())
        validate_completed_cell(cell,evaluation,r,s,numeric)
        key={k:cell[k] for k in ['stage','sequence','seed','group','T','M','D']}
        primary=next(x['means'] for x in evaluation['summary'] if x['mode']=='COMMON_GT' and x['cohort']=='ALL')
        diagnostics=evaluation['tracking_diagnostics']
        trajectory.append(dict(**key,**evaluation['trajectory'],**{k:diagnostics[k] for k in ['accepted_steps','geometry_qualified_steps','geometry_failure_steps','prediction_fallback_steps','semantic_removed_correspondences']}))
        dynamic=diagnostics['dynamic_consumption']
        mapping.append(dict(**key,**{k:primary[k] for k in ['PSNR_dB','SSIM','LPIPS','coverage','static_RGB_PSNR_dB','static_depth_MAE_m','static_depth_coverage']},dynamic_mapping_status=diagnostics['dynamic_mapping_status'] if cell['D'] else 'DISABLED',**{k:dynamic.get(k,0) for k in ['D_birth_candidates_removed','D_depth_loss_pixel_visits_removed','D_RGB_loss_pixel_visits_removed']}))
        semantic.append(dict(**key,sparse_mIoU=primary['sparse_mIoU'],semantic_label_coverage=primary['semantic_label_coverage'],assigned_gaussians=s['known_count'],gaussians=s['gaussian_count'],**{k:primary[k] for k in ['static_vehicle_depth_coverage','static_vehicle_label_coverage','static_vehicle_semantic_accuracy']}))
        producer=REMOTE/'bundles'/cell['sequence']/'producer_timings.jsonl'
        inference=[json.loads(line) for line in producer.read_text().splitlines()] if producer.exists() else []
        timing=producer.parent/'semantic_inference_timing.json'
        semantic_inference=json.loads(timing.read_text())['timing'] if timing.exists() else {}
        resources=evaluation['resource_audit']
        cost_resource_note=dict(user_priority_pause_seconds=resources.get('user_priority_pause_seconds',0.))
        cost.append(dict(**key,algorithm_seconds=r['algorithm_seconds'],peak_torch_bytes=r['peak_torch_bytes'],final_gaussians=r['final_gaussians'],semantic_gradient_calls=s['semantic_gradient_calls'],**s['timing_seconds'],**r['render_timing'],semantic_inference_total_seconds=semantic_inference.get('inference_total_seconds'),semantic_generation_wall_seconds=semantic_inference.get('generation_wall_seconds'),semantic_mean_inference_seconds_per_frame=semantic_inference.get('mean_inference_seconds_per_frame'),producer_process_timings=inference,shared_GPU_status=resources['status'],shared_GPU_samples=resources['samples'],shared_GPU_overlap_samples=resources['overlap_samples'],runtime_comparison_eligible=resources['runtime_comparison_eligible'],timing_scope='Raw observed elapsed; inspect resource audit before speed comparisons. Nested subtotals; shared input generation not summed across groups; not end-to-end FPS'))
        cost[-1].update(cost_resource_note)
        lookup[(cell['stage'],cell['sequence'],cell['seed'],cell['group'])]=evaluation
    for name,rows in [('定位',trajectory),('地图质量',mapping),('语义质量',semantic),('开销',cost)]:
        # Nested arrays are retained in complete JSON; TSV stays readable.
        keys=[k for k,v in rows[0].items() if not isinstance(v,(list,dict))]
        with (out/(name+'.tsv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=keys,delimiter='\t',extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    contrasts=[]
    dev_contrasts=[]
    for candidate,control in [('T','B'),('M','B'),('TM','B'),('D','B'),('TD','D'),('MD','D'),('TMD','D'),('TMD','TM')]:
        a=lookup[('E2','08',0,candidate)];b=lookup[('E2','08',0,control)]
        primary=lambda e:next(x['means'] for x in e['summary'] if x['mode']=='COMMON_GT' and x['cohort']=='ALL')
        dev_contrasts.append(dict(candidate=candidate,control=control,delta_ATE_m=a['trajectory']['ATE_RMSE_m']-b['trajectory']['ATE_RMSE_m'],
            delta_RPE_m=a['trajectory']['RPE_translation_RMSE_m_1frame']-b['trajectory']['RPE_translation_RMSE_m_1frame'],
            **{'delta_'+k:primary(a)[k]-primary(b)[k] for k in ['PSNR_dB','LPIPS','static_RGB_PSNR_dB','static_depth_MAE_m','sparse_mIoU']}))
    with (out/'8组消融.tsv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(dev_contrasts[0]),delimiter='\t');writer.writeheader();writer.writerows(dev_contrasts)
    for candidate,control in [('TM','B'),('TMD','D')]:
        windows=[]
        for seq in ['05','10']:
            pairs=[(lookup[('E3',seq,seed,candidate)],lookup[('E3',seq,seed,control)]) for seed in [0,1,2]]
            gains={}
            for metric in ['ATE_RMSE_m','RPE_translation_RMSE_m_1frame']:
                a=np.mean([p[0]['trajectory'][metric] for p in pairs]);b=np.mean([p[1]['trajectory'][metric] for p in pairs]);gains[metric]=float(1-a/b)
            delta={}
            for metric in ['static_RGB_PSNR_dB','SSIM','LPIPS','coverage','static_depth_MAE_m']:
                means=lambda e:next(x['means'] for x in e['summary'] if x['mode']=='COMMON_GT' and x['cohort']=='ALL')
                delta[metric]=float(np.mean([means(a)[metric]-means(b)[metric] for a,b in pairs]))
            paired={}
            for metric in ['ATE_RMSE_m','RPE_translation_RMSE_m_1frame']:
                differences=[a['trajectory'][metric]-b['trajectory'][metric] for a,b in pairs]
                paired[metric]=dict(differences=differences,mean=float(np.mean(differences)),standard_deviation=float(np.std(differences,ddof=1)))
            paired_mapping={}
            for metric in ['PSNR_dB','SSIM','LPIPS','coverage','static_RGB_PSNR_dB','static_depth_MAE_m','sparse_mIoU']:
                differences=[means(a)[metric]-means(b)[metric] for a,b in pairs]
                paired_mapping[metric]=dict(differences=differences,mean=float(np.mean(differences)),standard_deviation=float(np.std(differences,ddof=1)))
            failures_not_increased=sum(a['tracking_diagnostics']['geometry_failure_steps'] for a,b in pairs)<=sum(b['tracking_diagnostics']['geometry_failure_steps'] for a,b in pairs)
            windows.append(dict(sequence=seq,gains=gains,delta=delta,paired_trajectory_differences=paired,paired_mapping_differences=paired_mapping,proposal_failures_not_increased=failures_not_increased))
        ate=float(np.mean([w['gains']['ATE_RMSE_m'] for w in windows]));rpe=float(np.mean([w['gains']['RPE_translation_RMSE_m_1frame'] for w in windows]))
        track_gate=ate>=.1 and rpe>=.05 and all(min(w['gains'].values())>=-.05 and w['proposal_failures_not_increased'] for w in windows)
        map_gates=[]
        for seq in ['05','10']:
            pairs=[(lookup[('E3',seq,seed,candidate)],lookup[('E3',seq,seed,control)]) for seed in [0,1,2]]
            mean=lambda e,cohort:next(x['means'] for x in e['summary'] if x['mode']=='COMMON_GT' and x['cohort']==cohort)
            delta=lambda metric,cohort='ALL':float(np.mean([mean(a,cohort)[metric]-mean(b,cohort)[metric] for a,b in pairs]))
            depth_ratio=float(np.mean([mean(a,'ALL')['static_depth_MAE_m']/mean(b,'ALL')['static_depth_MAE_m'] for a,b in pairs]))
            tail=[]
            for a,b in pairs:
                aa={r['frame']:r for r in a['rows'] if r['mode']=='COMMON_GT' and r['cohort']=='RECENT'}
                bb={r['frame']:r for r in b['rows'] if r['mode']=='COMMON_GT' and r['cohort']=='RECENT'}
                assert set(aa)==set(bb)
                tail.extend(aa[i]['LPIPS']-bb[i]['LPIPS'] for i in aa)
            measured=dict(static_PSNR_gain=delta('static_RGB_PSNR_dB'),LPIPS_delta=delta('LPIPS'),SSIM_delta=delta('SSIM'),coverage_delta=delta('coverage'),
                          depth_ratio=depth_ratio,recent_PSNR_delta=delta('PSNR_dB','RECENT'),recent_LPIPS_delta=delta('LPIPS','RECENT'),recent_LPIPS_tail=max(tail))
            gates=dict(static_PSNR=measured['static_PSNR_gain']>=.5,LPIPS=measured['LPIPS_delta']<=0,SSIM=measured['SSIM_delta']>=-.01,
                       coverage=measured['coverage_delta']>=-.02,depth=depth_ratio<=1.01,recent_PSNR=measured['recent_PSNR_delta']>=-.2,
                       recent_LPIPS=measured['recent_LPIPS_delta']<=.01,recent_tail=measured['recent_LPIPS_tail']<=.03)
            map_gates.append(dict(sequence=seq,measured=measured,gates=gates,passed=all(gates.values())))
        contrasts.append(dict(candidate=candidate,control=control,windows=windows,window_equal_ATE_gain=ate,window_equal_RPE_gain=rpe,
                              tracking_target_pass=track_gate,mapping_by_window=map_gates,joint_target_pass=track_gate and all(x['passed'] for x in map_gates)))
    paired_tables=paired_table_statistics(dict(trajectory=trajectory,mapping=mapping,semantic=semantic,cost=cost))
    with (out/'逐窗种子配对差值.tsv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=[k for k in paired_tables[0] if k!='pairs'];writer=csv.DictWriter(f,fieldnames=keys,delimiter='\t',extrasaction='ignore');writer.writeheader();writer.writerows(paired_tables)
    result=dict(status='COMPLETE32_REGISTERED_RESULTS',trajectory=trajectory,mapping=mapping,semantic=semantic,cost=cost,formal_contrasts=contrasts,formal_paired_tables=paired_tables,
                development_contrasts=dev_contrasts,GT_source='Official sparse SemanticKITTI plus KITTI odometry trajectories',new_manual_labels=False,
                quality_adoption=False,registered_joint_targets_met=any(x['joint_target_pass'] for x in contrasts),
                adoption_note='All preregistered numerical targets are evaluated above; adopting a project method is recorded separately from finishing experiments.')
    from report_views import collect
    from report_document import write
    views=collect(REMOTE,out,allcells,root_index)
    result['publication_views_count']=len(views)
    from semantic_confusion import collect_registered
    result['semantic_class_support']=collect_registered(REMOTE,out,allcells,root_index)
    write(out,result,views)
    (out/'全部结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    required=['定位.tsv','地图质量.tsv','语义质量.tsv','开销.tsv','8组消融.tsv','逐窗种子配对差值.tsv',
        '全部结果.json','结果说明.md','典型视图索引.md','典型视图/图像清单.json','各类语义支持.tsv','各类语义支持与混淆.json']
    assert all((out/name).is_file() for name in required)
    files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file()]
    manifest=dict(status='COMPLETE_S23_TABLES_CONCLUSIONS_AND_FIXED_VIEWS',native_groups=32,
        development_groups=8,formal_groups=24,fixed_common_GT_images=64,new_manual_annotation=False,
        required=required,files=files,quality_adoption=False)
    (out/'交付清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
