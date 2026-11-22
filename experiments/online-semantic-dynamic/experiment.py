"""Frozen S23 design. Importing this module never runs an experiment."""
from pathlib import Path
import json
import os
import sys

REMOTE = Path(os.environ.get('S23_WORKDIR', 'outputs/s23')).resolve()
PARENT = Path(os.environ.get('S23_LEGACY_PARENT', REMOTE / 'integration')).resolve()
PYTHON = sys.executable
FLAGS = {'B': (False,False,False), 'T': (True,False,False),
         'M': (False,True,False), 'TM': (True,True,False),
         'D': (False,False,True), 'TD': (True,False,True),
         'MD': (False,True,True), 'TMD': (True,True,True)}
LIMITS = dict(E0_PSNR_span_dB=.1,E0_LPIPS_span=.002,E0_coverage_span=.002,
              E0_points_relative_span=.01,translation_m=.0003,rotation_deg=.002,
              startup_free_GiB=23,minimum_free_GiB=8,run_seconds=9000,
              semantic_confidence=.8,semantic_loss_weight=.1,
              ATE_improvement=.1,RPE_improvement=.05,window_max_regression=.05,
              static_PSNR_gain_dB=.5,static_LPIPS_max_delta=0.,SSIM_min_delta=-.01,
              coverage_min_delta=-.02,static_depth_MAE_max_ratio=1.01,
              recent_PSNR_min_delta=-.2,recent_LPIPS_max_delta=.01,
              recent_LPIPS_tail_max_delta=.03)


def design():
    dev=[dict(stage='E2',sequence='08',source_start=3000,frames=1000,seed=0,
              group=g,T=f[0],M=f[1],D=f[2]) for g,f in FLAGS.items()]
    formal=[dict(stage='E3',sequence=s,source_start=0,frames=1000,seed=seed,
                 group=g,T=FLAGS[g][0],M=FLAGS[g][1],D=FLAGS[g][2])
            for s in ['05','10'] for seed in [0,1,2] for g in ['B','TM','D','TMD']]
    return dict(task='S23_EFFECTS',status='REGISTERED_NOT_EXECUTED',limits=LIMITS,
                holdout_rule='i%20==19',evaluation_frames=list(range(19,200,20))+list(range(819,1000,20)),
                tracking='MASS',history_budget=3,mapping_steps=60,birth_cap=3000,
                GT_prediction=False,development=dev,formal=formal,
                gates=['E0 technical and repeatability','E1 mechanisms','E2 complete and freeze','E3'],
                manual_ROI_required=False,new_manual_annotation=False,
                evaluation_revision='r1_no_manual',execution_boundary='DESIGN_ONLY',
                static_evaluation='fixed official sparse LiDAR support; no dense ROI',
                quality_adoption=False)


if __name__=='__main__':
    p=Path(__file__).with_name('design.json')
    with p.open('x',encoding='utf-8') as stream:json.dump(design(),stream,ensure_ascii=False,indent=2)
