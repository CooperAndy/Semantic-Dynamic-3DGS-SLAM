"""Prepare isolated Stage B source delta and audited worker, no launch."""
from pathlib import Path
import ast,hashlib,json

R=Path(__file__).resolve().parents[1];P=R/'scripts'
OUT=R/'02_Gaussian适配审计/GS_SLAM借鉴_20261012/stage_b_patch';OUT.mkdir(exist_ok=True)
parent=R/'02_Gaussian适配审计/GS_SLAM借鉴_20261012/stage_a_patch/dg_slam/gs_tracking_mapping.py'
s=parent.read_text(encoding='utf-8')
s='from depth_consistent_selector import InsufficientTrackingSupport\n'+s
needle='        geo_loss = torch.clamp((torch.abs(gt_depth-depth)), min=0.0, max=1e3)[mask].sum()'
assert s.count(needle)==1
s=s.replace(needle,"""        if self.cfg.get('t04_depth_selection', False):
            support = int(mask.sum().item())
            if support < self.cfg['t04_min_tracking_pixels']:
                raise InsufficientTrackingSupport('too_few_loss_pixels:' + str(support))
"""+needle)
ast.parse(s);(OUT/'gs_tracking_mapping.py').write_text(s,encoding='utf-8')
w=(P/'run_stage_a.py').read_text(encoding='utf-8')
w=w.replace("source=ROOT/'source_v3_depthguard'", "source=ROOT/'source_v4_depthselection'")
w=w.replace("intervention='Stage A depth guard only'", "intervention='Stage A plus B observed-depth Gaussian selection'")
w=w.replace("        pixel_hash_to_id=", """        from depth_consistent_selector import install_selector
        install_selector(mapper,importlib.import_module(type(mapper).__module__),{x[0]:x[2].clone() for x in data},output)
        meta['selection_protocol']={'observed_depth':'immutable same-frame input SGBM pseudo-depth; never rendered or DROID-optimized depth',
            'tolerance':'0.5m + 0.02 * observed_depth','min_gaussians':256,'min_pixels':128,
            'frozen_per_refinement':True,'fallback':'retain DROID coarse pose',
            'depth_range_m':[0.5,40.0],'paper_scope':'Eq12-inspired selection only; no new coarse-resolution pass or BA',
            'preparation_sha256':sha256(ROOT/'stage_b/preparation.json'),
            'selector_sha256':sha256(Path(__file__).with_name('depth_consistent_selector.py'))}
        pixel_hash_to_id=""")
w=w.replace("            coarse.append(args[2]", "            mapper._t04_source_id=actual_id\n            coarse.append(args[2]")
ast.parse(w);(P/'run_stage_b.py').write_text(w,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'status':'PREPARED_NOT_RUN','parent_mapper_sha256':sha(parent),'mapper_sha256':sha(OUT/'gs_tracking_mapping.py'),
 'worker_sha256':sha(P/'run_stage_b.py'),'selector_sha256':sha(P/'depth_consistent_selector.py'),
 'fixed_parameters':{'abs_tolerance_m':0.5,'relative_tolerance':0.02,'min_gaussians':256,'min_loss_pixels':128,'tracking_max_depth_m':40},
 'basis':'engineering pilot values fixed before GPU/GT evaluation, not claimed paper hyperparameters or tuned optimum',
 'change':'tracking-only frozen Gaussian selection before render plus insufficient support fallback to coarse pose',
 'unchanged':['A map admission','DROID','50 tracking iterations','learning rates','BA false','mapping depth 40m','same seed and byte-identical inputs'],
 'comparison':'completed Stage A on same 5090 compatibility environment; independent online runs, not bitwise paired maps',
 'limits':['SGBM pseudo-depth is noisy, not GT','center selection does not fully model splat footprint occlusions','fallback alters effective iteration count and must be reported']}
(OUT/'preparation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared Stage B mapper, selector and worker')
