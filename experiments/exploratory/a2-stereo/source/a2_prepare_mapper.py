"""Copy mapper and insert A2-only geometric L1 replacement and gradient guards."""
import hashlib,json,shutil
from pathlib import Path
R=Path('/root/autodl-tmp/A2_stereo_robust_geometry/mapper_v1');BASE=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200/source')
R.mkdir(exist_ok=False);shutil.copytree(BASE,R/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=R/'source/scripts/splatam_external_tm.py';s=p.read_text(encoding='utf-8')
before="""        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss"""
after="""        else:
            if curr_data.get('a2_mode') is None:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
            else:
                from a2_losses import robust_loss
                _a2_obs = curr_data['depth'][mask].detach()
                _a2_conf = curr_data['a2_confidence'][mask].detach()
                _a2_residual = depth[mask] - _a2_obs
                losses['depth'] = robust_loss(_a2_residual, _a2_obs, _a2_conf,
                    curr_data['a2_fb'], curr_data['a2_mode'], curr_data['a2_gamma'])
    
    # RGB Loss"""
guard_before="""                # Backprop
                loss.backward()
                with torch.no_grad():"""
guard_after="""                # Backprop
                loss.backward()
                if params['means3D'].grad is None or not torch.isfinite(params['means3D'].grad).all():
                    raise RuntimeError('A2 missing or nonfinite geometry gradient')
                with torch.no_grad():"""
if s.count(before)!=1 or s.count(guard_before)!=1:raise RuntimeError('unexpected source; abort')
p.write_text(s.replace(before,after).replace(guard_before,guard_after),encoding='utf-8')
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
result={'base_core_sha256':sha(BASE/'scripts/splatam_external_tm.py'),'isolated_core_sha256':sha(p),'patch':'mapping depth branch and finite geometry gradient guard','quality_adoption':False}
(R/'preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
