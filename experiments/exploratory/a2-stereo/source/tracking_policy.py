"""Fixed-support acceptance; pure NumPy, no GT inputs or adaptive thresholds."""
import numpy as np

DEFAULTS=dict(min_pixels=1000,min_depth_fraction=.1,grid_rows=4,grid_cols=4,
              min_pixels_per_cell=50,min_cells=6,min_improvement=.01,
              max_component_ratio=1.05,max_translation_m=.5,max_rotation_deg=3.,
              min_candidate_coverage=.98)

def support_info(mask,valid_depth,settings=DEFAULTS):
    m=np.asarray(mask,dtype=bool);v=np.asarray(valid_depth,dtype=bool)
    assert m.shape==v.shape and m.ndim==2
    cells=[int(c.sum()) for row in np.array_split(m,settings['grid_rows'],axis=0)
           for c in np.array_split(row,settings['grid_cols'],axis=1)]
    count=int(m.sum());frac=count/max(int(v.sum()),1)
    occupied=sum(x>=settings['min_pixels_per_cell'] for x in cells)
    return dict(pixels=count,fraction=frac,cells=cells,occupied_cells=occupied,
                sufficient=count>=settings['min_pixels'] and frac>=settings['min_depth_fraction'] and occupied>=settings['min_cells'])

def residuals(rgb,depth,target_rgb,target_depth,mask,silhouette,weights):
    m=np.asarray(mask,dtype=bool);n=int(m.sum());assert n>0
    arrays=[np.asarray(x) for x in [rgb,depth,target_rgb,target_depth,silhouette]]
    r,d,tr,td,s=arrays
    finite=bool(np.isfinite(r[:,m]).all() and np.isfinite(d[m]).all() and np.isfinite(s[m]).all())
    if not finite:return dict(finite=False,rgb=None,depth=None,total=None,coverage=None)
    er=float(np.abs(r[:,m]-tr[:,m]).mean());ed=float(np.abs(d[m]-td[m]).mean())
    return dict(finite=True,rgb=er,depth=ed,total=weights['im']*er+weights['depth']*ed,
                coverage=float((s[m]>.99).mean()),pixels=n)

def decide(initial,candidate,translation_m,rotation_deg,settings=DEFAULTS):
    finite=bool(initial['finite'] and candidate['finite'] and np.isfinite([translation_m,rotation_deg]).all())
    if not finite:return dict(accepted=False,reason='nonfinite',map_write_allowed=False)
    gates={'improvement':candidate['total']<=(1-settings['min_improvement'])*initial['total'],
           'rgb_guard':candidate['rgb']<=settings['max_component_ratio']*initial['rgb']+1e-12,
           'depth_guard':candidate['depth']<=settings['max_component_ratio']*initial['depth']+1e-12,
           'translation_guard':translation_m<=settings['max_translation_m'],
           'rotation_guard':rotation_deg<=settings['max_rotation_deg'],
           'coverage_guard':candidate['coverage']>=settings['min_candidate_coverage']}
    accepted=all(gates.values())
    return dict(accepted=accepted,reason='accepted' if accepted else ','.join(k for k,v in gates.items() if not v),
                gates=gates,map_write_allowed=True)
