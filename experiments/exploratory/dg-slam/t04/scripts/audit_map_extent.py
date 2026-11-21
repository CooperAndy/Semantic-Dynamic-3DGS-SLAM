"""Inspect frozen map parameters; do not filter or rewrite historical maps."""
from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[1];out={}
for mode in ['off','on']:
    g=np.load(R/'04_后续运行'/('matched_v2_'+mode)/'gaussian_snapshot.npz')
    xyz=g['xyz'];c=g['estimated_keyframe_c2w'][:,:3,3]
    nearest=np.min(np.linalg.norm(xyz[:,None,:]-c[None,:,:],axis=2),axis=1)
    # GaussianModel.get_scaling/get_opacity return RAW parameters in this source.
    # Mapper activates exp(scaling), sigmoid(opacity); labels must reflect that.
    scale=np.exp(g['scaling'].astype(np.float64)).max(axis=1)
    logits=g['opacity'].astype(np.float64).reshape(-1);opacity=np.exp(-np.logaddexp(0,-logits))
    ids=np.where(nearest>80)[0]
    out[mode]={'points':len(xyz),'nearest_keyframe_distance_quantiles_m':dict(zip(['p50','p90','p95','p99','p999','max'],np.quantile(nearest,[.5,.9,.95,.99,.999,1]).tolist())),
        'more_than_80m_from_every_camera':int(np.sum(nearest>80)),
        'more_than_1000m_from_every_camera':int(np.sum(nearest>1000)),
        'max_activated_gaussian_scale_m':float(scale.max()),
        'flagged_points':[{'index':int(i),'xyz_m':xyz[i].tolist(),'nearest_camera_distance_m':float(nearest[i]),'activated_opacity':float(opacity[i]),'max_activated_scale_m':float(scale[i])} for i in ids],
        'parameter_definition':'raw snapshot log scaling and opacity logits; exp/sigmoid applied for diagnostics',
        'limits':'distance is a diagnostic flag, not GT geometric error; no points removed; source frame and causal effect not established'}
dest=R/'02_Gaussian适配审计/差异排查_20261012/map_extent_audit.json';dest.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
