"""Verify real checkpoint replay, create diagnostic visuals and full archive."""
import json,hashlib,tarfile,time,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from semantic_cpu import Evidence,factor_weights
root=Path('/root/autodl-tmp/stage3_chy_20261006');step=root/'02_基础因子DEV与Oracle诊断'
run=step/'dev_sgbm_native_full_20261007_v1';smoke=step/'dev_sgbm_native_smoke_20261007_v1'
result=json.loads((run/'result.json').read_text());assert result['fusion_frames']==180 and result['evaluation_views']==20
config=json.loads((run/'config.json').read_text());fusion=config['fusion_local_frames']
assert len(set(fusion))==180 and not set(fusion)&set(config['eval_local_frames'])
input_locks=json.loads((run/'input_locks.json').read_text())
assert all(hashlib.sha256(Path(r['stage3_reference']).read_bytes()).hexdigest()==r['sha256'] for r in input_locks)
with np.load(root/'inputs/base_map/params.npz') as p:ids=p['source_gaussian_index']
checks=[]
for group in ['S0','S1','S-G','S2','S2-H']:
    final=Evidence.load(run/'states'/f'{group}_final.npz',ids)
    expected_final=next(g for g in result['groups'] if g['group']==group)
    assert hashlib.sha256((run/'states'/f'{group}_final.npz').read_bytes()).hexdigest()==expected_final['state_sha256']
    for count in [30,60,90,120,150,180]:
        state=Evidence.load(run/'states'/f'{group}_checkpoint_{count:03d}.npz',ids)
        assert len(state.processed)==count and state.support.max()<=count
        if count==180:
            assert all(np.array_equal(getattr(state,k),getattr(final,k)) for k in ['evidence','support','total_weight'])
    # Reconstruct last-frame actual cache and replay after restore.
    f=fusion[-1]
    with np.load(run/'observations'/f'{f:06d}.npz') as c:
        active=c['gaussian_index'];assert np.array_equal(ids[active],c['gaussian_id'])
        d=np.zeros_like(final.evidence);d[active]=c['distribution']
        w=np.zeros(len(ids));w[active]=factor_weights(group,np.ones(len(active),bool),c['semantic_reliability'],c['geometry_reliability'],.5)
        version=str(c['input_version'])
    before=hashlib.sha256(final.evidence.tobytes()+final.support.tobytes()+final.total_weight.tobytes()).hexdigest()
    assert not final.update_frame({'dataset':'KITTI','sequence':'08','camera':'image_2','original_frame':f+3000},version,d,w)
    after=hashlib.sha256(final.evidence.tobytes()+final.support.tobytes()+final.total_weight.tobytes()).hexdigest()
    assert before==after
    checks.append({'group':group,'checkpoint_counts_verified':[30,60,90,120,150,180],'restored_real_frame_replay_noop':True,'final_equals_checkpoint180':True})
for f in config['eval_local_frames']:
    masks=[]
    for group in ['S0','S1','S-G','S2','S2-H']:
        with np.load(run/'renders'/group/f'{f:06d}.npz') as p:masks.append(p['alpha']>=.05)
    assert all(np.array_equal(masks[0],m) for m in masks[1:])
(run/'restart_and_shared_support_validation.json').write_text(json.dumps({'status':'PASS','checks':checks,'all_20_views_geometry_support_identical':True,'input_hashes_before_after_equal':True,'input_files_verified':len(input_locks)},indent=2))
palette=json.loads((root/'00_输入与基图_GT曝光审计/reaudit_20261007_v1/palette.json').read_text())
print('PALETTE_FORMAT',type(palette).__name__,flush=True)
if isinstance(palette,dict):colors=np.array([palette[str(i)]['rgb'] for i in range(19)],np.uint8)
else:colors=np.array(palette,np.uint8)
def semantic_image(labels):
    rgb=np.zeros((*labels.shape,3),np.uint8);known=labels>=0
    rgb[known]=colors[labels[known]];rgb[labels==-1]=[255,0,255]
    return Image.fromarray(rgb)
fig=run/'figures';fig.mkdir(exist_ok=True)
source=root/'inputs/kitti08_source_locked_20261007_v1'
for f in [9,99,199]:
    with Image.open(source/'image_2'/f'{f:06d}.png') as im:rgb=im.copy()
    teacher=np.load(source/'labels'/f'{f:06d}_semantic.npy')
    panels=[('RGB',rgb),('2D teacher (NOT GT)',semantic_image(teacher))]
    for group in ['S0','S1','S-G','S2','S2-H']:
        with np.load(run/'renders'/group/f'{f:06d}.npz') as p:panels.append((group,semantic_image(p['label'])))
    width,height=613,185
    canvas=Image.new('RGB',(width*2,(height+25)*4+35),'white');draw=ImageDraw.Draw(canvas)
    for i,(title,im) in enumerate(panels):
        x=(i%2)*width;y=(i//2)*(height+25)
        draw.text((x+5,y+5),f'{title} - original {f+3000}',fill='black');canvas.paste(im.resize((width,height),Image.Resampling.NEAREST),(x,y+25))
    draw.text((5,canvas.height-25),'ENGINEERING DEV | magenta=unknown, black=no-map | historically exposed; no GT accuracy',fill='black')
    canvas.save(fig/f'{f+3000}_comparison.png')
# Fixed threshold curves are teacher diagnostics, no independent GT risk curve.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
curves=json.loads((run/'teacher_disagreement_coverage.json').read_text())
plot,ax=plt.subplots(figsize=(7,4.5))
for group in ['S0','S1','S-G','S2','S2-H']:
    points=[]
    for threshold in np.linspace(0,1,21):
        rows=[r for r in curves if r['group']==group and abs(r['threshold']-threshold)<1e-8]
        accepted=sum(r['accepted'] for r in rows);disagree=sum(r['teacher_disagreements'] for r in rows)
        if accepted:points.append((accepted/(20*370*1226),disagree/accepted))
    points.sort();ax.plot([p[0] for p in points],[p[1] for p in points],marker='.',label=group)
ax.set(xlabel='Coverage / full image pixels',ylabel='Disagreement with 2D prediction teacher',title='Exposed DEV diagnostic: NOT GT risk or accuracy')
ax.legend();ax.grid(alpha=.25);plot.tight_layout();plot.savefig(fig/'teacher_disagreement_coverage.png',dpi=160);plt.close(plot)
manifest={}
for directory in [run,smoke]:
    for p in directory.rglob('*'):
        if p.is_file():manifest[p.relative_to(step).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['full_dev_20261007_v1.log','full_dispatch_20261007.json']:
    p=step/name;manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(step/'native_dev_evidence_SHA256_20261007.json').write_text(json.dumps(manifest,indent=2))
bundle=step/'step02_native_dev_complete_20261007_v1.tar.gz'
with tarfile.open(bundle,'w:gz',compresslevel=1) as t:
    for p in [run,smoke,step/'full_dev_20261007_v1.log',step/'full_dispatch_20261007.json',step/'native_dev_evidence_SHA256_20261007.json']:t.add(p,arcname=p.relative_to(step).as_posix())
print(json.dumps({'status':'PASS_FULL_EVIDENCE_PACKAGED','files':len(manifest),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}),flush=True)
