"""CPU-only exact-union restore and changed-parent rejection on isolated synthetic trees."""
from common import *
import subprocess,tarfile
D=R/'archive_contract';assert not D.exists();D.mkdir()
F=D/'stage3_g21_stereo_20261029_p47149';F.mkdir();O=D/'stage3_g21_stereo_evidence_20261029_p47149';O.mkdir();P=D/'stage3_g21_inputs_20261029_p47149'
original=(R/'archive_stereo.py').read_bytes();(F/'archive_stereo.py').write_bytes(original)
text=(R/'common.py').read_text();anchor='R=Path('+repr(str(R))+')';assert text.count(anchor)==1
(F/'common.py').write_text(text.replace(anchor,'R=Path('+repr(str(F))+')'))
windows=[];image_data={}
for wid in ['K05_F000000_000999','K10_F000000_000999']:
    W=P/'windows'/wid;inputs=[]
    for side in ['image_2','image_3']:
        p=W/'input_view'/side/'000000.png';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('synthetic image fixture '+wid+side).encode());image_data[str(p)]=p.read_bytes()
        inputs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    for rel in ['input_view/calib.txt','input_view/times.txt','output/external_causal_P2_c2w.txt']:
        p=W/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SYNTHETIC NON-GT '+wid)
    dump(W/'audit.json',dict(synthetic_only=True))
    dump(W/'registration.json',dict(inputs=inputs,calib_sha256=sha(W/'input_view/calib.txt'),times_sha256=sha(W/'input_view/times.txt'),synthetic_only=True))
    windows.append(dict(id=wid,input_registration_sha256=sha(W/'registration.json'),frontend_audit_sha256=sha(W/'audit.json'),primary_pose_sha256=sha(W/'output/external_causal_P2_c2w.txt')))
dump(F/'registration_cpu.json',dict(windows=windows,synthetic_only=True))
sources={p.relative_to(F).as_posix():sha(p) for p in F.rglob('*') if p.is_file()}
dump(F/'execution_registration.json',dict(source_files=sources,synthetic_only=True))
def run(label,negative=False):
    name='negative_parent.log' if negative else ('archive_union.log' if label=='union' else label+'.log')
    with (O/name).open('x') as log:
        p=subprocess.run(['python3',str(F/'archive_stereo.py'),'--label',label],cwd=F,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,timeout=60)
    if negative:
        assert p.returncode!=0 and not (O/'archives'/('G21D_'+label+'.tar.gz')).exists()
        assert "sha(p)==item['sha256']" in (O/name).read_text()
    else:assert p.returncode==0,(O/name).read_text()
run('source');negative_rejected=False
for index,win in enumerate(windows):
    wid=win['id'];W=F/'windows'/wid
    dump(W/'receipts/audit.json',dict(status='PASS_1000_PAST_ONLY_CONFIDENCE_INPUT_REPLAY',synthetic_only=True))
    dump(W/'receipts/confidence_access_audit.json',dict(status='PASS_5000_CURRENT_OR_PAST_INPUT_OPENS_WITH_UNFINISHED_ENTRIES_CHECKED',synthetic_only=True))
    for side in ['left','right']:dump(W/'receipts'/f'stereo_{side}_audit.json',dict(status='PASS_1000_STEREO_CURRENT_PAIR_TRACE_AND_FLOAT32_METRIC_DEPTH',synthetic_only=True))
    (W/'fixture.bin').write_bytes(bytes(range(256))*20)
    if index==1:
        image=P/'windows'/wid/'input_view/image_2/000000.png';image.write_bytes(b'modified synthetic parent')
        run(wid,negative=True);negative_rejected=True;image.write_bytes(image_data[str(image)])
    run(wid)
dump(O/'prediction_complete.json',dict(status='COMPLETE_TWO_WINDOWS_SYNTHETIC_FIXTURE_ONLY'))
dump(O/'completed_steps.json',dict(synthetic_only=True))
run('union');restored={};members=0
for label in ['source']+[w['id'] for w in windows]+['union']:
    x=json.loads((O/'archives'/(label+'.json')).read_text());archive=Path(x['path']);assert sha(archive)==x['sha256'];count=0
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            assert m.isfile();data=tar.extractfile(m).read();count+=1
            if m.name in ['MANIFEST.json','UNION_MANIFEST.json']:continue
            rel=Path(m.name).relative_to('G21_STEREO').as_posix()
            if rel in restored:assert restored[rel]==data
            restored[rel]=data
    assert count==x['members'];members+=count
union=json.loads((O/'archives/G21D_UNION_MANIFEST.json').read_text());assert set(restored)=={x['path'] for x in union['files']}
for x in union['files']:assert len(restored[x['path']])==x['bytes'] and hashlib.sha256(restored[x['path']]).hexdigest()==x['sha256']==sha(F/x['path'])
for group in union['parents']:
    assert group['root']=='G21'
    for x in group['files']:assert sha(P/x['path'])==x['sha256']
dump(R/'archive_contract_receipt.json',dict(status='PASS_UNMODIFIED_ARCHIVER_SYNTHETIC_EXACT_UNION_AND_CHANGED_PARENT_REJECTION',at=now(),operator_sha256=hashlib.sha256(original).hexdigest(),data_root_only_substituted_in_common=True,restored_union_files=len(restored),archive_members=members,changed_parent_rejected_before_archive=negative_rejected,synthetic_only=True,actual_dataset_GT_reads=0,GPU_calls=0))
print('PASS_G21D_ARCHIVE_UNION_CONTRACT',flush=True)
