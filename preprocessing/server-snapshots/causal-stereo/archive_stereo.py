"""Archive all registered source, each complete window, and the exact final scientific union."""
from common import *
import argparse,tarfile,shutil

def parents(initial,ids):
    files={}
    for win in initial['windows']:
        if win['id'] not in ids:continue
        W=I/'windows'/win['id'];reg=json.loads((W/'registration.json').read_text())
        assert sha(W/'registration.json')==win['input_registration_sha256']
        for item in reg['inputs']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        assert sha(W/'audit.json')==win['frontend_audit_sha256']
        assert sha(W/'output/external_causal_P2_c2w.txt')==win['primary_pose_sha256']
        assert sha(W/'input_view/calib.txt')==reg['calib_sha256'] and sha(W/'input_view/times.txt')==reg['times_sha256']
        paths=[Path(x['path']) for x in reg['inputs']]+[W/'registration.json',W/'audit.json',W/'output/external_causal_P2_c2w.txt',W/'input_view/calib.txt',W/'input_view/times.txt']
        for p in paths:
            relative=p.relative_to(I).as_posix();files[relative]=dict(path=relative,bytes=p.stat().st_size,sha256=sha(p))
    return [dict(root='G21',files=list(files.values()))]

def pack(label,paths,parent_refs,union=None):
    A=E/'archives';A.mkdir(exist_ok=True);target=A/f'G21D_{label}.tar.gz';mp=A/(label+'_MANIFEST.json')
    assert not target.exists() and not (A/(label+'.json')).exists()
    rows=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
    manifest=dict(root='G21_STEREO',label=label,at=now(),files=rows,parents=parent_refs,GT_scoring=False,quality_adoption=False)
    if union is not None:manifest['union_manifest_sha256']=sha(union)
    dump(mp,manifest)
    assert shutil.disk_usage(R).free>sum(x['bytes'] for x in rows)*1.01+8*2**30
    with tarfile.open(target,'w:gz',compresslevel=1,dereference=True) as tar:
        for x in rows:
            tar.add(R/x['path'],arcname='G21_STEREO/'+x['path'],recursive=False)
            assert shutil.disk_usage(R).free>8*2**30
        tar.add(mp,arcname='MANIFEST.json',recursive=False)
        if union is not None:tar.add(union,arcname='UNION_MANIFEST.json',recursive=False)
    remaining={'G21_STEREO/'+x['path']:x for x in rows};count=0
    with tarfile.open(target,'r:gz') as tar:
        for member in tar:
            assert member.isfile();stream=tar.extractfile(member);h=hashlib.sha256();size=0
            for block in iter(lambda:stream.read(2**20),b''):h.update(block);size+=len(block)
            if member.name in ['MANIFEST.json','UNION_MANIFEST.json']:assert h.hexdigest()==sha(mp if member.name=='MANIFEST.json' else union)
            else:
                x=remaining.pop(member.name);assert size==x['bytes'] and h.hexdigest()==x['sha256']
            count+=1
    assert not remaining
    dump(A/(label+'.json'),dict(status='PASS_SERVER_WHOLE_EVERY_MEMBER',label=label,path=str(target),bytes=target.stat().st_size,sha256=sha(target),members=count,parent_files=sum(len(x['files']) for x in parent_refs),quality_adoption=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',required=True);args=parser.parse_args()
    initial=json.loads((R/'registration_cpu.json').read_text());ids=[w['id'] for w in initial['windows']]
    assert args.label in ['source','union']+ids
    reg=json.loads((R/'execution_registration.json').read_text())
    for path,digest in reg['source_files'].items():assert sha(R/path)==digest,path
    if args.label=='source':
        pack('source',[R/p for p in reg['source_files']]+[R/'execution_registration.json'],parents(initial,ids));return
    if args.label in ids:
        W=R/'windows'/args.label
        assert json.loads((W/'receipts/audit.json').read_text())['status']=='PASS_1000_PAST_ONLY_CONFIDENCE_INPUT_REPLAY'
        assert json.loads((W/'receipts/confidence_access_audit.json').read_text())['status']=='PASS_5000_CURRENT_OR_PAST_INPUT_OPENS_WITH_UNFINISHED_ENTRIES_CHECKED'
        for side in ['left','right']:assert json.loads((W/'receipts'/f'stereo_{side}_audit.json').read_text())['status']=='PASS_1000_STEREO_CURRENT_PAIR_TRACE_AND_FLOAT32_METRIC_DEPTH'
        paths=[p for p in W.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
        pack(args.label,paths,parents(initial,[args.label]));return
    assert json.loads((E/'prediction_complete.json').read_text())['status'].startswith('COMPLETE_TWO_WINDOWS')
    base={};refs={}
    for label in ['source']+ids:
        m=json.loads((E/'archives'/(label+'_MANIFEST.json')).read_text())
        for x in m['files']:
            if x['path'] in base:assert base[x['path']]==x
            base[x['path']]=x
        for parent in m['parents']:
            for x in parent['files']:
                key=(parent['root'],x['path'])
                if key in refs:assert refs[key]==x
                refs[key]=x
    snapshot=R/'operators_final_snapshot';snapshot.mkdir()
    for name in ['prediction_complete.json','completed_steps.json']:
        shutil.copy2(E/name,snapshot/('before_union_'+name))
    for p in E.glob('*.log'):
        if p.name!='archive_union.log':shutil.copy2(p,snapshot/p.name)
    current={p.relative_to(R).as_posix():dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    assert not set(base)-set(current)
    for path,x in base.items():assert current[path]==x,('Previously archived source/output changed',path)
    parent_refs=[dict(root='G21',files=[x for (root,_),x in refs.items() if root=='G21'])]
    up=E/'archives/G21D_UNION_MANIFEST.json';dump(up,dict(root='G21_STEREO',files=list(current.values()),parents=parent_refs,base_labels=['source']+ids,GT_scoring=False,quality_adoption=False))
    pack('union',[R/p for p in current if p not in base],parent_refs,union=up)
    print('PASS_COMPLETE_G21D_SERVER_SCIENTIFIC_UNION',len(current),flush=True)

if __name__=='__main__':main()
