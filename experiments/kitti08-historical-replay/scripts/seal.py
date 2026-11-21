from common import *
locks(json.loads((R/'registration.json').read_text()));assert json.loads((R/'receipts/score_audit.json').read_text())['status'].startswith('PASS42')
files=[]
for p in sorted(R.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files.append(dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p),dereferenced_symlink=p.is_symlink()))
total=sum(x['bytes'] for x in files);assert total<16*2**30
dump(E/'manifest_full.json',dict(at=now(),files=files,root='G14',payload_bytes=total,scope='All new maps,42float/240view evidence,source,steps,and all fresh input pipeline files; symlink RGB becomes ordinary archive files'))
print('REMOTE_RAW_TREE_SEALED',len(files),total,flush=True)
