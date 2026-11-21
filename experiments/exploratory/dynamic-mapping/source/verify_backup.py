from pathlib import Path
import hashlib,json,tarfile,os,shutil,time
root=Path('/root/autodl-tmp/stage2_round17_delivery_v1');root.mkdir(exist_ok=False);scratch=root/'restore_check';scratch.mkdir();spec=json.loads(Path('/root/autodl-tmp/stage2_round17_expected_manifest.json').read_text());(root/'expected_manifest.json').write_text(json.dumps(spec,indent=2));results=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for x in spec:
 p=Path(x['path']);r={'source':str(p),'exists':p.is_file(),'expected_size':x['bytes'],'expected_sha256':x['sha256']};assert p.is_file(),r;r.update(actual_size=p.stat().st_size,actual_sha256=sha(p));r['size_match']=r['actual_size']==r['expected_size'];r['sha_match']=r['actual_sha256']==r['expected_sha256'];assert r['size_match'] and r['sha_match'],r
 dest=root/p.name;os.link(p,dest);r['delivery_path']=str(dest);r['delivery_kind']='hardlink on same filesystem; not an independent backup';r['tar_integrity']='not_applicable'
 if p.name.endswith('.tar.gz'):
  count=0;symlinks=0;bytes_read=0
  with tarfile.open(p,'r:gz') as tar:
   for member in tar:
    count+=1;symlinks+=member.issym() or member.islnk()
    if member.isfile():
     f=tar.extractfile(member)
     while True:
      q=f.read(4*1024*1024)
      if not q:break
      bytes_read+=len(q)
  r.update(tar_integrity='full_regular_members_read_ok',members=count,links=symlinks,uncompressed_regular_bytes=bytes_read)
 results.append(r)
# Restore bridge in a fresh directory; inspect safe normalized paths first.
bridge=root/'scene02_bridge_dereferenced.tar.gz';restore=scratch/'bridge_only';restore.mkdir()
with tarfile.open(bridge,'r:gz') as tar:
 for m in tar:
  if m.issym() or m.islnk() or m.name.startswith('/') or '..' in Path(m.name).parts:raise ValueError('unsafe bridge member '+m.name)
  tar.extract(m,restore,filter='data')
seq=restore/'bridge/replica_layout/vkitti2_Scene02_clone_206_231/results';freeze=json.loads(Path('/root/autodl-tmp/stage2_round16_scene02_v1/freeze.json').read_text());checks=[]
for row in freeze['records']:
 i=row['window_index'];rgb=seq/f'frame{i:06}.jpg';depth=seq/f'depth{i:06}.png';state=restore/'bridge/sidecars'/f'{i:06}_state.npy';ok=sha(rgb)==row['rgb_sha256'] and sha(depth)==row['depth_sha256'] and sha(state)==row['state_sha256'];assert ok;checks.append({'frame':row['frame'],'rgb_depth_state_match':ok})
# Inspect archive links and list target dependency risk; no unrestricted extraction.
links=[]
with tarfile.open(root/'scene02_run.tar.gz','r:gz') as tar:
 for m in tar:
  if m.issym() or m.islnk():links.append({'name':m.name,'target':m.linkname,'target_exists_now':Path(m.linkname).exists() if m.linkname.startswith('/') else None})
report={'items':results,'restore_bridge_frames':len(checks),'restore_checks':checks,'scene02_run_links':links,'limitations':['hardlinks share disk and do not survive server loss','full environment/dependency restore was not executed','bridge extraction and hashes checked; rerun not executed']};(root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({'items':results,'restore_bridge_frames':len(checks),'link_count':len(links),'broken_absolute_links':sum(x['target_exists_now'] is False for x in links)},indent=2))
