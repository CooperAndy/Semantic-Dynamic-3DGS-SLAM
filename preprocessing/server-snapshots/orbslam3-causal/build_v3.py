from common import *
import subprocess,shutil,shlex
assert not (R/'registration.json').exists()
old=json.loads((P/'input_registration.json').read_text());root=Path(old['vocabulary']).parent.parent;failure=R/'provenance/build_v2';failure.mkdir()
for name in ['prepare_v2.log','compile_v2.log','controller_v2.log','controller_v2.json']:
 p=E/name
 if p.exists():shutil.copy2(p,failure/name)
shutil.copy2(R/'build/command_v2.json',failure/'command.json')
args=json.loads((R/'build/command_v2.json').read_text())['argv'];args[-2:-2]=[str(p) for p in sorted((R/'runtime/lib').glob('libpango*.so*'))];dump(R/'build/command_v3.json',dict(argv=args,reason='ExplicitPangolin DSOs for inline API header dependencies'))
with (E/'compile_v3.log').open('x') as f:subprocess.run(args,env=env(),stdout=f,stderr=subprocess.STDOUT,timeout=600,check=True)
inputs=[]
for row in old['rows']:
 for cam in ['image_2','image_3']:
  p=P/'input_view'/cam/f'{row["frame_id"]:06}.png';assert sha(p)==row[cam]['sha256'];inputs.append(dict(path=str(p),source=row[cam]['source'],sha256=sha(p),bytes=p.stat().st_size))
headers=[]
for name in shlex.split((R/'build/dependencies.d').read_text().replace('\\\n',' ').split(':',1)[1]):
 p=Path(name)
 if p.is_file() and p.is_relative_to(root):
  target=R/'provenance/headers'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);headers.append(dict(path=str(p),sha256=sha(p)))
libraries=[]
for p in (R/'runtime/lib').iterdir():libraries.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p)))
reg=dict(at=now(),status='REGISTERED_CAUSAL_INPUT_NOT_EXECUTED',window=old['window'],inputs=inputs,libraries=libraries,source_headers=headers,settings_sha256=sha(R/'inputs/settings.yaml'),vocabulary_sha256=sha(R/'runtime/ORBvoc.txt'),binary_sha256=sha(R/'bin/causal_stereo'),source_files={p.relative_to(R).as_posix():sha(p) for p in R.iterdir() if p.is_file()},build_v1_preserved=True,GT_allowed=False,quality_adoption=False);dump(R/'registration.json',reg);print('PASS_PRIVATE_BUILD_AND2000_INPUT_LOCKS',flush=True)
