from common import *
import subprocess,shutil,shlex
assert not (R/'registration.json').exists()
old=json.loads((P/'input_registration.json').read_text());root=Path(old['vocabulary']).parent.parent;failure=R/'provenance/build_v1';failure.mkdir()
for name in ['prepare.log','compile.log','controller.log','controller.json']:
 p=E/name
 if p.exists():shutil.copy2(p,failure/name)
shutil.copy2(R/'build/command.json',failure/'command.json')
args=json.loads((R/'build/command.json').read_text())['argv'];args[1:1]=['-D_LINUX_','-DHAVE_EIGEN','-DHAVE_EPOXY'];args[-2:-2]=['-lepoxy','-lOpenGL','-lEGL'];dump(R/'build/command_v2.json',dict(argv=args,reason='Pangolin imported-target compile definitions and link interface'))
with (E/'compile_v2.log').open('x') as f:subprocess.run(args,env=env(),stdout=f,stderr=subprocess.STDOUT,timeout=600,check=True)
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
