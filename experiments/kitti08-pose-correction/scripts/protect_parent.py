"""Reject all writes to the closed G20 parent science and operator directories."""
from pathlib import Path
import sys,os
ROOTS=[Path('/root/autodl-tmp/stage3_g20_20261029_p47149'),Path('/root/autodl-tmp/stage3_g20_evidence_20261029_p47149')]
for version in ['', '_v2', '_v3', '_v4']:
 for name in ['stage3_g29_stereo_prefix','stage3_g29_evidence']:
  ROOTS.append(Path('/root/autodl-tmp')/(name+version+'_20261029_p47149'))
ROOTS.extend([Path('/root/autodl-tmp/stage3_g30_stereo_full_20261029_p47149'),Path('/root/autodl-tmp/stage3_g30_evidence_20261029_p47149')])

ROOTS.extend([Path('/root/autodl-tmp/stage3_g30_recovery_run_20261101_p44181'),Path('/root/autodl-tmp/stage3_g30_recovery_20261101_p44181')])

def forbidden(p):
 if not isinstance(p,(str,bytes,os.PathLike)):return False
 p=Path(os.fsdecode(p)).resolve()
 return any(p==r or p.is_relative_to(r) for r in ROOTS)
def install():
 def hook(event,args):
  if event=='open':
   mode=args[1] or '';flags=args[2] or 0;write=any(x in mode for x in ['w','a','x','+']) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
   if write and forbidden(args[0]):raise RuntimeError('CLOSED_G20_PARENT_WRITE_BLOCKED')
  elif event in ['os.rename','os.remove','os.rmdir','os.mkdir','os.link','os.symlink']:
   if any(forbidden(a) for a in args[:2]):raise RuntimeError('CLOSED_G20_PARENT_MUTATION_BLOCKED')
 sys.addaudithook(hook)
