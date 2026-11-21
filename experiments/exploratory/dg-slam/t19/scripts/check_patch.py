"""Reconstruct candidate from unified patch and archived baseline without mutating either."""
from pathlib import Path
import hashlib,json,re
F=Path(__file__).resolve().parents[1]
raw=(F/'patches/T19_vs_T17.patch').read_text(encoding='utf-8').splitlines(keepends=True)
lines=[]
for line in raw:
 if line.startswith('\\ No newline at end of file'):
  assert lines and lines[-1].endswith('\n')
  lines[-1]=lines[-1][:-1]
 else:lines.append(line)
cursor=0;records=[]
while cursor<len(lines):
 assert lines[cursor].startswith('--- ')
 oldname=lines[cursor][4:].strip();newname=lines[cursor+1][4:].strip();cursor+=2
 assert newname.startswith('b/')
 rel=newname[2:];old=F/'base_snapshot/source'/rel
 before=old.read_text(encoding='utf-8').splitlines(keepends=True) if oldname!='/dev/null' else []
 position=0;result=[]
 while cursor<len(lines) and not lines[cursor].startswith('--- '):
  match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[cursor]);assert match,lines[cursor]
  start=int(match[1]);count=int(match[2] or 1);new_count=int(match[4] or 1)
  start=start-1 if start else 0
  assert start>=position
  result.extend(before[position:start]);position=start;cursor+=1;removed=added=0
  while cursor<len(lines) and not lines[cursor].startswith(('@@ ','--- ')):
   line=lines[cursor];cursor+=1
   if line.startswith(' '):
    assert before[position]==line[1:];result.append(line[1:]);position+=1;removed+=1;added+=1
   elif line.startswith('-'):
    assert before[position]==line[1:];position+=1;removed+=1
   elif line.startswith('+'):result.append(line[1:]);added+=1
   else:raise AssertionError('Unexpected unified patch line')
  assert removed==count and added==new_count
 result.extend(before[position:]);text=''.join(result)
 assert text==(F/'candidate'/rel).read_text(encoding='utf-8'),rel
 records.append({'path':rel,'reconstructed_candidate_matches':True,'normalized_UTF8_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest()})
assert len(records)==6
receipt={'status':'PASS','files':records,'baseline_or_candidate_modified':False}
(F/'receipts/patch_reconstruction.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','patch_reconstructed_files':len(records)}))
