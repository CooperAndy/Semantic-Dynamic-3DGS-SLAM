from common import *
import sys,tarfile,gzip,io
m=json.loads((E/'manifest_full.json').read_text());assert json.loads((E/'state.json').read_text())['state']=='READY_FOR_LOCAL_STREAM_ARCHIVE'
h=hashlib.sha256();counter=[0]
class Sink:
 def write(self,b):h.update(b);counter[0]+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
with gzip.GzipFile(fileobj=Sink(),mode='wb',compresslevel=1,mtime=0) as z:
 with tarfile.open(fileobj=z,mode='w|',dereference=True) as t:
  for x in m['files']:
   p=R/x['path'];assert p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];t.add(p,arcname='G14/'+x['path'],recursive=False)
  b=json.dumps(m,indent=2).encode();a=tarfile.TarInfo('G14/ARCHIVE_MANIFEST_FULL.json');a.size=len(b);t.addfile(a,io.BytesIO(b))
sys.stdout.buffer.flush();dump(E/'receipt_full.json',dict(status='REMOTE_COMPRESSED_STREAM_SHA_AND_RAW_MEMBERS_PASS',at=now(),bytes=counter[0],sha256=h.hexdigest(),entries=len(m['files'])+1,remote_compressed_copy=False,remote_raw_tree_preserved=True,local_archive_required=True))
