"""Network-only prefetch, official cached ZIP offsets; formal fetch rechecks CRC."""
import copy,concurrent.futures,hashlib,json,requests,zipfile,zlib
from pathlib import Path
from http_range_zip import RangeZip
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/semantickitti08_v1'
audit=json.loads((R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1/correspondence_audit.json').read_text());assert audit['visual_review_passed'] and audit['status']=='PASS_SMALL_CORRESPONDENCE_AUDIT'
cached=json.loads((D/'labels_member_index.json').read_text());meta=cached['archive'];url=meta['url'];head=requests.head(url,timeout=45);head.raise_for_status();assert head.headers['ETag']==meta['etag'] and int(head.headers['Content-Length'])==meta['archive_size'];reader=RangeZip.__new__(RangeZip);reader.url=url;reader.size=meta['archive_size'];reader.etag=meta['etag'];reader.info={}
for row in cached['members']:
 i=zipfile.ZipInfo(row['name'])
 for k in ['file_size','compress_size','compress_type','header_offset','CRC']:setattr(i,k,row[k])
 reader.info[i.filename]=i
assert sum(i.compress_size+30 for i in reader.info.values())<64*1024*1024
def work(row):
 name=row['name'];frame=int(Path(name).stem);p=D/'labels'/f'{frame:06d}.label';r=copy.copy(reader);r.session=requests.Session();r.bytes=0;r.budget=8*1024*1024
 if p.exists():b=p.read_bytes()
 else:b=r.read(name);tmp=p.with_suffix('.prefetch.partial');tmp.write_bytes(b);tmp.replace(p)
 assert len(b)==row['file_size'] and zlib.crc32(b)&0xffffffff==row['CRC'];r.session.close();print(json.dumps({'prefetched_label_frame':frame}),flush=True);return {'frame':frame,'sha256':hashlib.sha256(b).hexdigest(),'CRC32':row['CRC'],'HTTP_bytes':r.bytes}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(work,cached['members']))
(D/'labels_prefetch_receipt.json').write_text(json.dumps({'status':'NETWORK_ONLY_PREFETCH_COMPLETE','archive':meta,'records':rows,'semantic_methods_unchanged':True},indent=2)+'\n');print('COMPLETE',flush=True)
