"""Only requested KITTI08 members; small first, full only after audit pass."""
import argparse,hashlib,json,time,copy,concurrent.futures,requests,zipfile
from pathlib import Path
from http_range_zip import RangeZip
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/semantickitti08_v1'
URLS={'velodyne':'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip','labels':'https://semantic-kitti.org/assets/data_odometry_labels.zip'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['small','full'],required=True);a=ap.parse_args();frames=list(range(3000,3010 if a.mode=='small' else 3200))
 if a.mode=='full':
  audit=json.loads((R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1/correspondence_audit.json').read_text());assert audit['status']=='PASS_SMALL_CORRESPONDENCE_AUDIT' and audit['visual_review_passed']==True
 D.mkdir(exist_ok=True,parents=True);receipts=[]
 for kind in ['velodyne','labels']:
  folder=D/kind;folder.mkdir(exist_ok=True);rows=[];ext='bin' if kind=='velodyne' else 'label';cache=D/(kind+'_member_index.json');budget=600*1024*1024 if kind=='velodyne' else 64*1024*1024
  if cache.exists():
   cached=json.loads(cache.read_text());head=requests.head(URLS[kind],timeout=45);head.raise_for_status();assert head.headers.get('ETag')==cached['archive']['etag'] and int(head.headers['Content-Length'])==cached['archive']['archive_size'];z=RangeZip.__new__(RangeZip);z.url=URLS[kind];z.session=requests.Session();z.bytes=0;z.budget=budget;z.size=cached['archive']['archive_size'];z.etag=cached['archive']['etag'];z.modified=cached['archive']['last_modified'];z.directory_sha256=cached['archive']['central_directory_sha256'];z.info={}
   for row in cached['members']:
    info=zipfile.ZipInfo(row['name'])
    for k in ['file_size','compress_size','compress_type','header_offset','CRC']:setattr(info,k,row[k])
    z.info[row['name']]=info
  else:
   z=RangeZip(URLS[kind],budget);memberrows=[]
   for f in range(3000,3200):
    name=next(n for n in z.info if n.endswith(f'/08/{kind}/{f:06d}.{ext}'));info=z.info[name];memberrows.append({'name':name,**{k:getattr(info,k) for k in ['file_size','compress_size','compress_type','header_offset','CRC']}})
   cache.write_text(json.dumps({'archive':z.provenance(),'members':memberrows},indent=2)+'\n')
  total_request=sum(z.info[next(n for n in z.info if n.endswith(f'/08/{kind}/{f:06d}.{ext}'))].compress_size+30 for f in frames);assert total_request<600*1024*1024
  def worker(f):
   reader=copy.copy(z);reader.session=requests.Session();reader.bytes=0;reader.budget=32*1024*1024
   names=[n for n in z.info if n.endswith(f'/08/{kind}/{f:06d}.{ext}')];assert len(names)==1;name=names[0];p=folder/f'{f:06d}.{ext}';i=z.info[name]
   if p.exists():
    import zlib
    b=p.read_bytes();assert len(b)==i.file_size and zlib.crc32(b)&0xffffffff==i.CRC
   else:b=reader.read(name);p.write_bytes(b)
   print(json.dumps({'phase':'fetch','kind':kind,'frame':f,'mode':a.mode}),flush=True);reader.session.close();return {'frame':f,'member':name,'path':str(p),'bytes':len(b),'sha256':sha(p),'CRC32':i.CRC,'HTTP_member_bytes':reader.bytes}
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(worker,frames))
  z.bytes+=sum(r['HTTP_member_bytes'] for r in rows)
  receipts.append({'kind':kind,'archive':z.provenance(),'records':rows})
 (D/(a.mode+'_download_manifest.json')).write_text(json.dumps({'status':'MEMBERS_DOWNLOADED_CRC_AND_SHA256_LOCKED','sequence':'08','frames':frames,'archive_receipts':receipts,'no_whole_80GB_archive_download':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
