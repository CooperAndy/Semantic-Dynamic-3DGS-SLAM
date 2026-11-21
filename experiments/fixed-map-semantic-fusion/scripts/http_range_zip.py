"""Bounded public HTTP ZIP retrieval: verify ranges, archive version and CRC."""
import io,struct,zipfile,zlib,hashlib,time
import requests

class RangeZip:
 def __init__(self,url,budget=512*1024*1024):
  self.url=url;self.session=requests.Session();self.bytes=0;self.budget=budget
  r=self.session.head(url,timeout=60);r.raise_for_status()
  self.size=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag');self.modified=r.headers.get('Last-Modified')
  tail=self.fetch(max(0,self.size-65557),self.size-1)
  at=tail.rfind(b'PK\x05\x06');assert at>=0,'missing ZIP EOCD'
  e=struct.unpack_from('<4s4H2LH',tail,at);count,cdsize,offset=e[4:7]
  if offset==0xffffffff or cdsize==0xffffffff or count==65535:
   loc=tail[at-20:at];assert loc[:4]==b'PK\x06\x07'
   pos=struct.unpack_from('<Q',loc,8)[0];z=self.fetch(pos,pos+55)
   assert z[:4]==b'PK\x06\x06';count,cdsize,offset=struct.unpack_from('<QQQ',z,32)
  assert cdsize<64*1024*1024,'central directory too large'
  cd=self.fetch(offset,offset+cdsize-1);self.directory_sha256=hashlib.sha256(cd).hexdigest()
  # zipfile only needs directory bytes; synthetic EOCD points to offset zero.
  end=struct.pack('<4s4H2LH',b'PK\x05\x06',0,0,min(count,65535),min(count,65535),cdsize,0,0)
  zf=zipfile.ZipFile(io.BytesIO(cd+end));self.info={i.filename:i for i in zf.infolist()};zf.close()
 def fetch(self,start,end):
  n=end-start+1;assert n>0 and self.bytes+n<=self.budget,'HTTP transfer budget exceeded'
  for attempt in range(4):
   try:
    headers={'Range':f'bytes={start}-{end}'}
    if self.etag:headers['If-Match']=self.etag
    r=self.session.get(self.url,headers=headers,stream=True,timeout=(20,90))
    try:
     assert r.status_code==206,f'range unsupported: HTTP {r.status_code}'
     assert r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
     if self.etag:assert r.headers.get('ETag')==self.etag,'remote archive changed'
     b=r.raw.read(n+1);assert len(b)==n,'bad bounded response size'
    finally:r.close()
    self.bytes+=n;return b
   except (requests.RequestException,OSError):
    if attempt==3:raise
    time.sleep(2)
 def read(self,name):
  i=self.info[name];assert i.file_size<32*1024*1024 and i.compress_size<32*1024*1024
  h=self.fetch(i.header_offset,i.header_offset+29);assert h[:4]==b'PK\x03\x04'
  nl,xl=struct.unpack_from('<HH',h,26);start=i.header_offset+30+nl+xl
  raw=self.fetch(start,start+i.compress_size-1)
  if i.compress_type==0:b=raw
  elif i.compress_type==8:b=zlib.decompress(raw,-15)
  else:raise ValueError('unsupported compression')
  assert len(b)==i.file_size and zlib.crc32(b)&0xffffffff==i.CRC
  return b
 def provenance(self):return {'url':self.url,'archive_size':self.size,'etag':self.etag,'last_modified':self.modified,'central_directory_sha256':self.directory_sha256,'HTTP_bytes_read':self.bytes,'hard_transfer_budget':self.budget,'members':len(self.info)}
