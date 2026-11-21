"""Credential-free resumable archive downloader; verify SHA256 before cleanup."""
import argparse,concurrent.futures,getpass,hashlib,json,pathlib,time
import paramiko
p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
p.add_argument('--remote',required=True);p.add_argument('--local',required=True);p.add_argument('--sha256',required=True)
a=p.parse_args();target=pathlib.Path(a.local).resolve();assert target.parent.is_dir()
if len(a.sha256)!=64 or any(c not in '0123456789abcdef' for c in a.sha256):
    raise ValueError('expected SHA256 must be exactly 64 lowercase hexadecimal characters')
password=getpass.getpass('SSH password: ')
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(a.host,port=a.port,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);del password
sftp=client.open_sftp();size=sftp.stat(a.remote).st_size;sftp.close()
parts=[target.with_name(target.name+f'.part{i}') for i in range(4)]
offsets=[size*i//4 for i in range(5)]
if target.exists() and target.stat().st_size<size:
    # Keep already transferred sequential bytes, split into verified range slots.
    with target.open('rb') as source:
        for i in range(4):
            take=min(offsets[i+1]-offsets[i],max(0,target.stat().st_size-offsets[i]))
            if take and not parts[i].exists():
                with parts[i].open('wb') as dest:
                    for start in range(0,take,1024*1024):dest.write(source.read(min(1024*1024,take-start)))
            else:source.seek(offsets[i]+take)
def worker(i):
    begin,end=offsets[i],offsets[i+1];have=parts[i].stat().st_size if parts[i].exists() else 0
    assert have<=end-begin
    channel=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*1024*1024,max_packet_size=128*1024)
    try:
        with channel.open(a.remote,'rb') as src,parts[i].open('ab') as dest:
            src.seek(begin+have);src.prefetch(end)
            remaining=end-begin-have
            while remaining:
                block=src.read(min(1024*1024,remaining))
                if not block:raise EOFError('truncated remote archive')
                dest.write(block);remaining-=len(block)
    finally:channel.close()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    tasks=[pool.submit(worker,i) for i in range(4)]
    while any(not t.done() for t in tasks):
        concurrent.futures.wait(tasks,timeout=5)
        total=sum(p.stat().st_size for p in parts if p.exists())
        print(json.dumps({'transferred_bytes':total,'total_bytes':size}),flush=True)
    for t in tasks:t.result()
with target.open('wb') as dest:
    for part in parts:
        with part.open('rb') as src:
            for block in iter(lambda:src.read(8*1024*1024),b''):dest.write(block)
digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==a.sha256
for part in parts:
    assert part.parent==target.parent;part.unlink()
client.close();print(json.dumps({'status':'PASS','bytes':size,'sha256':digest,'path':str(target)}),flush=True)
