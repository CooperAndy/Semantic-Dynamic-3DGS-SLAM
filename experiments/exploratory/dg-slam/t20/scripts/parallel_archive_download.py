"""Eight read-only SSH byte ranges, with in-memory authentication and final SHA256."""
import concurrent.futures
import getpass
import hashlib
import json
import shlex
import time
from pathlib import Path
import paramiko

F=Path(__file__).resolve().parents[1]
receipt=json.loads((F/'receipts/remote_bundle_receipt.json').read_text(encoding='utf-8'))
REMOTE='/root/autodl-tmp/dg_slam_chy_20261011/11_T20_filler_GPU_20261012/T20_GPU_raw_20261012.zip'
PARTS=F.parent/'.T20_raw_transfer_parts_20261012'
TARGET=F.parent/'T20_CHY_GPU原始诊断_20261012_parallel.zip'
PARTS.mkdir(exist_ok=False)
assert not TARGET.exists()
password=getpass.getpass('SSH password (memory only): ')
start=time.time()
def download(index):
    total=receipt['bytes'];step=(total+7)//8;offset=index*step;length=min(step,total-offset)
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('SERVER_HOST',port=36491,username='root',password=password,timeout=20,auth_timeout=20,banner_timeout=20,allow_agent=False,look_for_keys=False)
    transport=client.get_transport();transport.default_window_size=64*1024**2;transport.default_max_packet_size=1024**2
    assert hashlib.sha256(transport.get_remote_server_key().asbytes()).hexdigest()=='962677eaf342b0d70d757796b387fe83964884f33f66284ff37e2fc6cf1496a7'
    command=f'dd if={shlex.quote(REMOTE)} bs=1048576 skip={offset} count={length} iflag=skip_bytes,count_bytes status=none'
    stdin,stdout,stderr=client.exec_command(command,timeout=1800)
    path=PARTS/f'part{index:02d}.bin';size=0;h=hashlib.sha256()
    with path.open('xb') as dest:
        while True:
            block=stdout.read(1024**2)
            if not block:break
            dest.write(block);size+=len(block);h.update(block)
    assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
    client.close();assert size==length
    print(json.dumps({'part':index,'bytes':size,'completed_seconds':time.time()-start}),flush=True)
    return {'index':index,'offset':offset,'bytes':size,'sha256':h.hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results=list(pool.map(download,range(8)))
del password
h=hashlib.sha256();size=0
with TARGET.open('xb') as dest:
    for index in range(8):
        with (PARTS/f'part{index:02d}.bin').open('rb') as source:
            for block in iter(lambda:source.read(2**20),b''):
                dest.write(block);h.update(block);size+=len(block)
assert size==receipt['bytes'] and h.hexdigest()==receipt['sha256']
result={'status':'PASS','archive':TARGET.name,'bytes':size,'sha256':h.hexdigest(),'seconds':time.time()-start,'parts':results,'host_key_pinned':True,'credentials_stored':False}
(F/'receipts/parallel_transfer.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='parts'}),flush=True)
