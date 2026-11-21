"""Credential-free SSH orchestration; password read interactively, never persisted."""
import getpass
import json
import pathlib
import sys
import hashlib
import argparse
import paramiko

client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', required=True, type=int)
args = parser.parse_args()
password = getpass.getpass('SSH password: ')
client.connect(args.host, port=args.port, username='root',
               password=password, timeout=20, auth_timeout=20, banner_timeout=20,
               look_for_keys=False, allow_agent=False)
del password
key = client.get_transport().get_remote_server_key()
print(json.dumps({'connected': True, 'host_key_sha256_hex': hashlib.sha256(key.asbytes()).hexdigest()}), flush=True)
sftp = client.open_sftp()
for line in sys.stdin:
    try:
        req = json.loads(line)
        if req['op'] == 'exec':
            stdin, stdout, stderr = client.exec_command(req['cmd'], timeout=req.get('timeout', 120))
            out, err = stdout.read().decode('utf-8', 'replace'), stderr.read().decode('utf-8', 'replace')
            result = {'exit_code': stdout.channel.recv_exit_status(), 'stdout': out, 'stderr': err}
            if req.get('save'):
                pathlib.Path(req['save']).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        elif req['op'] == 'get':
            pathlib.Path(req['local']).parent.mkdir(parents=True, exist_ok=True)
            sftp.get(req['remote'], req['local'])
            result = {'downloaded': req['remote'], 'local': req['local']}
        elif req['op'] == 'put':
            sftp.put(req['local'], req['remote'])
            result = {'uploaded': req['remote']}
        elif req['op'] == 'close':
            break
        else:
            raise ValueError('unsupported operation')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), flush=True)
sftp.close()
client.close()
