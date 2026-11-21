"""Ephemeral SSH console. Password comes only from a non-echoing terminal prompt."""
import getpass
import hashlib
import json
import stat
import sys
from pathlib import Path
import paramiko

sys.dont_write_bytecode = True
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
password = getpass.getpass('SSH password (not stored): ')
client.connect('SERVER_HOST', port=36491, username='root',
               password=password, timeout=15, auth_timeout=20, banner_timeout=20,
               allow_agent=False, look_for_keys=False)
del password
print('CONNECTED host_key_sha256=' + hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest(), flush=True)
while True:
    try:
        request = json.loads(input('ACTION> '))
        if request['op'] == 'close':
            break
        if request['op'] == 'exec':
            stdin, stdout, stderr = client.exec_command(request['cmd'], timeout=request.get('timeout', 60))
            if 'stdin' in request:
                stdin.write(request['stdin'])
                stdin.channel.shutdown_write()
            out = stdout.read().decode('utf-8', errors='replace')
            err = stderr.read().decode('utf-8', errors='replace')
            result = {'exit_code': stdout.channel.recv_exit_status(), 'stdout': out, 'stderr': err}
        elif request['op'] == 'upload':
            with client.open_sftp() as sftp:
                for local, remote in request['files']:
                    sftp.put(local, remote)
            result = {'uploaded': len(request['files'])}
        elif request['op'] == 'download':
            with client.open_sftp() as sftp:
                for remote, local in request['files']:
                    Path(local).parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote, local)
            result = {'downloaded': len(request['files'])}
        elif request['op'] == 'collect':
            with client.open_sftp() as sftp:
                def walk(remote, local):
                    local = Path(local)
                    local.mkdir(parents=True, exist_ok=True)
                    for item in sftp.listdir_attr(remote):
                        rp = remote + '/' + item.filename
                        lp = local / item.filename
                        if stat.S_ISDIR(item.st_mode):
                            walk(rp, lp)
                        elif stat.S_ISREG(item.st_mode):
                            sftp.get(rp, str(lp))
                        else:
                            raise RuntimeError('Unexpected non-regular result: ' + rp)
                for remote, local in request['trees']:
                    walk(remote, local)
            result = {'collected_trees': len(request['trees'])}
        else:
            raise ValueError('Unknown operation')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except EOFError:
        break
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False), flush=True)
client.close()
