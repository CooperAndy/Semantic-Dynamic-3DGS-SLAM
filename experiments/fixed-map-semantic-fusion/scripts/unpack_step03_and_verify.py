"""Verify downloaded remote archive, extract only bounded regular files, recount."""
import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];step=root/'03_方法冻结与隔离评价'
package=json.loads(json.loads((step/'remote_package_command_20261007.json').read_text(encoding='utf-8'))['stdout'])
archive=step/'step03_remote_evidence_20261007_v1.tar.gz'
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual==package['sha256'] and archive.stat().st_size==package['bytes']
with tarfile.open(archive,'r:gz') as t:
    for member in t.getmembers():
        assert member.isfile() or member.isdir()
        target=(root/member.name).resolve()
        assert target.is_relative_to((step/'engineering_freeze_20261007_v1').resolve())
        assert not target.exists(),str(target)
    t.extractall(root,filter='data')
(step/'download_package_SHA256_20261007.json').write_text(json.dumps({'status':'PASS','archive_sha256':actual,'bytes':archive.stat().st_size,'remote_digest_matches':True},indent=2)+'\n',encoding='utf-8')
command=[sys.executable,str(root/'scripts/freeze_step03_cpu.py'),'--root',str(root),'--verify-only']
result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
receipt={'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
(step/'local_independent_recount_20261007.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False));sys.exit(result.returncode)
