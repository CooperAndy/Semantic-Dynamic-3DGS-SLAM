"""Trace only the child ORB launched here; capture SIGILL instruction, no other processes."""
import ctypes,os,signal,subprocess,json,time
from pathlib import Path
R=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012');O=R/'orb_matched_v2';P=R/'orb_sigill_debug';P.mkdir(exist_ok=False)
m=json.loads((O/'preparation.json').read_text());v=next(x for x in m['hashes'] if x.endswith('ORBvoc.txt'))
lib=ctypes.CDLL(None,use_errno=True);lib.ptrace.restype=ctypes.c_long
def ptrace(req,pid,addr=0,data=0):
    ret=lib.ptrace(req,pid,ctypes.c_void_p(addr),ctypes.c_void_p(data))
    if ret==-1:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return ret
def child():ptrace(0,0)
env=os.environ.copy();env['LD_LIBRARY_PATH']=m['build_command'][m['build_command'].index('-lORB_SLAM3')-1][2:]
# Preserve the verified complete library search path from run.sh.
line=next(x for x in (O/'run.sh').read_text().splitlines() if x.startswith('export LD_LIBRARY_PATH='));env['LD_LIBRARY_PATH']=line.split('=',1)[1].strip('"');env['OMP_NUM_THREADS']='4'
log=(P/'tracking.log').open('w');cmd=[str(O/'stereo_kitti_headless'),v,str(O/'settings.yaml'),str(R/'matched_stereo')]
p=subprocess.Popen(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=child)
os.waitpid(p.pid,0);ptrace(0x4200,p.pid,0,8);ptrace(7,p.pid);started=time.time();report={'command':cmd,'child_pid':p.pid}
try:
    while time.time()-started<100:
        tid,status=os.waitpid(-1,0x40000000|os.WNOHANG)
        if not tid:time.sleep(.01);continue
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            if tid==p.pid:report.update(status='EXITED_WITHOUT_SIGILL',wait_status=status);break
            continue
        sig=os.WSTOPSIG(status)
        if sig==signal.SIGILL:
            regs=(ctypes.c_ulonglong*27)();ptrace(12,tid,0,ctypes.addressof(regs));rip=regs[16]
            maps=Path(f'/proc/{tid}/maps').read_text();(P/'maps.txt').write_text(maps)
            mapping=next(x for x in maps.splitlines() if int(x.split()[0].split('-')[0],16)<=rip<int(x.split()[0].split('-')[1],16))
            fields=mapping.split();start=int(fields[0].split('-')[0],16);offset=rip-start+int(fields[2],16);binary=fields[-1]
            obj=subprocess.run(['objdump','-d',f'--start-address={offset-16}',f'--stop-address={offset+32}',binary],capture_output=True,text=True)
            report.update(status='SIGILL_CAPTURED',tid=tid,rip=hex(rip),mapping=mapping,file_offset=hex(offset),disassembly=obj.stdout+obj.stderr);break
        ptrace(7,tid,0,0 if sig in [signal.SIGTRAP,signal.SIGSTOP] else sig)
finally:
    try:os.kill(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    log.close();(P/'diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
