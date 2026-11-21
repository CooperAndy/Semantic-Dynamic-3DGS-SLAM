from pathlib import Path
import json,subprocess,time,shutil,datetime
P=Path('/root/autodl-tmp/A2_union_20261029');out=P/'resource_samples.jsonl'
for _ in range(720):
 if (P/'stop_monitor').exists():break
 def cmd(args):
  p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);return p.stdout.strip()
 data={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_processes':cmd(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),'gpu':cmd(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader']),'free_bytes':shutil.disk_usage(P).free,'cpu_processes':cmd(['ps','-eo','pid,args'])}
 data['cpu_processes']='\n'.join(x for x in data['cpu_processes'].splitlines() if 'union_worker' in x or 'run_00.py' in x or 'run_03_08.py' in x)
 with out.open('a') as f:f.write(json.dumps(data)+'\n')
 time.sleep(30)
