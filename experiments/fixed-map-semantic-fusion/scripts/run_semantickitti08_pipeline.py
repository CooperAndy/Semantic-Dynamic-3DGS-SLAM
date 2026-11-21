"""Small correspondence+visual gate must pass before all200 fixed evaluation."""
import json,hashlib,subprocess,sys
from pathlib import Path
R=Path('/root/autodl-tmp/stage3_chy_20261006');B=R/'02_基础因子DEV与Oracle诊断/semantickitti08_gt_v1';B.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipts=[]
def phase(name,script,args=[]):
 dump(B/'pipeline_status.json',{'status':'RUNNING','phase':name,'completed':receipts});cmd=['/root/miniconda3/bin/python',str(R/'scripts'/script)]+args
 with (B/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 rec={'phase':name,'command':cmd,'exit_code':p.returncode,'log_sha256':sha(B/(name+'.log'))};receipts.append(rec);dump(B/(name+'_receipt.json'),rec)
 if p.returncode:raise RuntimeError(name+' failed; preserve existing evidence')
try:
 audit=R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1/correspondence_audit.json';a=json.loads(audit.read_text());assert a['status']=='PASS_SMALL_CORRESPONDENCE_AUDIT' and a['visual_review_passed']
 source=['run_semantickitti08_pipeline.py','fetch_semantickitti08.py','prepare_semantickitti08_full.py','render_semantickitti08_fixed.py','evaluate_semantickitti08.py','semantickitti_projection.py','http_range_zip.py','semantic_cpu.py','native_semantic_render.py','evaluate_vkitti_gt_dev.py'];assert not (B/'GT_evaluation').exists();dump(B/'source_freeze.json',{'status':'CODE_LOCKED_BEFORE_ALL200_GT_SCORING','files':{'scripts/'+n:sha(R/'scripts'/n) for n in source},'small_audit_sha256':sha(audit),'GT_feedback_or_weight_changes':False})
 phase('01_fetch_all200','fetch_semantickitti08.py',['--mode','full']);phase('02_prepare_sparse_GT','prepare_semantickitti08_full.py');phase('03_fixed_native_render','render_semantickitti08_fixed.py');phase('04_sparse_GT_evaluation','evaluate_semantickitti08.py');dump(B/'pipeline_status.json',{'status':'COMPLETE_REAL_SPARSE_GT_DEV_ALL200','completed':receipts,'new_Test':False,'scientific_PASS':False})
except Exception as e:dump(B/'pipeline_status.json',{'status':'FAILED','error':str(e),'completed':receipts});print(str(e),flush=True);sys.exit(1)
