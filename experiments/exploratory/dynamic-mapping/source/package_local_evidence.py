from pathlib import Path
import zipfile,hashlib,json
base=Path(r'C:\Users\LOCAL_USER\Desktop\fyp');stage=base/'阶段2';out=stage/'第十七轮_最终备份与论文_20261031/results/local_reports_and_code.zip';selected=[]
for d in stage.iterdir():
 if d.is_dir() and any(d.name.startswith('第'+x+'轮_') for x in ['三','四','五','六','七','八','九','十','十一','十二','十三','十四','十五','十六']):
  for p in d.rglob('*'):
   if p.is_file() and p.suffix.lower() in ('.md','.py','.diff','.csv','.json','.yaml','.txt','.log') and p.stat().st_size<8_000_000 and '__pycache__' not in p.parts and 'scene20' not in str(p).lower():selected.append(p)
for p in [stage/'Scene01_576_DEV第二批复核报告.md',stage/'Scene01_576_DEV_预测ID跨GT映射.csv',stage/'Scene01_576_DEV_GT关联逐对审计.csv',stage/'06_独立复核与TestOneShot/06_下一轮DEV窗口选择与概率融合预注册_20261003.md',base/'2026.9.6-9/第11R_E3-E4-E6a_Holdout公平对照_结果讲解.md',base/'2026.9.6-9/第9R至11R_置信度感知GS建图_操作手册.md',base/'2026.9.6-9/第10R_E6b_置信度加权GS建图/README_E6b结果与修正诊断.md',base/'2026.9.6-9/第10R_E6c_受控高斯增殖/README_E6c结果与结论.md',base/'2026.9.6-9/第11R_静态语义GaussianMap与最终评估/第11R_Holdout公平对照/server_results/analysis/fair_holdout_summary.json',base/'2026.9.6-9/第14R_E6a跨场景泛化验证/server_results/results/formal_analysis/cross_scene_summary.json']:
 if p.is_file():selected.append(p)
for rel in ['06_独立复核与TestOneShot/dev_motion_expanded_20261002_v5/run_expanded.py','06_独立复核与TestOneShot/dev_motion_expanded_20261002_v5/motion_history_v5.py','06_独立复核与TestOneShot/dev_motion_jointtrust_20261002_v7/trust_gate.py','06_独立复核与TestOneShot/dev_motion_trustgate_20261002_v6/trust_gate.py']:
 q=stage/rel
 if q.is_file():selected.append(q)
selected=list(dict.fromkeys(selected));manifest=[]
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for p in selected:
  rel=p.relative_to(base).as_posix();z.write(p,rel);manifest.append({'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 z.writestr('CONTENT_SHA256.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print('files',len(selected),'bytes',out.stat().st_size,'sha256',hashlib.sha256(out.read_bytes()).hexdigest())
