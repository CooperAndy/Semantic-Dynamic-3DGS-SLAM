"""Document the measured live GPU audit without modifying frozen main archive."""
import json,hashlib,tarfile,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'04_系统整合与最终归档';O=D/'live_gpu_20261007_v1'
A=D/'live_gpu_raw_evidence_20261007_v1.tar.gz'
assert hashlib.sha256(A.read_bytes()).hexdigest()=='32810f8bc8bef5505b98a158534be2da3dc5a0c34d4632edb94410d42b277485'
with tarfile.open(A) as t:
 for m in t.getmembers():assert m.isfile() and (R/m.name).resolve().is_relative_to(R.resolve())
 t.extractall(R,filter='data')
j=json.loads((O/'gpu_audit.json').read_text());assert j['status']=='PASS_FULL_NATIVE_GPU_RENDERING'
browser={'status':'PASS_BROWSER_LIVE_GPU_UI','continuously_rendered_frames':115,'rolling_last_20_mean_request_display_ms':246.995,'rolling_last_20_fps':1000/246.995,'original_resolution':[1226,370],'RGB_semantic_score_switch':True,'yaw_and_half_resolution_interaction':True,'half_resolution':[613,185],'no_render_errors_observed':True,'default_restored_to_S0_frame3099':True,'timing_scope':'single observed continuous playback; request+SSH+JPEG+image decode; not a global throughput guarantee','initial_audit_panel_JS_bug':'fixed before final browser QA; no renderer or state changes'}
(O/'browser_live_QA.json').write_text(json.dumps(browser,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=[]
for p in j['performance']:lines.append(f"|{p['frame']}|{p['mode']}|{p['GPU_median_fps']:.2f}|{p['full_frame_server_median_fps']:.2f}|{p['server_p95_ms']:.2f}|")
report='''# 04全量语义Gaussian实时GPU渲染补充核验（2026-10-07）

**全量原生GPU即时渲染与交互验证通过；30 FPS语义端到端门槛未通过。** 这次已实际使用服务器RTX 5090，不是播放离线图片或中心点浏览。全部907,488个KITTI08 Gaussian作为原生光栅器输入，允许原生可见性裁剪，没有采样、LOD或按类别跳过opacity。

## 交付和使用

当前入口：http://127.0.0.1:8766/ 。下次双击[打开实时GPU展示.cmd](打开实时GPU展示.cmd)，已有转发则直接打开；否则输入SSH密码建立转发，密码不保存。服务器GPU渲染服务和本机SSH转发均仅绑定localhost。需要服务器及网络保持在线。

界面支持冻结S0语义、原地图RGB、排序分数，原相机200帧轨迹播放、局部水平转向±15度，原分辨率1226×370和半分辨率613×185。降低分辨率仍使用全量Gaussian；没有承诺一定更快。轨迹或转向展示不是额外GT评价。

服务器进程启动入口（仅在服务停止后使用）：

```bash
nohup /root/miniconda3/bin/python /root/autodl-tmp/stage3_chy_20261006/scripts/live_semantic_gpu.py --serve-existing-audit > /root/autodl-tmp/stage3_chy_20261006/04_系统整合与最终归档/live_gpu_restart.log 2>&1 < /dev/null &
```

代码在阶段3/scripts，源文件快照及JSON结果保留在本目录。`gpu_audit.json`为GPU测量，`live_service_QA.json`为服务请求和实际nvidia-smi显存，`browser_live_QA.json`为浏览器交互记录。

## 正确性与实现

- 几何、RGB、opacity、尺度、旋转和S0证据常驻GPU；仅相机变换及屏幕结果更新。7次三通道原生Gaussian光栅化输出19类别+unknown数值质量；GPU数值argmax后着色，没有通过调色板RGB混合反查类别。
- 3009、3099、3199三个预先固定视角与原数值渲染器类别逐像素一致，分数在既定容差内一致。独立all-one渲染的alpha与20通道之和最大差约1.13e-6。
- 前景unknown/背景已知受控测试通过；保留前景opacity时输出unknown，刻意删除前景opacity的负对照才透出背景类别。
- 原基图和S0状态前后SHA256一致，没有GT融合、训练、方法阈值调整或重新选择方法。只加载冻结NPZ进行显示，不从PLY反向重建证据；全量PLY仍是另一个保持原参数的交付格式。

## 性能实测

RTX5090；PyTorch2.8.0+cu128 / CUDA12.8。每视角/模式预热3次、测12次，顺序执行；以下是本次有限样本，不能外推所有相机姿态。GPU事件耗时包括变换、光栅化和解码路径；服务器耗时还含CPU复制与JPEG编码；均不含SSH与浏览器。

|原帧号|模式|GPU中位FPS|服务器中位FPS|服务器p95 ms|
|---|---|---:|---:|---:|
'''+ '\n'.join(lines)+'''

三个语义视角服务器p95均未达到33.33ms，因此没有签发稳定30FPS实时PASS。3199因原生可见表面/投影负载不同而显著更快，不能用单个高帧率视角代表整个地图。GPU峰值torch分配521,484,288字节（约0.49GiB）；nvidia-smi整设备占用1324MiB（包含CUDA上下文、扩展等，不能与torch分配混用）。

浏览器连续即时渲染115帧，记录时最后20帧请求+解码平均246.995ms，约4.05FPS。真实交互受SSH传输和串行请求限制；这是即时交互式展示，仍未达到端到端30FPS。已核验RGB/语义/分数切换、转向、半分辨率、播放/暂停，默认恢复3099的S0语义。

![3099冻结S0语义即时渲染](3099_semantic.jpg)

![3099原地图RGB即时渲染](3099_rgb.jpg)

## 与此前04的关系

旧04主包及CPU完成记录原样保留，本补充独立版本化封存。04现在同时提供离线结果整合和全量GPU即时交互；渲染正确性与服务功能PASS，稳定30FPS性能未PASS。方法质量与数据曝光身份不变，默认S0；后续科学评价、候选方法和05/06仍暂停。
'''
(O/'04_全量实时GPU渲染验证报告_20261007.md').write_text(report,encoding='utf-8')
(O/'打开实时GPU展示.cmd').write_text('@echo off\n"C:\\anaconda\\python.exe" "%~dp0..\\..\\scripts\\open_live_gpu.py"\n',encoding='utf-8')
for name in ['live_semantic_gpu.py','live_semantic_gpu.html','live_gpu_tunnel.py','open_live_gpu.py','finalize_live_gpu_delivery.py']:shutil.copy2(R/'scripts'/name,O/name)
summary='''\n\n## 04全量GPU即时渲染补充（2026-10-07）

已实际使用RTX5090完成907,488个Gaussian全量原生语义即时渲染，支持RGB/语义/分数、相机轨迹与转向。原固定3视角类别对应和unknown遮挡测试通过；基图与冻结S0状态哈希不变。稳定30FPS性能未通过，前两视角GPU中位约16–22FPS，浏览器连续播放最后20帧约4.05FPS。见[GPU补充报告](LINK)。GPU展示服务已启动；方法质量判定和后续研究暂停状态不变。
'''
for p,link in [(R/'执行进度与下一步.md','04_系统整合与最终归档/live_gpu_20261007_v1/04_全量实时GPU渲染验证报告_20261007.md'),(R/'reports/阶段3_最终报告.md','../04_系统整合与最终归档/live_gpu_20261007_v1/04_全量实时GPU渲染验证报告_20261007.md'),(D/'04_执行完成报告_20261007.md','live_gpu_20261007_v1/04_全量实时GPU渲染验证报告_20261007.md'),(D/'README.md','live_gpu_20261007_v1/04_全量实时GPU渲染验证报告_20261007.md')]:
  text=p.read_text(encoding='utf-8-sig')
  if '## 04全量GPU即时渲染补充' not in text:p.write_text(text+summary.replace('LINK',link),encoding='utf-8')
status=json.loads((D/'04_completion_status.json').read_text());status.update(CPU_only_this_step=False,live_full_GPU_rendering='PASS_FUNCTION_AND_CORRECTNESS',stable_30fps='NOT_PASSED',live_viewer='http://127.0.0.1:8766/',method_quality_unchanged=True)
(D/'04_completion_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=[p for p in O.iterdir() if p.is_file()]+[D/'README.md',D/'04_执行完成报告_20261007.md',D/'04_completion_status.json',R/'执行进度与下一步.md',R/'reports/阶段3_最终报告.md']+[R/'scripts'/n for n in ['live_semantic_gpu.py','live_semantic_gpu.html','live_gpu_tunnel.py','open_live_gpu.py','finalize_live_gpu_delivery.py']]
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files};mp=D/'live_gpu_delivery_hashes.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');archive=D/'stage3_04_live_GPU_complete_20261007_v1.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in files+[mp]:t.add(p,arcname=p.relative_to(R).as_posix())
receipt={'path':archive.relative_to(R).as_posix(),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'sealed_files':len(files),'status':'SEALED_LOCAL_FULL_GPU_AUDIT_AND_DISPLAY'}
(D/'live_gpu_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
