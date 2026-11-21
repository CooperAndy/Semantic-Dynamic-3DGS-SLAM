import json,hashlib,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'04_系统整合与最终归档';O=D/'final_delivery_20261007_v1'
proof=json.loads((O/'server_delivery_sync.json').read_text(encoding='utf-8'))
assert proof['status']=='PASS_REMOTE_04_DELIVERY'
report='''# 04执行完成报告（2026-10-07）

**04前置已补齐，系统整合与最终归档完成。后续评价和候选方法扩展按用户要求暂停。** 本次仅CPU整合、核验和归档，没有新推理、融合或调参。

## 前置与交付

- 01初始化/遮挡/RGB一致性核验、02固定方法与独立GT评价、03冻结选择及隔离评价均有既有证据入口；各数据窗口的曝光身份保持原样。
- 展示包含5份固定地图、15个预先固定案例、81份原生数值渲染方法对照；支持类别、错误和分数视图。三维中心点浏览仅供属性检查，不用于渲染指标。
- 完整导出907,488个KITTI08 S0语义Gaussian。CPU重读验证稳定ID、几何、RGB、opacity、尺度和旋转一致；unknown保持遮挡属性。
- 本地和服务器均通过25份状态对应、81份显示对应及完整导出核验。服务器校验主包、全部封存文件及4份既有大型依赖包通过。
- KITTI-360评价RGB不读取、不展示；展示适配编号问题及修复记录保留。冻结的语义渲染、GT评价和判优结果没有修改。

## 使用入口

双击[打开展示.cmd](final_delivery_20261007_v1/打开展示.cmd)；当前服务为 http://127.0.0.1:8765/ 。详细[最终报告](final_delivery_20261007_v1/阶段3_最终报告_20261007.md)、[复现说明](final_delivery_20261007_v1/复现与交付说明.md)、[前置核验](final_delivery_20261007_v1/04_前置核验与完成边界.md)位于交付目录。

主包：`stage3_04_final_delivery_20261007_v1.tar.gz`，74,402,753字节、316文件；SHA256：`c5397d6a68f1184506e44610b539a81f9a230bc14d757ab6e1c1309378ec702e`。大型历史资产通过`bundle_dependencies.json`引用，不重复打包。服务器位置：`/root/autodl-tmp/stage3_chy_20261006/04_系统整合与最终归档`。

## 完成边界

04工程完成不等于方法质量PASS。默认S0保持冻结；加权方法未通过跨窗口既定判据。KITTI08真实GT为稀疏LiDAR投影、已曝光DEV，不能代替稠密二维GT或逐Gaussian三维准确率；KITTI-360是项目隔离公开数据评价，不是官方隐藏Test。Stage2正式FAIL和冻结边界保留。

稠密边界、严格长期静态/遮挡变化评价、S2-Z/S2-V及学习扩展仍属后续研究；目前暂停，不作为04未完成事项。
'''
(D/'04_执行完成报告_20261007.md').write_text(report,encoding='utf-8')
(D/'README.md').write_text('# 04_系统整合与最终归档\n\n状态：2026-10-07完成；本地及服务器CPU核验、封存哈希通过。\n\n入口：[完成报告](04_执行完成报告_20261007.md)、[启动展示](final_delivery_20261007_v1/打开展示.cmd)、[总报告](final_delivery_20261007_v1/阶段3_最终报告_20261007.md)、[复现说明](final_delivery_20261007_v1/复现与交付说明.md)。\n\n历史CPU准备材料保留。04工程完成，方法质量未通过既定跨窗口判据；默认S0，05/06及后续评价暂停。\n',encoding='utf-8')
src=(O/'阶段3_最终报告_20261007.md').read_text(encoding='utf-8').replace('(quality_coverage_by_protocol.png)','(../04_系统整合与最终归档/final_delivery_20261007_v1/quality_coverage_by_protocol.png)')
(R/'reports/阶段3_最终报告.md').write_text(src,encoding='utf-8')
p=R/'执行进度与下一步.md';old=p.read_text(encoding='utf-8-sig')
lead='''# 阶段3执行进度（2026-10-07）

**最新状态：04前置补齐，04系统整合与最终归档完成；后续评价与扩展按用户要求暂停。** 本地与服务器主包/封存文件/四份依赖包哈希和CPU对应核验通过。默认S0保持冻结，工程完成不等于质量PASS。

见[04完成报告](04_系统整合与最终归档/04_执行完成报告_20261007.md)、[总报告](reports/阶段3_最终报告.md)。展示15个固定案例、81份原生方法对照，完整导出907,488个S0语义Gaussian。本轮没有新GPU运行、融合或阈值调整。05/06条件扩展未启动。

## 04之前的执行记录（历史状态保留）
'''
if '最新状态：04前置补齐' not in old:p.write_text(lead+old.replace('# 阶段3执行进度（2026-10-07）','',1),encoding='utf-8')
status={'status':'COMPLETE_04','CPU_only_this_step':True,'local_independent_QA':'PASS','remote_delivery_QA':proof['status'],'quality_gate_passed':False,'default_method':'S0','research_followup':'PAUSED_BY_USER','05_06_started':False}
(D/'04_completion_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=[D/'README.md',D/'04_执行完成报告_20261007.md',D/'04_completion_status.json',R/'reports/阶段3_最终报告.md',p,O/'server_delivery_sync.json',O/'archive_receipt.json',O/'remote_delivery_verify_command.json',R/'scripts/complete_stage04_docs.py',R/'scripts/sync_verify_stage04.py']
manifest={f.relative_to(R).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
mp=D/'completion_sync_hashes.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
archive=D/'stage3_04_completion_sync_20261007_v1.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for f in files+[mp]:t.add(f,arcname=f.relative_to(R).as_posix())
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files)+1}),flush=True)
