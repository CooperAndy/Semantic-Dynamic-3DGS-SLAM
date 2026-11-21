"""Package a local CPU-tested candidate and register future diagnostic work, without running it."""
from pathlib import Path
from datetime import datetime,timezone
import ast,difflib,hashlib,json,shutil
F=Path(__file__).resolve().parents[1]
B=F.parent/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012'
T18=F.parent/'CHY_T18_CPU离线补全误差诊断_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
expected={B:'36f3b582ad32adf3c3a9536d66cb7ffe63c38589f7e9f8a0d3267294eba8e30a',T18:'3c3e27180180455b4b065157f15291a403b59cbfb218fe9ddfe5f04de1e032c1'}
integrity=[]
for root,h in expected.items():
 assert sha(root/'delivery_manifest.json')==h
 manifest=json.loads((root/'delivery_manifest.json').read_text(encoding='utf-8'))
 for row in manifest['files']:assert sha(root/row['path'])==row['sha256'],row['path']
 integrity.append({'root':str(root),'files':len(manifest['files']),'manifest_sha256':h,'unchanged':True})
dump(F/'receipts/historical_integrity.json',{'status':'PASS','inputs':integrity,'frozen_files_modified':0})
snapshot=F/'base_snapshot';snapshot.mkdir(exist_ok=True)
paths=['source/dg_slam/'+name for name in ['trajectory_filler.py','factor_graph.py','depth_video.py']]
for seed in [1729,2718,31415]:
 for arm in ['NO_GS_REFINE','LOCAL_ID']:paths.append(f'runs/DEV200_s{seed}_{arm}/keyframe_poses.npz')
records=[]
for rel in paths:
 dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/rel,dst)
 assert sha(dst)==sha(B/rel);records.append({'path':rel,'bytes':dst.stat().st_size,'sha256':sha(dst)})
dump(F/'receipts/base_snapshot_manifest.json',{'origin':str(B),'files':records,'GT_included':False})
validation=json.loads((F/'receipts/CPU_validation.json').read_text(encoding='utf-8'))
assert validation['status']=='PASS' and validation['CUDA_initialized'] is False
if not (F/'receipts/CPU_validation_original_inputs.json').exists():
 shutil.copyfile(F/'receipts/CPU_validation.json',F/'receipts/CPU_validation_original_inputs.json')
patch=[];source=[]
for p in sorted((F/'candidate').rglob('*.py')):
 ast.parse(p.read_text(encoding='utf-8'))
 rel=p.relative_to(F/'candidate').as_posix();old=B/'source'/rel
 before=old.read_text(encoding='utf-8').splitlines(keepends=True) if old.exists() else []
 after=p.read_text(encoding='utf-8').splitlines(keepends=True)
 for line in difflib.unified_diff(before,after,fromfile='a/'+rel if old.exists() else '/dev/null',tofile='b/'+rel):
  patch.append(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n')
 source.append({'path':rel,'sha256':sha(p),'base_sha256':sha(old) if old.exists() else None})
(F/'patches/T19_vs_T17.patch').write_text(''.join(patch),encoding='utf-8')
dump(F/'receipts/candidate_source.json',{'files':source,'patch_sha256':sha(F/'patches/T19_vs_T17.patch'),'baseline':str(B/'source')})
registration={
 'task':'T19','status':'CPU_CANDIDATE_VALIDATED_GPU_NOT_RUN',
 'registered_at_UTC':datetime.now(timezone.utc).isoformat(),
 'timing':'Registered after CPU engineering tests and T18 exposed-DEV analysis, before any T19 GPU experiment. Not a blind scientific preregistration.',
 'current_scope':'Local repair, CPU tests, instrumentation and future diagnostic registration only; no remote, GPU, SLAM, deployment or original-file edits.',
 'historical_inputs':integrity,'candidate_source':source,
 'future_data_role':'Historically exposed DEV; no new formal acceptance or test-independence claim',
 'future_window':{'state_build_input_frames':[0,200],'filler_input_frames':[144,176],
                  'reason':'Includes original batches 144--159 and 160--175 surrounding known 157--172 error peak.',
                  'cannot_reconstruct_state_from_T17':True,
                  'needed_missing_state':['source disparities','feature maps','context tensors','network state and RNG','video masks and intrinsics']},
 'future_seed':2718,'seed_selection_reason':'Known T18 abnormal ID seed, selected for mechanism diagnosis, not unbiased quality evaluation.',
 'future_builds':['NO_GS_REFINE full DEV200 state','LOCAL_ID full DEV200 state'],
 'future_filler_cases':[
  {'case':'N_MAP_LEGACY','arm':'NO_GS_REFINE','anchors':'mapping_snapshot','preserve_KF':False,'mixed_geometry_diagnostic':True},
  {'case':'N_MAP_FIXED','arm':'NO_GS_REFINE','anchors':'mapping_snapshot','preserve_KF':True,'mixed_geometry_diagnostic':True},
  {'case':'N_NATIVE_FIXED','arm':'NO_GS_REFINE','anchors':'native_final_DROID','preserve_KF':True,'mixed_geometry_diagnostic':False},
  {'case':'I_MAP_LEGACY','arm':'LOCAL_ID','anchors':'mapping_snapshot','preserve_KF':False,'mixed_geometry_diagnostic':True},
  {'case':'I_MAP_FIXED','arm':'LOCAL_ID','anchors':'mapping_snapshot','preserve_KF':True,'mixed_geometry_diagnostic':True}],
 'future_state_control':'Capture native pre-injection active video state and RNG once per arm, restore before each case; hash all active poses/disparities/features/context/masks/intrinsics, preserve network eval mode and buffers. Check same pre-policy state hashes before every branch. Never reuse a mutated filler state as the next control.',
 'future_engineering_gates':[
  'Index/order match 32 requested frames; no duplicates or omitted frames',
  'Fixed-mode output at keyframe timestamps equals selected anchors in w2c representation; c2w inverse tolerance <= 1e-6 m',
  'Only non-keyframes are optimization variables in fixed mode; original anchors unchanged and video count restored',
  'All poses, learned updates, effective weights and residuals finite; exact selected source recorded',
  'Initialization and all six update stages logged for selected batches; max trace reserved bytes 256 MiB per case',
  'Same-state hashes PASS, network eval and RNG restored across paired cases',
  'No GT access in execution; offline GT scoring only after raw sealing',
  'Any failure aborts the diagnostic; no silent clamp, GT-based acceptance or fallback'],
 'future_scoring':'Compare on same 32 frames with a single per-arm alignment fitted to native DROID 24 anchors and reused across modes; separately report KF/nonKF errors, step size, mask-applied weights and within-iteration fixed-target residuals. These surrogate residuals are not the full BA objective. No improvement percentage is an adoption gate for this selected diagnostic.',
 'GPU_runtime_validation':False,'coherent_ID_geometry_implemented':False,'formal_acceptance_changed':False,
 'GPU_runs':0,'remote_connections':0}
dump(F/'diagnostic_registration.json',registration)
report='''# T19：补全接口修复与 CPU 验证

已完成独立候选代码、CPU 控制逻辑测试及后续诊断登记。**没有启动 GPU，没有连接服务器，没有部署或改写 T17/T18。** 当前状态为 `CPU_CANDIDATE_VALIDATED_GPU_NOT_RUN`。

## 已实现的改动

1. **关键帧直接返回固定锚点**。`trajectory_filler.py` 在原来的 16 输入帧批次内识别关键帧，只为非关键帧建立临时节点。输出按原输入顺序重组，精确关键帧时刻不再返回优化过的副本；全关键帧批次无需进入网络补全。显式的 `preserve_keyframes=False` 仅供后续旧逻辑仪表化对照。
2. **异常清理和身份检查**。拒绝重复/逆序/小数帧号、无左锚点及临时缓冲溢出。优化或日志失败时，通过 `finally` 恢复 video 计数和原回调；原锚点被移动则报错终止。保持六轮更新、BA 参数、神经权重、分割掩码和插值公式。
3. **显式的锚点来源契约**。NO_GS 的 `native_final_DROID` 路径保留原 DROID 位姿与几何状态，避免用 mapping 当时的粗位姿覆盖最终位姿。它是新的管线候选，不等同于冻结的 T17 NO_GS，也没有精度通过结论。`mapping_snapshot` 只在显式标记混合几何诊断后允许注入；ID 不能通过改用原 DROID 锚点来悄悄丢掉精修。**一致的 ID 几何衔接仍未实现，现阶段不宣称修复完成。**
4. **逐轮旁路记录**。选中的批次保存临时初值、固定锚点、源视差和特征、learned delta/weight、分割掩码作用后的实际 native 权重、每轮 BA 前后位姿和重投影残差。日志不调整目标、权重或步长，不使用 GT 决定更新。256 MiB 是每个 case 的未压缩负载预留上限，超限明确终止。

## CPU 验证结果

18 组检查通过。六份现有运行的实际 Python 路由均为 **24 个固定锚点 + 176 个待补全帧**，保留原 16 输入帧边界；40 个随机窗口覆盖交错锚点、全锚点批次、单锚点及尾批。还执行了候选的实际 `__call__` / `__fill` 控制流程、六轮模拟优化、优化异常/日志失败清理和临时缓冲溢出检查。

用相同 CPU 后端替身分别执行原版和候选的 FactorGraph.update / DepthVideo.ba，比较传给 native BA 的 target、有效权重、边和优化参数以及返回状态，一致。日志读取器对已知位移的 CPU 样例正确计算 c2w 步长与固定 learned-target 残差，并拒绝把不完整 mock 日志当作真实网络诊断。

测试使用真实 CPU PyTorch 张量，但 **SE(3)、网络/相关模块和 native BA 使用测试替身**。CUDA lazy initialization 被主动阻止，最终 `CUDA_initialized=false`。因此这里只证明 Python 接口控制和日志链路，不证明实际 CUDA 兼容、真实网络更新行为或位姿精度改善。没有重跑 T17，也没有给候选生成伪造的 ATE/RPE。

详见 `receipts/CPU_validation.json`。内附原始小型测试输入及逐文件哈希，可在 CPU 独立复现。

## 逐轮日志如何读

`scripts/analyze_trace.py` 只使用 NumPy/SciPy，读取每轮四个真实阶段：graph_before_ba、native_before_ba、native_after_ba、graph_after_ba。输出最大 c2w 平移/旋转步长、有效权重覆盖率、learned delta 最大值，以及本轮固定 learned target / 有效权重下的前后残差。

这个残差是诊断代理量，不是完整 native BA 目标；不同轮的 target/weight 会变化，不能把跨轮下降直接当作同一目标下降，更不能替代 GT 轨迹精度。CPU 测试样例日志只证明记录器和读取器，不是真实神经补全结果。

## 下一轮已登记，尚未运行

`diagnostic_registration.json` 将未来 GPU 工作限定为已曝光 DEV 的机制诊断：选择已知异常种子 2718；需要分别重建 NO_GS 与 ID 的完整 0–199 前端状态，再只补全 **144–175**，覆盖异常附近原来的两个 16 帧批次。T17 未保存完整视差/特征/上下文，不能仅凭已有位姿数组重演真实补全。

五个诊断 case 为 NO_GS mapping 锚点的原逻辑/保留关键帧逻辑、NO_GS 最终 DROID 锚点的保留逻辑，以及 ID mapping 锚点的原逻辑/保留逻辑。每个分支都必须恢复同一份注入前视频状态和 RNG，核对活跃位姿、视差、特征、上下文、掩码与内参哈希，不能接着上一分支已改动的状态运行。

登记已冻结窗口、种子、case、状态控制、日志上限与工程门槛；创建时间在 CPU 测试和 T18 分析之后、任何 T19 GPU 实验之前，明确不是盲测预注册。GT 只允许在原始输出封存后离线评分。此窗口是按已知误差选出的诊断窗口，任何改善都不能当作正式采用通过。

## 文件与接入方式

- `candidate/dg_slam/`：三个改动文件与三个新增模块，共六份候选源码。
- `patches/T19_vs_T17.patch`：相对冻结 T17 源码的完整差异。
- `base_snapshot/`：原三个源码与六份关键帧测试输入，不含 GT。
- `scripts/validate_cpu.py`：CPU 复现测试。
- `scripts/analyze_trace.py`：未来逐轮日志的 CPU 读取器。
- `receipts/historical_integrity.json`：冻结 T17/T18 的完整逐文件核验。
- [README_CPU与接入.md](README_CPU与接入.md)：CPU 命令与未来独立 runner 的接入片段。

候选须应用到新工作目录，使用新的 runner、输出路径和登记文件；不能覆盖冻结目录或复用 T17 采用标签。T14 FULL 不采用、T15 原 FAIL、T16 负面结果、T17 两者未资格通过的状态均保留。
'''
(F/'T19_补全接口修复与CPU验证报告.md').write_text(report,encoding='utf-8')
readme='''# T19 CPU 验证及接入

依赖：Python 3、CPU PyTorch、NumPy、SciPy。无需 CUDA、DG-SLAM 扩展、网络或 GPU。内附源码引用原作者/前序实现，CHY 只拥有 T19 的改动和诊断工作，不替队友作验收签字。

在本目录执行 CPU 测试，使用包内小型输入快照：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:T19_BASE_ROOT = (Resolve-Path './base_snapshot').Path
python ./scripts/validate_cpu.py
Remove-Item Env:T19_BASE_ROOT
```

测试替身明确隔离 SE3/网络/相关模块/native BA，CUDA 初始化调用会失败。每次测试写入新 `receipts/cpu_cases_*`，避免覆盖先前样例；`CPU_validation.json` 记录最近一次验证。测试重新运行后交付 manifest 需要重新封存，原交付 ZIP 不受影响。

## 未来 GPU runner 的接入片段（本轮不执行）

先将六份 candidate/dg_slam 源码放入独立的新 source/dg_slam 目录，其余依赖沿用已校验版本。使用 T19 独立 runner，完整跟踪后、作者 terminate_woBA 前接入。**去掉 T17 runner 原先覆盖 video.poses 的语句**，否则会在 adapter 前就破坏 native state。

```python
from dg_slam.filler_adapter import configure_filler

# matrices: 当前 mapping 保存的 (N,4,4) CPU NumPy c2w。
# 先保存完整的注入前视频/网络缓冲状态和 RNG，后续每个 case 都恢复。
# output 必须是新 T19 路径；本片段不能直接替换冻结 T17 runner 后原地运行。
configure_filler(
    model, matrices, arm="NO_GS_REFINE", policy="native_final_DROID",
    output=output / "N_NATIVE_FIXED", preserve_keyframes=True,
)
trajectory32 = model.terminate_woBA(data[144:176])
```

ID mapping 锚点对照必须显式传 `policy="mapping_snapshot", mixed_geometry_diagnostic=True`；旧逻辑对照另传 `preserve_keyframes=False`。不同 case 使用新输出目录、同一份恢复后的原生视频状态，并核对哈希；此 adapter 本身不代替状态快照恢复器或完整 runner。

adapter 是实际可调用的候选接入函数，但尚未部署、尚未验证真实 CUDA。ID 的一致几何重求解尚缺，混合诊断标记不能解释为已经解决。

未来有真实日志后，CPU 读取示例：

```powershell
python ./scripts/analyze_trace.py ./YOUR_CASE/trace --output ./trace_steps.csv
```

输出必须是新文件。每轮固定 learned target 残差仅作诊断代理量，不能据此宣布真实轨迹改善。没有真实日志时不要把 receipts 中的 CPU 测试替身样例用于科学结论。
'''
(F/'README_CPU与接入.md').write_text(readme,encoding='utf-8')
print(json.dumps({'status':'PREPARED','candidate_files':len(source),'snapshot_files':len(records),'CPU_check_groups':validation['check_groups'],'GPU_runs':0,'historical_files_checked':sum(x['files'] for x in integrity)},ensure_ascii=False))
