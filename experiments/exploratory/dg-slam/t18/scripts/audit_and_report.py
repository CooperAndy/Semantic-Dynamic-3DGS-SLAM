"""Independent closed-form SE(3) audit, local evidence snapshot and report. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,hashlib,json,shutil,sys
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).resolve().parents[1]
B=F.parent/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def skew(w):
 x,y,z=w;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
def jac(w):
 t=np.linalg.norm(w);K=skew(w)
 if t<1e-7:return np.eye(3)+K/2+K@K/6
 return np.eye(3)+(1-np.cos(t))/t**2*K+(t-np.sin(t))/t**3*K@K
def screw(W0,W1,alpha):
 # Closed-form group exponential/logarithm using SO(3) rotvec and left Jacobian.
 H=W1@np.linalg.inv(W0);w=Rotation.from_matrix(H[:3,:3]).as_rotvec()
 v=np.linalg.solve(jac(w),H[:3,3]);T=np.eye(4)
 T[:3,:3]=Rotation.from_rotvec(alpha*w).as_matrix();T[:3,3]=jac(alpha*w)@(alpha*v)
 return np.linalg.inv(T@W0)
A=load(F/'receipts/analysis.json')
audit=[];maxmatrix=0.;maxanchor=0.;maxsse=0.
for r in A['runs']:
 seed,arm=r['seed'],r['arm'];z=np.load(F/f'arrays/s{seed}_{arm}.npz')
 k=z['indices'];W=np.linalg.inv(z['seed_c2w']);C=z['interp_injected_mapped'];G=z['groundtruth']
 for frame in range(200):
  a=int(np.searchsorted(k,frame,side='right')-1);b=min(a+1,len(k)-1)
  alpha=0. if a==b else (frame-k[a])/(k[b]-k[a]+.001)
  independent=screw(W[a],W[b],alpha)
  maxmatrix=max(maxmatrix,float(np.max(abs(independent-C[frame]))))
 maxanchor=max(maxanchor,float(np.max(abs(C[k]-z['seed_c2w']))))
 Q,t=z['common_Q'],z['common_t']
 e0=np.linalg.norm(C[:,:3,3]@Q.T+t-G[:,:3,3],axis=1)*100
 e1=np.linalg.norm(z['actual_post_filler'][:,:3,3]@Q.T+t-G[:,:3,3],axis=1)*100
 d=e1**2-e0**2;mask=np.isin(np.arange(200),k)
 maxsse=max(maxsse,abs(float(d.sum())-r['delta_SSE_all_cm2']),abs(float(d[mask].sum())-r['delta_SSE_KF_cm2']),abs(float(d[~mask].sum())-r['delta_SSE_nonKF_cm2']))
 assert np.array_equal(z['actual_post_filler'][~mask],z['actual_restore_KF_only'][~mask])
 audit.append({'seed':seed,'arm':arm,'frames':200,'keyframes':len(k),'nonKF_restore_unchanged':True})
assert maxmatrix<1e-10 and maxanchor<1e-12 and maxsse<1e-10
# Formula fixtures with a known constant and a pure translation, independent of input data.
I=np.eye(4);T=I.copy();T[:3,3]=[1.,-2.,3.]
for alpha in [0.,.125,.5,1.]:
 expected=I.copy();expected[:3,3]=-alpha*T[:3,3]
 assert np.max(abs(screw(I,T,alpha)-expected))<1e-12
 assert np.max(abs(screw(T,T,alpha)-np.linalg.inv(T)))<1e-12
assert 'torch' not in sys.modules and 'lietorch' not in sys.modules
validation={'status':'PASS','independent_closed_form_SE3_vs_SciPy_expm_logm_max_matrix_abs':maxmatrix,'comparisons':1200,'interpolated_exact_KF_vs_seed_max_matrix_abs':maxanchor,'delta_SSE_independent_recompute_max_cm2':maxsse,'constant_and_pure_translation_fixtures':'PASS','original_dense_metrics_max_difference_cm':A['original_dense_metrics_max_difference_cm'],'runs':audit,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_or_native_loaded':False,'remote_connections':0,'GPU_or_SLAM_runs':0}
dump(F/'receipts/independent_CPU_validation.json',validation)
# Preserve only the small inputs necessary for this diagnosis; no networks, map binaries or GPU state.
snapshot=F/'inputs_snapshot';snapshot.mkdir(exist_ok=True)
paths=['delivery_manifest.json','receipts/evaluation_reference.npz','receipts/trial_scores.json','receipts/registered_launch.json','inputs/DEV200_runtime.csv']
for r in A['runs']:
 for name in ['keyframe_poses.npz','trajectories.npz','droid_pose_seeds.npy','keyframe_audit.jsonl']:
  paths.append(f"runs/DEV200_s{r['seed']}_{r['arm']}/{name}")
for name in ['trajectory_filler.py','factor_graph.py','depth_video.py','dg_model.py','motion_filter.py']:
 paths.append('source/dg_slam/'+name)
paths.append('scripts/run_tracking.py')
records=[]
for rel in paths:
 src=B/rel;dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 assert sha(src)==sha(dst);records.append({'path':rel,'bytes':dst.stat().st_size,'sha256':sha(dst)})
dump(F/'receipts/input_snapshot_manifest.json',{'origin':str(B),'files':records,'note':'Small CPU analysis inputs only; original T17 manifest retained as provenance, not all 281 files copied.'})
native=F.parent/'CHY_正式Tracking采用验收_20261012/archive/native/droid/src/droid_kernels.cu'
dst=F/'source_reference/droid_kernels.cu';dst.parent.mkdir(exist_ok=True);shutil.copyfile(native,dst)
dump(F/'receipts/native_source_reference.json',{'source':str(native),'sha256':sha(native),'copy_sha256':sha(dst),'runtime_identity_limit':'Native source is the T14 archive used by the T17 locked environment; no native library executed in T18.'})
# Figures: preserve the per-run distinction and make the 157--172 interval visible.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
order=[(s,a) for s in [1729,2718,31415] for a in ['NO_GS_REFINE','LOCAL_ID']]
variants=['interp_injected_mapped','actual_post_filler','actual_restore_KF_only'];labels=['CPU initialization model','Saved filler output','Restore KF entries only (diagnostic)']
colors=['#4c78a8','#e45756','#54a24b'];x=np.arange(6)
for ax,key,ylabel in zip(axes,['ATE_cm','common_alignment_RMSE_cm'],['ATE, independent rigid alignment (cm)','RMSE, fixed per-run alignment (cm)']):
 for j,(v,l,c) in enumerate(zip(variants,labels,colors)):
  vals=[next(r[key] for r in A['stage_metrics'] if r['seed']==s and r['arm']==a and r['variant']==v) for s,a in order]
  bars=ax.bar(x+(j-1)*.24,vals,.23,label=l,color=c)
  ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
 ax.set_xticks(x,[f'{s}\n'+('NO_GS' if a=='NO_GS_REFINE' else 'ID') for s,a in order]);ax.set_ylabel(ylabel);ax.set_ylim(0,2.7);ax.grid(axis='y',alpha=.2)
axes[0].legend(loc='upper left',ncols=3,fontsize=8);fig.suptitle('T18 / Existing DEV200 results / CPU diagnosis, not acceptance')
fig.savefig(F/'figures/stage_comparison.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained',sharex=True)
frame_rows=list(csv.DictReader((F/'T18_逐帧误差分解.csv').open(encoding='utf-8-sig')))
for ax,seed in zip(axes,[2718,31415]):
 rows=[r for r in frame_rows if int(r['seed'])==seed and r['arm']=='LOCAL_ID']
 ix=np.array([int(r['frame']) for r in rows]);e0=[float(r['interp_injected_mapped_common_error_cm']) for r in rows];e1=[float(r['actual_post_filler_common_error_cm']) for r in rows]
 ax.plot(ix,e0,label='CPU initialization model',color=colors[0]);ax.plot(ix,e1,label='Saved filler output',color=colors[1]);ax.axvspan(157,172,color='#f2cf5b',alpha=.2,label='KF interval [157,172)')
 k=[int(r['frame']) for r in rows if r['is_keyframe']=='True'];ax.scatter(k,np.array(e1)[k],marker='x',color='black',s=30,label='Returned KF duplicates',zorder=4)
 ax.set_ylabel(f'ID seed {seed}\nposition error (cm)');ax.set_ylim(0,10);ax.grid(alpha=.2)
axes[0].legend(loc='upper left',ncols=2,fontsize=9);axes[-1].set_xlabel('Input frame index (0--199)');fig.suptitle('Fixed per-run alignment / Large extra error near frame 157')
fig.savefig(F/'figures/ID_error_localization.png',dpi=180);plt.close(fig)
# Assemble source-backed report from the verified numeric evidence.
def stage(seed,arm,v):return next(r for r in A['stage_metrics'] if r['seed']==seed and r['arm']==arm and r['variant']==v)
table='| 种子 | 分支 | CPU 插值初值 ATE | 实际补全 ATE | 仅恢复 KF 输出 ATE |\n|---|---|---:|---:|---:|\n'
for seed,arm in order:
 table+=f"| {seed} | {'NO_GS' if arm=='NO_GS_REFINE' else 'ID'} | "+' | '.join(f"{stage(seed,arm,v)['ATE_cm']:.6f}" for v in variants)+' |\n'
sse='| 种子 | 分支 | 全部 ΔSSE | KF 部分 | 非 KF 部分 |\n|---|---|---:|---:|---:|\n'
for seed,arm in order:
 r=next(r for r in A['runs'] if r['seed']==seed and r['arm']==arm)
 sse+=f"| {seed} | {'NO_GS' if arm=='NO_GS_REFINE' else 'ID'} | {r['delta_SSE_all_cm2']:+.6f} | {r['delta_SSE_KF_cm2']:+.6f} | {r['delta_SSE_nonKF_cm2']:+.6f} |\n"
text=f'''# T18：CPU 离线轨迹补全误差诊断

本轮已完成现有六次结果的 CPU 检查。最明确的误差入口是：补全器把已有关键帧也重新建成临时节点，经过六轮 motion-only 优化后返回临时副本，而不是返回原关键帧锚点。ID 的两个种子在这一步变差；NO_GS 的三个种子则改善，不能据此宣布补全算法整体无效。

## 范围与结论边界

- 输入：T17 已冻结的 DEV200（帧 0–199），种子 1729、2718、31415，NO_GS_REFINE / LOCAL_ID 六次结果。每次 24 个关键帧，关键帧集合相同。
- 本轮没有连接服务器，没有导入 Torch、lietorch 或 CUDA 扩展，没有启动 GPU、SLAM 或神经补全回放。只在独立 T18 目录写入诊断产物。
- T17 的 281 个交付文件在分析前逐一核验。原轨迹 ATE、RPE1、RPE10 与独立 CPU 重算最大差 {A['original_dense_metrics_max_difference_cm']:.3g} cm。
- 本文涉及已曝光 DEV 的解释性分析。T14 FULL 不采用、T15 原 FAIL、T16 负面结果、T17 `INCONCLUSIVE_NEITHER_QUALIFIED` 均保留；本轮不新增正式通过结论，不修改门槛。

## 1. 更正“关键帧被补全移动”的含义

原先观察到的最大 6.923399 cm 偏移，发生在 **ID / seed 2718 / 帧 157 的返回副本**。存储中的原关键帧锚点在六次补全前后最大仅约 5.1×10⁻⁸ m 差异，属于 float32 逆变换与归一化量级。不能把两者混称为“锚点被改动”。这份新报告更正解释，T17 原档保持冻结。

源码链路如下，链接均为本包保存的原文件：

1. [trajectory_filler.py](inputs_snapshot/source/dg_slam/trajectory_filler.py) 第 93–106 行遍历全部输入帧，按 16 帧分批，没有跳过已有关键帧。类注释说补非关键帧，实际返回包括关键帧在内的全部帧。
2. 第 51–59 行从左右关键帧插值得到初值；第 67–75 行把当前批次全部帧放到新索引 `N:N+M`，连接左右锚点，执行六轮 `graph.update(N, N+M, motion_only=True)`，返回这些临时节点。
3. [droid_kernels.cu](source_reference/droid_kernels.cu) 第 904 行及第 1385–1394 行只更新 `[t0,t1)`，这里即 `[N,N+M)`；原 `0:N` 锚点固定。这也与保存的锚点数据一致。
4. 帧 157 是精确关键帧时刻，插值系数为 0，初值应等于注入锚点；其临时索引为 37（原锚点数 24，批次从 144 开始）。输出副本最终偏离锚点 6.923399 cm，不能由插值间隔或时间戳舍入解释。

ID / seed 31415 的帧 157 同样偏离 3.909628 cm。NO_GS 的最大副本偏移约 1.282 cm，位于帧 35。已有证据将异常定位到“临时节点初值之后至最终返回”的优化阶段；六轮内部的首个异常迭代尚无保存数据。

## 2. 在相同帧集上比较补全前后

为避免把 24 帧稀疏 ATE 与 200 帧完整 ATE 直接比较，这里对每个阶段都使用相同的 200 个 GT 对应帧。ATE 使用各阶段一次无尺度刚体对齐，单位 cm。

“CPU 插值初值”是 NumPy/SciPy 对源码 SE(3) 插值公式的 float64 数学重建；不是当时 GPU 保存的中间轨迹，也没有重跑六轮网络。“仅恢复 KF 输出”只把实际输出中 24 个关键帧位置和旋转恢复为注入锚点，176 个非关键帧保持逐元素不变，属于诊断反事实。

{table}

这些数字表明：NO_GS 三个种子及 ID/1729 的补全降低 ATE；ID/2718、ID/31415 的补全提高 ATE。CPU 插值的 RPE1 较小部分源于平滑插值，不能把它当成具有相同图像跟踪信息的可部署胜出算法。

同一 24 个关键帧子集重新对齐后，ID/2718 ATE 从 1.747276 增至 2.250947 cm，ID/31415 从 1.448152 增至 1.696644 cm。这支持“返回副本产生额外误差”，而不是仅凭稀疏/稠密指标差异作判断。

![阶段对照](figures/stage_comparison.png)

## 3. 增大误差集中在哪里

为得到可相加的误差分解，每次运行只拟合一次对齐：用该次 **注入前最终 DROID 的 24 个锚点** 对 GT 拟合无尺度刚体变换，固定用于全部阶段和全部 200 帧。不同阶段不重新对齐。ΔSSE 为“实际补全减 CPU 插值”的位置平方误差总和，单位 cm²；正值表示变差。不同运行之间没有共用对齐变换。

{sse}

- ID/2718 的净增加 61.081200 cm²，其中 60.255652 cm²（98.65%）来自关键帧输出，非关键帧净增 0.825548 cm²。这里是**净值分解**，非关键帧内部存在很大的正负抵消，不能说非关键帧均正常。
- ID/31415 的净增加 96.084164 cm²，其中非关键帧占 73.87%，关键帧占 26.13%。因此仅恢复关键帧无法解决整个问题。
- 两个异常种子都在 `[157,172)` 区间出现明显误差峰。2718 的帧 158 偏离插值初值 7.894064 cm；31415 的帧 158、159 分别偏离 9.166187、9.023156 cm。异常跨越 144–159 与 160–175 两个批次，现有证据不支持把问题简单归结为某一个批次边界。
- 恢复关键帧后，ID/2718 全轨迹 ATE 从 2.095049 降至 2.033429 cm，仍高于 NO_GS/2718 的 1.607045 cm；ID/31415 从 1.833695 降至 1.806241 cm，仍高于 NO_GS/31415 的 1.607070 cm。ID/31415 的 RPE1 反而从 1.328559 增至 1.337797 cm，说明硬覆盖锚点可能留下相邻帧的不连续，不能把这个反事实直接当成修复上线。

![逐帧定位](figures/ID_error_localization.png)

## 4. 另一个接口风险：注入位姿与最终 DROID 状态不一致

[run_tracking.py](inputs_snapshot/scripts/run_tracking.py) 第 87–95 行取 mapping 时保存的 `estimate_c2w_list`，覆盖当时最终 DROID 的全部锚点位姿，再调用作者补全器。该处没有同步重新求解 DROID 的视差/特征相关状态。

这在 NO_GS 下也发生：NO_GS 的 coarse 与 tracked/mapped 相同，但这些是 **mapping 当时的粗位姿**，不是 frontend 在窗口末尾最终优化后的位姿。与最终 DROID 的最大同帧平移差为约 1.678 cm；ID 的注入差异最大约 2.487 cm。它们不是帧号错位造成的。

这证明存在“替换锚点、保留另一阶段几何状态”的接口条件，但不能单凭现有结果证明它就是网络更新异常的唯一原因。尤其 CPU 直接插值最终 DROID 的 ATE 约 1.800 cm，NO_GS mapping 粗位姿插值约 1.745 cm，最终 NO_GS 补全约 1.607 cm；简单换回最终 DROID 并没有在这个 CPU 反事实里自动更好。

ID/31415 的注入后插值 ATE 1.639411 cm 优于其 mapping 粗位姿插值 1.745756 cm，经过补全却变成 1.833695 cm。它说明这一种子上，补全阶段会破坏原本较好的插值结果；ID/2718 则在注入后插值阶段已经较差，补全又进一步变差。两者不能用一个“ID 初值全部无效”的解释概括。

## 5. 已核查的替代解释与尚缺证据

- **帧身份/顺序**：六次均为 200 帧、严格对应原 RGB 时间戳；关键帧编号、输入帧号、mapping audit 的 coarse 矩阵一致，PASS。
- **c2w/w2c 与四元数**：保存的 xyzw seed 与 mapped c2w 矩阵最大元素误差约 3×10⁻⁸；逆变换后的四元数分量差最多 5.96×10⁻⁸。未发现能解释厘米级偏移的转换错误。
- **深度/视差**：[depth_video.py](inputs_snapshot/source/dg_slam/depth_video.py) 第 80–91 行对传入深度采样并取倒数，再存入 `disps` / `disps_sens`。不能把 filler 传入 `depths, depths` 直接解释为“把深度当视差”。这里仅核对代码，不声称已验证全部未保存的深度张量。
- **插值分母 +1e-3**：与去掉该项相比，六次最大位置影响不超过 0.001209 cm（约 0.0121 mm）。不足以解释 6.92–9.17 cm 偏移。
- **帧索引时间与 RGB 实际时间**：换用实际 RGB 时间的 CPU 插值与帧索引插值最大差约 0.728 cm；这是另一种插值假设，不是已发现的时间戳错误。精确关键帧处两者都回到同一锚点，解释不了帧 157 的异常。
- **CPU 插值实现**：用独立 SO(3) rotvec + 左雅可比闭式 SE(3) 实现交叉核验 1200 个矩阵，最大元素差 {maxmatrix:.3g}；关键帧插值与 seed 最大矩阵差 {maxanchor:.3g}。常量与纯平移解析例子通过，误差分解独立重算通过。
- **尚缺**：T17 没有保存每轮 learned delta、weight、重投影残差、视差/相关特征和中间临时位姿。因此可以定位误差进入的阶段及帧，但还不能区分错误匹配、低置信边、位姿/深度状态不协调或其他网络/BA 数值原因；也不能确认动态物体或遮挡就是原因。

## 6. 下一步修复方向（本轮未执行）

先修订并测试输出契约：关键帧时间戳直接返回固定锚点，只为非关键帧建立补全变量；同时明确 NO_GS 使用 mapping 时粗位姿还是最终 DROID 位姿，分别定义一致的几何状态来源。ID 路线需要解决精修锚点与补全器几何状态的衔接。

仅保留锚点仍不足以解决非关键帧峰值。后续如获准运行 GPU，应先对已有 DEV 的帧 153–175 做最小仪表化回放，记录初始化及六轮临时位姿、边权/残差、步长和视差状态，定位第一轮异常；GT 只用于事后评分，不能按逐帧 GT 接受/拒绝更新。冻结新候选及验证门槛后，才评估完整 DEV 窗口。此次 CPU 反事实不构成新候选验收，不重跑正式批次。

## 产物与复现

- 六张 CSV：阶段、相同子集、逐帧、关键帧身份、关键帧区间、补全批次误差。
- `arrays/`：六次运行的五种阶段矩阵、GT、锚点及固定对齐，便于逐帧核对。
- `inputs_snapshot/`：本诊断所需的小型原始输入和源码，逐文件 SHA256 记录在 `receipts/input_snapshot_manifest.json`；原始 CUDA 源码只作文本证据，存于 `source_reference/`。
- `receipts/independent_CPU_validation.json`：独立数值核验及 CPU 环境记录。`receipts/snapshot_reproducibility.json` 记录内附输入快照重算的指标、CSV 与数组一致性；`receipts/T17_final_integrity.json` 记录交付时再次核验冻结输入。
- [README_CPU.md](README_CPU.md)：仅 NumPy/SciPy/Matplotlib 的复现命令、包范围与限制。

本轮结论是“已定位可观察的误差入口与接口风险”，不是“已证实并修复所有内部根因”。没有改动冻结实验，没有启动 GPU。
'''
(F/'T18_CPU离线补全诊断与下一步.md').write_text(text,encoding='utf-8')
readme='''# T18 CPU 复现

本包由 CHY 接手分析工作独立生成。T17 的轨迹与源码仅作为历史输入保留，不改写队友产物或冻结结果。诊断只覆盖六次 DEV200，不能外推完整 TUM 序列或未曝光测试。

依赖：Python 3、NumPy、SciPy、Matplotlib。无需 Torch、CUDA、网络或 GPU。环境版本见 `receipts/independent_CPU_validation.json`。

默认 `scripts/analyze_filler.py` 读取同级原 T17 目录并核验其 281 文件 manifest。独立解压本包后，可以使用内附小型快照：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:T18_INPUT_ROOT = (Resolve-Path './inputs_snapshot').Path
python ./scripts/analyze_filler.py
Remove-Item Env:T18_INPUT_ROOT
```

快照模式核验 `receipts/input_snapshot_manifest.json`，不是声称快照包含原 T17 的所有交付文件。内附原 `delivery_manifest.json` 仅用于核对来源及冻结指纹。`audit_and_report.py` 另外需要同级 T17 与 T14 原始包，以便重新获取历史源码；本包已附其核验结果、报告、图和输入，不运行该脚本也能重算数值。

`source_reference/` 与 `inputs_snapshot/source/`、快照中的原 run_tracking.py 含 GPU 代码，只供文本核对，不要执行。CPU 复现只执行 T18 自有分析脚本。

“插值初值”是 float64 数学反事实，不是保存的原 GPU 中间位姿；“恢复 KF”也是离线诊断反事实。二者都不是新的通过实验。固定对齐取自各次运行最终 DROID 的 24 个锚点，与各阶段独立对齐的 ATE 分开计分。
'''
(F/'README_CPU.md').write_text(readme,encoding='utf-8')
print(json.dumps({'status':'PASS','independent_matrix_error':maxmatrix,'anchor_matrix_error':maxanchor,'snapshot_files':len(records),'report':str(F/'T18_CPU离线补全诊断与下一步.md')},ensure_ascii=False))
