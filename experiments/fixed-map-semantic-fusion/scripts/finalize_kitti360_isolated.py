"""CPU evidence audit, report and complete independent-owner archive."""
import json,hashlib,tarfile,zipfile,io
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,factor_weights
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/kitti360_isolated_v1';B=R/'03_方法冻结与隔离评价/kitti360_isolated_v1';E=B/'GT_final_evaluation';GROUPS=['S0','S1','S-G','S2','S2-H']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 man=json.loads((D/'manifest.json').read_text());rows=json.loads((E/'GT_main_results.json').read_text());checks=json.loads((E/'fixed_comparison_checks.json').read_text());summary=json.loads((E/'equal_drive_descriptive_summary.json').read_text());audits=[];cpu=[]
 sources=json.loads((R/'01_语义表示_初始化与渲染审计/native_audit_20261007_v1/source_manifest.json').read_text())[:4]
 sources+=json.loads((R/'03_方法冻结与隔离评价/gt_dev_reused_sources_supplement_20261007/provenance.json').read_text())['sources']
 snapshots=B/'reused_source_snapshots';snapshots.mkdir(exist_ok=True);source_receipts=[]
 for i,r in enumerate(sources):
  if 'source' not in r:continue
  assert sha(r['source'])==r['sha256'];p=snapshots/(str(i)+'_'+Path(r['source']).name);p.write_bytes(Path(r['source']).read_bytes());source_receipts.append({'source':r['source'],'sha256':sha(p),'snapshot':str(p.relative_to(R))})
 dump(B/'reused_source_manifest.json',source_receipts)
 for s in man['sequences']:
  d=Path(s['path']);run=B/s['name'];n=0
  generator=json.loads((d/'prediction/generation_summary.json').read_text());checkpoint=generator['checkpoint'];assert sha(checkpoint['path'])==checkpoint['sha256']=='4a3c78ef62be0b5d777d2387965e079db9f7cd0448d59e038b5991f135bf7dd5'
  for rec in s['records']:
   assert all(sha(v['path'])==v['sha256'] for v in rec['files'].values())
   if rec['local_frame'] in s['evaluation']:assert set(rec['files'])=={'gt_labels','gt_instances'}
   else:assert set(rec['files'])=={'image_2','image_3','depth_sgbm'}
  with np.load(d/'base_map.npz') as p:ids=p['source_gaussian_index'].copy();poses=np.load(d/'c2w.npy');assert np.allclose(p['gt_w2c_all_frames'],np.linalg.inv(poses),atol=1e-4);n=len(ids)
  assert json.loads((d/'base_map_lock.json').read_text())['geometry_frames']==s['fusion']
  for g in GROUPS:
   original=Evidence.load(run/'states'/f'{g}_final.npz',ids);state=Evidence(ids,19)
   for f in s['fusion']:
    rec=s['records'][f]
    with np.load(run/'observations'/f'{f:06d}_prediction.npz') as p:
     index=p['gaussian_index'];distribution=np.zeros((n,19),np.float32);distribution[index]=p['distribution'];weights=np.zeros(n,np.float64);weights[index]=factor_weights(g,np.ones(len(index),bool),p['semantic_reliability'],p['geometry_reliability'],.5);version=str(p['input_version'])
     assert np.array_equal(ids[index],p['gaussian_id'])
    obs={'dataset':'KITTI360','sequence':s['name'],'camera':'image_00','original_frame':rec['original_frame']};state.update_frame(obs,version,distribution,weights)
   assert all(np.array_equal(getattr(state,k),getattr(original,k)) for k in ['evidence','support','total_weight']) and state.processed==original.processed;cpu.append({'sequence':s['name'],'group':g,'CPU_replay_bitwise_identical':True})
  audits.append({'sequence':s['name'],'input_hashes':True,'evaluation_RGB_depth_excluded':True,'pose_and_gaussian_identity':True,'geometry_frames':s['fusion'],'gaussians':n})
 dump(B/'postrun_input_audit.json',audits);dump(B/'CPU_independent_fusion_replay.json',cpu)
 lines=['# KITTI-360 隔离评价补齐报告（2026-10-07）','','**两个新驾驶序列小窗口的隔离评价已执行；是否改善与是否完成评价分开记录。DEV仍保留S0，不在最终窗口重新选方法。**','','## 数据与冻结边界','','按元数据的最早可用26帧选取 drive0000 250–275、drive0002 4391–4416。每窗口21帧仅用于双目SGBM、RGB建图、Mask2Former和语义融合；5个排除视角只使用外部相机位姿与官方语义/实例GT，未下载或读取其RGB/深度。每图先固定唯一基图及Gaussian身份，五组清空语义再融合。','','预声明SHA256：`'+man['protocol_sha256']+'`。首次GT像素读取前已冻结帧、方法、深度容差、unknown/no-map、排序同分处理和判据；GPU执行前再封存源码哈希。输入适配修正过一次ZIP成员路径前缀，发生在GT像素读取前，未改变选窗或指标。','','这是本项目新隔离的官方公开train/val小窗口，不是官方隐藏Test成绩。已有记录中未找到这套2013数据的历史使用；本地一个旧审计目录无法读取，不能保证全生命周期/模型预训练从未曝光。官方pose由整条轨迹优化得到，包含更广泛源数据的先验；本实验只评价固定外部pose条件下的地图语义，不评价轨迹误差。','','官方入口：[下载](https://www.cvlibs.net/datasets/kitti-360/download.php)；[数据/相机/位姿定义](https://www.cvlibs.net/datasets/kitti-360/documentation.php)。分段下载逐响应验证206、Content-Range、ETag，逐成员CRC及本地SHA256核验，未下载128GB图像整包。','','## 主结果','','主指标是地图渲染的二维语义mIoU，使用预冻结的10个共通类中本窗口GT存在的类别。完整City19的GT存在类别另报次指标；GT缺失类别列NOT_EVALUABLE。unknown和无地图支持在有效GT上均计真实类别FN；全有效GT为覆盖分母，不能靠拒判缩小主指标分母。','','| 窗口 | 组别 | 主mIoU | 标签覆盖 | unknown | 无地图支持 |','|---|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['sequence'].split('_drive_')[1].split('_sync')[0]} | {r['group']} | {100*r['mIoU_GT_supported_common_classes']:.3f}% | {100*r['coverage']:.3f}% | {100*r['unknown_fraction']:.3f}% | {100*r['no_map_fraction']:.3f}% |")
 lines+=['','## 固定比较判据结果','']
 for c in checks:lines.append(f"- {c['group']}：两窗口等权mIoU差 {100*c['equal_drive_mIoU_difference']:+.3f} pp；{c['confirmation_status']}。失败项：{'; '.join(c['failed_gates']) or '无'}。缺少支持：{'; '.join(c['missing_required_support']) or '无'}。")
 lines+=['','S0在此前DEV已被保留，不能将最终评价中的任何偶然改善用于重新选择、调整阈值或回补DEV成功。这里不签发整体算法科学PASS，保留原DEV负结果、原KITTI08无独立GT及Stage2正式FAIL。','','## 已补齐的评价与核验','','- 10个评价视角×五组共50份原生语义渲染；逐组Gaussian/参数保持一致，RGB比较容差1e-6，实际差值见RGB_equality.json。unknown沿用原生遮挡。','- 全有效GT主/次mIoU、按类TP/FP/FN、coverage、条件准确率、unknown/no-map比例。','- 风险—覆盖完整分数并列曲线、共同可达覆盖比较；分数为类别质量/原生alpha，未经概率校准；无地图支持和unknown不补假标签。','- 共同GT小车辆/非小车辆、边界/内部、行人描述子组，掩码哈希锁定；缺少实例或像素支持时明确不可验，不另挑有利帧。','- 42帧观测CPU重新融合，十个序列×方法状态逐值一致；原始输入哈希、相机方向、Gaussian对应和评价RGB/深度排除审计。','- 完整原始小窗口输入、缓存、状态、基图、日志、配置、评价、曲线、源码与哈希归档；本地下载后执行独立GT布尔复算。','','## 仍不可给出的结论','','本轮没有独立动态状态/GT深度对应，稳定表面与动态/遮挡变化区的严格分层仍NOT_EVALUABLE；车辆子组不能冒充动态GT。主评价保留全部有效GT，是固定静态Gaussian地图对各评价时刻语义快照的结果。车辆移动产生的历史几何可能造成误差。只覆盖两个小轨迹窗口，不以大量像素宣称大量独立样本，无全驾驶场景泛化或逐Gaussian三维语义准确率声明。原KITTI08真实GT评价仍缺，不能用这里替代原数据质量证明。','','## 运行和结果入口','','服务器根目录：`/root/autodl-tmp/stage3_chy_20261006`。一次性流水线：`/root/miniconda3/bin/python scripts/run_kitti360_pipeline.py`（已经执行，现有输出保护，不应在同一路径重跑）。','','核心结果：`GT_final_evaluation/GT_main_results.json`、`GT_secondary_City19_results.json`、`GT_subgroup_results.json`、`GT_common_risk_coverage.json`、`fixed_comparison_checks.json`及`GT_support_and_unevaluable_groups.json`。最终判定：评价链完成，科学质量收益以固定比较结果和适用范围为准，不触发MLP、S2-Z/S2-V或Stage2重开。','']
 (B/'KITTI360隔离评价完成报告_20261007.md').write_text('\n'.join(lines),encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for s in man['sequences']:
  values=[next(r for r in rows if r['sequence']==s['name'] and r['group']==g) for g in GROUPS];label=s['name'].split('_drive_')[1].split('_sync')[0]
  axes[0].plot(GROUPS,[100*r['mIoU_GT_supported_common_classes'] for r in values],'-o',label='drive'+label);axes[1].plot(GROUPS,[100*r['coverage'] for r in values],'-o',label='drive'+label)
 axes[0].set_ylabel('GT-supported common-class mIoU (%)');axes[1].set_ylabel('Label coverage / valid GT (%)')
 for ax in axes:ax.grid(alpha=.25);ax.legend();ax.set_ylim(bottom=0)
 fig.suptitle('KITTI-360 project holdout; fixed methods, 5 views per drive');fig.tight_layout();fig.savefig(B/'holdout_quality_coverage.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,s in zip(axes,man['sequences']):
  for g in GROUPS:
   with np.load(E/f"{s['name']}_{g}_risk_curve.npz") as p:
    idx=np.unique(np.r_[np.linspace(0,len(p['risk'])-1,min(2000,len(p['risk']))).astype(int),len(p['risk'])-1]);ax.plot(100*p['coverage'][idx],100*p['risk'][idx],label=g,alpha=.85)
  ax.set_title(s['name'].split('_drive_')[1].split('_sync')[0]);ax.set_xlabel('Coverage / valid GT (%)');ax.set_ylabel('Error among accepted labels (%)');ax.grid(alpha=.25);ax.legend()
 fig.suptitle('Exact tied-threshold risk curves; plotting subsampled only');fig.tight_layout();fig.savefig(B/'holdout_risk_coverage.png',dpi=180);plt.close(fig)
 paths=[p for folder in [D,B] for p in folder.rglob('*') if p.is_file()]
 paths=sorted(set(paths));paths=[p for p in paths if not p.name.endswith('.tar.gz') and p.name not in ['artifact_SHA256.json','package_result.json']]
 for n in json.loads((B/'source_freeze.json').read_text())['files']:paths.append(R/n['path'])
 paths+=[R/'scripts/audit_kitti360_entry.py',R/'scripts/finalize_kitti360_isolated.py',R/'protocol/KITTI360隔离评价预声明_20261007_v1.json'];paths=sorted(set(paths))
 manifest={str(p.relative_to(R)):sha(p) for p in paths};dump(B/'artifact_SHA256.json',manifest);archive=B.parent/'kitti360_isolated_v1_complete_20261007.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for p in paths+[B/'artifact_SHA256.json']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 result={'status':'COMPLETE_REMOTE_AUDIT_AND_ARCHIVE','files':len(paths)+1,'bytes':archive.stat().st_size,'sha256':sha(archive),'archive':str(archive),'local_recount':'PENDING_DOWNLOAD'};dump(B/'package_result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
