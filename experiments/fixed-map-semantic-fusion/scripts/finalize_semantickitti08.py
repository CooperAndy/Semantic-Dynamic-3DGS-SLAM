"""Read-only final audit, evidence report and self-contained new-data archive."""
import hashlib,json,tarfile
import numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/semantickitti08_v1';B=R/'02_基础因子DEV与Oracle诊断/semantickitti08_gt_v1';A=R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert json.loads((B/'pipeline_status.json').read_text())['status']=='COMPLETE_REAL_SPARSE_GT_DEV_ALL200'
 frozen=json.loads((B/'source_freeze.json').read_text());assert all(sha(R/k)==v for k,v in frozen['files'].items());assert sha(A/'correspondence_audit.json')==frozen['small_audit_sha256']
 man=json.loads((D/'evaluation_input_manifest.json').read_text());assert sha(R/'inputs/base_map/params.npz')==man['base_sha256'];old=R/'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1/states';assert all(sha(old/(g+'_final.npz'))==v for g,v in man['state_sha256'].items())
 rows=json.loads((B/'GT_evaluation/GT_results.json').read_text());support=json.loads((B/'GT_evaluation/GT_support.json').read_text());diag=json.loads((B/'GT_evaluation/input_vs_map_diagnostics.json').read_text());audit=json.loads((A/'correspondence_audit.json').read_text())
 def row(g,k='all200'):return next(r for r in rows if r['group']==g and r['subgroup']==k)
 def pct(v):return '不可评价' if v is None else f'{100*v:.3f}%'
 lines=['# 原 KITTI08 SemanticKITTI 稀疏真实语义评价（2026-10-07）','','小窗口对应核验通过，原始帧 3000–3199 的 200 帧、五组固定语义地图和 1000 次原生渲染已完成真实标签评价。本窗口仍为已曝光 DEV；此补充不重新选方法、不调整权重、不改名 Test，也不宣称科学 PASS。','','## 先核验，再全量','','先核验 3000–3009 共 10 对点云/标签：帧号与 RGB 哈希一致，每帧 float32 xyzi 点数等于 uint32 标签数；低16位类别、高16位实例编号。官方完整 P2/Tr、原 P2 和基图内参一致；投影与独立 OpenCV 路径最大差为 '+str(max(x['projection_opencv_max_abs_px_error'] for x in audit['frames']))+' 像素。Codex 检视 3000、3004、3009 叠图，车辆、道路、建筑总体对应，没有发现翻转或错帧。运动骑行者、稀疏采样和跨传感器遮挡仍有局部差异。','','受控测试覆盖 P2 平移、标签打包、同像素最近深度、unknown 前景阻止背景标签透出和移动类别映射。所有标签（包括 ignore）先参与最近点选择，再映射/排除 ignore；不膨胀或插值 GT。','','## 评价对象与分母','','指标名称为“地图渲染在投影 LiDAR 稀疏 GT 位置的二维语义 mIoU”。SemanticKITTI 是逐 LiDAR 点标注，不能称为整图稠密 GT 或逐 Gaussian 三维准确率。本报告使用冻结的 raw→City19 映射及 16 个候选共通类；无 GT 支持的类别单列不可评价，不是标准 SemanticKITTI 19类排行榜协议。other-vehicle、parking、other-ground 等含糊映射类别忽略；映射详见预声明 JSON。','','有效 GT 总数 '+str(sum(x['mapped_GT_pixels'] for x in support))+'；占 200 张完整图像像素的 '+pct(sum(x['mapped_GT_pixels'] for x in support)/(200*man['image_height']*man['image_width']))+'，各帧范围 '+pct(min(x['mapped_GT_image_fraction'] for x in support))+'–'+pct(max(x['mapped_GT_image_fraction'] for x in support))+'。本分母覆盖全部映射有效 GT，包括方法 unknown 和无地图支持（均作真实类漏判）；ignore 才排除。GT 仅选择保存/评分的位置，没有融合进原地图。','','## 全200帧结果','','|组别|稀疏 GT mIoU|标签覆盖|有标签条件准确率|unknown|无地图支持|','|---|---:|---:|---:|---:|---:|']
 for g in ['S0','S1','S-G','S2','S2-H','2D_input_teacher']:
  r=row(g);lines.append('|'+g+'|'+ '|'.join(pct(r[k]) for k in ['mIoU_GT_supported_common_classes','coverage','conditional_accuracy','unknown_fraction','no_map_fraction'])+'|')
 s0=row('S0');s2=row('S2');sg=row('S-G');hard=row('S2-H');lines+=['',f"本窗口 S2 相对 S0 的 mIoU 提高 {(s2['mIoU_GT_supported_common_classes']-s0['mIoU_GT_supported_common_classes'])*100:.3f} 个百分点；S-G 提高 {(sg['mIoU_GT_supported_common_classes']-s0['mIoU_GT_supported_common_classes'])*100:.3f} 个百分点，S2 比 S-G 仅高 {(s2['mIoU_GT_supported_common_classes']-sg['mIoU_GT_supported_common_classes'])*100:.3f} 个百分点。几何可靠性在这份原地图上有明确的描述性收益，不能据此推翻其他数据窗口的冻结评价。S2-H 覆盖比 S2 低 {(s2['coverage']-hard['coverage'])*100:.3f} 个百分点，mIoU 也低 {(s2['mIoU_GT_supported_common_classes']-hard['mIoU_GT_supported_common_classes'])*100:.3f} 个百分点，提示硬拒绝的代价。"]
 lines+=['','2D_input_teacher 是原当前帧二维预测的诊断对照，不是独立地图方法。上述绝对数值不可直接与 VKITTI2/KITTI-360 稠密 GT mIoU 横比。','','## 当前融合与排除视角分开解释','','|组别|180融合帧 mIoU|20排除语义融合帧 mIoU|移动原始类别 mIoU|非移动原始类别 mIoU|','|---|---:|---:|---:|---:|']
 for g in ['S0','S1','S-G','S2','S2-H']:
  lines.append('|'+g+'|'+'|'.join(pct(row(g,k)['mIoU_GT_supported_common_classes']) for k in ['fusion180','excluded20','GT_moving','GT_nonmoving'])+'|')
 lines+=['',f"需特别保留负面结果：移动类别只有 {row('S2','GT_moving')['GT_valid_pixels']} 个有效采样，S2 条件准确率 {pct(row('S2','GT_moving')['conditional_accuracy'])}，当前二维预测也仅 {pct(row('2D_input_teacher','GT_moving')['conditional_accuracy'])}。静态地图不能表达所有时刻的移动表面，输入语义和跨传感器运动投影也可能有错；这些数字不能解释为可靠动态物体建图。另一个方向是地图融合损失：当前二维预测全200帧 mIoU {pct(row('2D_input_teacher')['mIoU_GT_supported_common_classes'])}，高于所有固定地图组，需要结合下表继续定位，不能只提升二维预测就宣称问题解决。"]
 lines+=['','180/20 仅区分本轮语义融合角色；基图几何和历史语义已经见过全部200帧，清空语义没有消除历史曝光。移动/非移动依据官方原始类别，非移动不等于经时序验证的长期稳定表面。小车辆仅为可见稀疏点包围框代理，至少5点且面积≤图像0.25%；没有足够点的对象不宣称通过小目标验证。稀疏标注不足以可靠复现稠密边界/内部或严格时序遮挡分组，不能用插值标签假装补齐。','','## 错误来源诊断','','|组别|二维预测错误但地图正确|二维预测正确但地图已知类别错误|地图 unknown|地图无支持|已知错误中深度一致|已知错误中深度不一致|','|---|---:|---:|---:|---:|---:|---:|']
 for g in ['S0','S1','S-G','S2','S2-H']:
  rr=[r for r in diag if r['group']==g];keys=['teacher_wrong_map_correct','teacher_correct_map_wrong_known','map_unknown','map_no_map','depth_agrees_known_wrong','depth_disagrees_known_wrong'];lines.append('|'+g+'|'+'|'.join(str(sum(x[k] for x in rr)) for k in keys)+'|')
 lines+=['','当前帧二维预测正确而地图错误，提示融合/关联/几何/时序等问题值得查，但不能唯一归因；地图正确而当前二维错误，是多视角融合可能纠错的证据。深度一致诊断使用冻结阈值 |z_map−z_LiDAR|≤1m+0.02z，不从主指标删掉不一致点。原生 alpha 加权均值深度有前后景混合的局限，LiDAR/相机视差与扫描时间也会产生残差。','','风险排序为最终最大类别证据/原生总 alpha；完整同分组一起接受，所有方法共同可达覆盖水平评分，unknown/no-map 不靠虚构类别提高覆盖。完整曲线见 GT_evaluation/*_risk_curve.npz。像素和连续帧高度相关，不作像素独立置信区间。','','## 冻结与可复现性','','唯一基图907,488个Gaussian，SHA256 `'+man['base_sha256']+'`。五组沿用原 SGBM 版本最终状态，没有新融合/训练；旧20视角×五组100份完整渲染逐位复现，1000份新稀疏采样渲染共用 alpha 和地图深度。源码、协议、GT、输入及状态哈希已核验。原地图依赖仍在旧 step02 完整归档中；本补充归档包含新增 raw 点云、标签、投影GT、渲染、评价、源码、日志和协议。','','P2 平移完整保留。基图采用既有相机表示，若其将 image_2 近似为参考相机，约6.1cm横向基线会构成关联误差来源；本次不改动几何/相机来优化指标。SemanticKITTI SuMa 估计位姿没有替换当前地图位姿，也不称轨迹 GT。','','当前补充用于解释原 DEV 地图，不覆盖此前方法判优结果；保留已冻结 S0 参考方法。稀疏 GT 不足以原样做整图正确语义输入 O-Sem，故没有为了它改变原观测支持。','','## 数据出处','','[SemanticKITTI 官方数据说明](https://semantic-kitti.org/dataset.html)，Behley et al., ICCV 2019；[官方类别配置](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml)。原图像/点云/标定来自 KITTI Odometry，Geiger et al., CVPR 2012。标签许可按官方 CC BY-NC-SA 说明用于本科研诊断；没有公开重新发布原始数据。官方压缩包 URL、ETag、逐成员 CRC32 和 SHA256 见 inputs/semantickitti08_v1/*download_manifest.json。']
 lines += ['', '## 可查看的叠图', '', '小窗口叠图在00审计目录；以下诊断图依次为RGB、当前帧二维预测错误、S0地图错误，显示用点半径不参与评价。', '', '![3004对照](diagnostic_overlays/003004_teacher_vs_S0_sparse_GT_errors.png)', '', '![3199对照](diagnostic_overlays/003199_teacher_vs_S0_sparse_GT_errors.png)']
 (B/'KITTI08_SemanticKITTI真实语义评价报告_20261007.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 visual=B/'diagnostic_overlays';visual.mkdir()
 for f in [3004,3099,3199]:
  rec=next(x for x in man['records'] if x['original_frame']==f);rgb=Image.open(rec['input_locks']['image_2']['path']).convert('RGB')
  with np.load(D/'sparse_gt'/f'{f:06d}.npz') as gt:pi=np.flatnonzero(gt['point_index'].ravel()>=0);label=gt['label'].ravel()[pi]
  teacher=np.load(rec['input_locks']['labels']['path']).ravel()[pi]
  with np.load(B/'renders/S0'/f'{f:06d}.npz') as pred:maplabel=pred['label']
  canvas=Image.new('RGB',(rgb.width,rgb.height*3));canvas.paste(rgb,(0,0));draw=ImageDraw.Draw(canvas);draw.text((5,5),f'KITTI08 {f}: RGB / teacher errors / S0 map errors',fill='white')
  for block,prediction in enumerate([teacher,maplabel],1):
   layer=rgb.copy();d=ImageDraw.Draw(layer)
   for pixel,true,predicted in zip(pi,label,prediction):
    if true==255:continue
    x=int(pixel%rgb.width);y=int(pixel//rgb.width);color=(0,255,0) if true==predicted else (255,0,0) if predicted>=0 else (255,180,0) if predicted==-1 else (0,255,255);d.ellipse((x-1,y-1,x+1,y+1),fill=color)
   d.text((5,5),'green=correct red=wrong orange=unknown cyan=no map; display circles only',fill='white');canvas.paste(layer,(0,block*rgb.height))
  canvas.save(visual/f'{f:06d}_teacher_vs_S0_sparse_GT_errors.png')
 dump(B/'final_integrity.json',{'status':'PASS_FIXED_INPUTS_STATES_AND_CODE','base_sha256':man['base_sha256'],'state_sha256':man['state_sha256'],'source_freeze_unchanged':True,'small_audit_unchanged':True,'new_Test':False})
 scripts=set(frozen['files'])|{'scripts/finalize_semantickitti08.py','scripts/test_semantickitti_projection.py','scripts/freeze_semantickitti08_protocol.py','scripts/verify_semantickitti08_local.py','scripts/prefetch_semantickitti08_labels.py'}
 files=set(p for folder in [D,B,A] for p in folder.rglob('*') if p.is_file());files|={R/n for n in scripts};files.add(R/'protocol/SemanticKITTI08稀疏GT评价预声明_20261007_v1.json');files|={Path(rec['input_locks']['labels']['path']) for rec in man['records']};assert all(p.is_file() for p in files)
 dump(B/'evidence_file_hashes.json',{str(p.relative_to(R)):sha(p) for p in sorted(files)});files.add(B/'evidence_file_hashes.json')
 archive=R/'02_基础因子DEV与Oracle诊断/semantickitti08_sparse_GT_complete_20261007_v1.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as tar:
  for p in sorted(files):tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
 receipt={'status':'COMPLETE','path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)};dump(B/'archive_receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
