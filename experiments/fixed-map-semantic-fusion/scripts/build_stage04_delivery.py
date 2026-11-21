"""CPU-only integration of frozen evidence; no fusion, GT scoring or tuning."""
import csv,hashlib,json,platform,shutil,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'04_系统整合与最终归档/final_delivery_20261007_v1';GROUPS=['S0','S1','S-G','S2','S2-H'];C=np.array([[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]],np.uint8)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
 return h.hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def relative(p):return p.relative_to(R).as_posix()
def color(label):
 out=np.full((*label.shape,3),90,np.uint8);valid=(label>=0)&(label<19);out[valid]=C[label[valid]];out[(label==-2)|(label==255)]=0;return out
def image(p,arr):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(p)
def main():
 assert not O.exists(),'keep sealed deliveries; create a new version';O.mkdir(parents=True);start=time.time();A=O/'assets';A.mkdir();I=R/'02_基础因子DEV与Oracle诊断';K=R/'03_方法冻结与隔离评价/kitti360_isolated_v1';V=I/'gt_dev_v2';S=I/'semantickitti08_gt_v1';records=[];cases=[];maps=[]
 checks=[];deps=[]
 archives=[(I/'step02_native_dev_complete_20261007_v1.tar.gz','cc5818907c64374ea56e1d0f02191574d644a6dd2d7dce9951b5b105008e71fd'),(I/'vkitti_gt_dev_v2_complete_20261007.tar.gz','efdc40f5146d39f499a198dd4d5950642c32a3e8419abc19a593200b03e4072d'),(K.parent/'kitti360_isolated_v1_complete_20261007.tar.gz','8021bc0a73cba02b277b45455d308c68632e6086757bef0cc98e2d7e44ccf1ea'),(I/'semantickitti08_sparse_GT_complete_20261007_v1.tar.gz','21d3e01460428496bfaa2070937eb8084cdfe0d044f5b6f6dfc40ce3a2dbf5e6')]
 for p,digest in archives:assert sha(p)==digest;deps.append({'path':relative(p),'sha256':digest,'bytes':p.stat().st_size,'available_local':True})
 proofs=[S/'local_independent_verification.json',V/'local_integrity_and_GT_recount.json',K/'local_integrity_and_independent_recount.json']
 for p in proofs:assert read(p)['status'].startswith('PASS_');checks.append({'check':'prior independent verification','path':relative(p),'sha256':sha(p),'result':'PASS'})
 assert read(V/'GT_evaluation_v1/DEV_candidate_decision.json')['selected_DEV_reference']=='S0';assert read(S/'pipeline_status.json')['status']=='COMPLETE_REAL_SPARSE_GT_DEV_ALL200';checks.append({'check':'frozen S0 reference and complete prior evaluation','result':'PASS'})
 specs=[{'id':'KITTI08','name':'KITTI08 · 已曝光 DEV / 稀疏 GT','base':R/'inputs/base_map/params.npz','run':I/'dev_sgbm_native_full_20261007_v1','data':R/'inputs/semantickitti08_v1','frames':[9,99,199],'original':3000,'kind':'sparse','scope':'历史几何与语义见过全部200帧；180/20是本轮语义融合角色，均为DEV。稀疏LiDAR投影GT占整图4.187%。'},]
 for name in ['Scene01_clone_330_355','Scene02_clone_110_135']:specs.append({'id':name,'name':name+' · 已曝光 DEV / 稠密 GT','base':R/'inputs/vkitti2_gt_dev_v2'/name/'base_map.npz','run':V/(name+'_full'),'data':R/'inputs/vkitti2_gt_dev_v2'/name,'original':330 if 'Scene01' in name else 110,'kind':'dense','scope':'历史DEV；几何见26帧，21帧语义融合、5帧排除评价。O-Sem仅为DEV定位诊断。'})
 for name in ['2013_05_28_drive_0000_sync','2013_05_28_drive_0002_sync']:specs.append({'id':name,'name':name.split('_drive_')[1].split('_')[0]+' · KITTI-360项目隔离窗口','base':R/'inputs/kitti360_isolated_v1'/name/'base_map.npz','run':K/name,'data':R/'inputs/kitti360_isolated_v1'/name,'original':250 if '0000' in name else 4391,'kind':'holdout','scope':'公开train/val上的项目新隔离窗口；非官方隐藏Test。评价RGB/深度未取用，官方pose含更广轨迹先验。'})
 mainrows=read(S/'GT_evaluation/GT_results.json')+read(V/'GT_evaluation_v1/GT_main_results.json')+read(K/'GT_final_evaluation/GT_main_results.json')
 for spec in specs:
  key=spec['id'];run=spec['run'];folder=A/key;folder.mkdir();cfg=read(run/'config.json') if spec['kind']!='sparse' else None
  if cfg:evals=cfg['evaluation_frames'];assert len(evals)==5;spec['frames']=[evals[0],evals[len(evals)//2],evals[-1]]
  with np.load(spec['base']) as raw:base={k:raw[k] for k in raw.files}
  ids=base['source_gaussian_index'];N=len(ids);take=np.linspace(0,N-1,min(12000,N),dtype=int);xyz=base['means3D'][take];center=(xyz.min(0)+xyz.max(0))/2;scale=max(float(np.ptp(xyz,axis=0).max())/2,1e-6);points={'xyz':np.round((xyz-center)/scale,6).tolist(),'rgb':np.clip(np.rint(base['rgb_colors'][take]*255),0,255).astype(int).tolist(),'ids':ids[take].astype(int).tolist(),'groups':{},'count':N,'sample_count':len(take),'view':'sampled Gaussian centers for attribute inspection, not native rendering or a metric','base_sha256':sha(spec['base'])};stateinfo=[]
  for g in GROUPS:
   sp=run/'states'/(g+'_final.npz')
   with np.load(sp) as st:assert np.array_equal(st['gaussian_id'],ids);ev=st['evidence'];total=ev.sum(1);lab=np.where(total>0,ev.argmax(1),-1).astype(np.int16);conf=np.divide(ev.max(1),total,out=np.zeros(N,np.float32),where=total>0);sup=st['support_count'];weight=st['total_weight'];points['groups'][g]={'label':lab[take].tolist(),'support':sup[take].astype(int).tolist(),'confidence':np.round(conf[take],6).tolist()};stateinfo.append({'group':g,'path':relative(sp),'sha256':sha(sp),'Gaussian_ID_correspondence':True,'labelled_gaussians':int((lab>=0).sum()),'Gaussian_count':N})
   if key=='KITTI08' and g=='S0':export_ply(O/'exports/KITTI08_S0_semantic_gaussians.ply',base,ids,lab,conf,sup,weight)
  dump(folder/'map_points.json',points);maps.append({'id':key,'name':spec['name'],'scope':spec['scope'],'base':relative(spec['base']),'base_sha256':sha(spec['base']),'Gaussian_count':N,'states':stateinfo,'points':'assets/'+key+'/map_points.json'});checks.append({'check':'five full state IDs correspond to immutable geometry','dataset':key,'result':'PASS'})
  for f in spec['frames']:
   original=f+spec['original'];case={'id':key+'_'+str(original),'dataset':key,'name':str(original),'scope':spec['scope'],'groups':{},'image_kind':'native numeric semantic render','GT_kind':'sparse projected LiDAR' if spec['kind']=='sparse' else 'dense official GT'}
   if spec['kind']=='sparse':
    with np.load(spec['data']/'sparse_gt'/f'{original:06d}.npz') as gt:gtlabel=gt['label'];valid=gtlabel!=255
    rgbfile=R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1/003009_RGB_semantic_projection.png' if f==9 else S/'diagnostic_overlays'/f'{original:06d}_teacher_vs_S0_sparse_GT_errors.png';rgb=np.asarray(Image.open(rgbfile))[:gtlabel.shape[0]];case['RGB_note']='RGB参考取自已归档核验/诊断图上半幅；3099/3199保留原图显示标题。'
   else:
    gtlabel=np.load(spec['data']/'gt_labels'/f'{f:06d}.npy');valid=gtlabel!=255;rgbfile=spec['data']/'image_2'/f'{f:06d}.png';rgb=None if spec['kind']=='holdout' else np.asarray(Image.open(rgbfile)) if rgbfile.exists() else None;case['RGB_note']='原评价RGB未读取；保持隔离，此处不补下载。' if rgb is None else '已曝光DEV源RGB。'
   prefix=key+'/'+str(original);image(A/(prefix+'_GT.png'),color(gtlabel));case['GT']='assets/'+prefix+'_GT.png'
   if rgb is not None:image(A/(prefix+'_RGB.png'),rgb);case['RGB']='assets/'+prefix+'_RGB.png'
   groups=GROUPS+(['O-Sem'] if spec['kind']=='dense' else [])
   for g in groups:
    file=run/'renders'/g/f'{f:06d}.npz'
    with np.load(file) as p:pred=p['label'];score=p['score'];alpha=p['alpha'];assert pred.shape==gtlabel.shape and np.array_equal(pred==-2,alpha<.05)
    image(A/(prefix+'_'+g+'_semantic.png'),color(pred));image(A/(prefix+'_'+g+'_score.png'),np.clip(score*255,0,255).astype(np.uint8));err=np.full((*pred.shape,3),45,np.uint8);err[valid&(pred==gtlabel)]=[0,190,95];err[valid&(pred>=0)&(pred!=gtlabel)]=[235,70,70];err[valid&(pred==-1)]=[255,180,0];err[valid&(pred==-2)]=[0,210,240];image(A/(prefix+'_'+g+'_error.png'),err)
    case['groups'][g]={'semantic':'assets/'+prefix+'_'+g+'_semantic.png','score':'assets/'+prefix+'_'+g+'_score.png','error':'assets/'+prefix+'_'+g+'_error.png','source':relative(file),'source_sha256':sha(file),'valid_GT_pixels':int(valid.sum()),'native_no_map_pixels':int((valid&(pred==-2)).sum())}
   cases.append(case)
  if key=='KITTI08':rs=[v for v in mainrows if v.get('subgroup')=='all200' and v['group'] in GROUPS]
  else:rs=[v for v in mainrows if v.get('sequence')==key and v['group'] in GROUPS]
  for v in rs:records.append({'dataset':key,'scope':spec['scope'],'group':v['group'],'mIoU':v['mIoU_GT_supported_common_classes'],'coverage':v['coverage'],'unknown_fraction':v['unknown_fraction'],'no_map_fraction':v['no_map_fraction'],'GT_valid_pixels':v['GT_valid_pixels']})
 dump(O/'prerequisites.json',{'status':'PASS_04_PREREQUISITES_WITH_DECLARED_QUALITY_LIMITS','checks':checks,'dependency_archives':deps,'history_and_exposure_preserved':True,'GT_sparse_and_dense_not_pooled':True,'no_05_or_06_dependency':True});dump(O/'asset_provenance.json',{'cases':cases,'selection':'first/middle/last existing frozen evaluation frames, before reading per-frame scores; no favorable-frame search','maps':maps,'source_metric_files':[{'path':relative(p),'sha256':sha(p)} for p in [S/'GT_evaluation/GT_results.json',V/'GT_evaluation_v1/GT_main_results.json',K/'GT_final_evaluation/GT_main_results.json']],'metric_rows':records});dump(O/'bundle_dependencies.json',deps)
 dump(O/'viewer_data.json',{'cases':cases,'maps':maps,'metrics':records,'palette':C.tolist(),'groups':GROUPS,'default_method':'S0','quality_acceptance':False})
 with (O/'metrics_by_window.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 graph(O,records);report(O,records,maps,deps);shutil.copy2(Path(__file__),O/'source_build_stage04_delivery.py');dump(O/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'wall_seconds':time.time()-start,'GPU_used':False,'no_new_inference_fusion_or_tuning':True});dump(O/'build_result.json',{'status':'COMPLETE_CPU_INTEGRATION_PENDING_VIEWER_QA_AND_ARCHIVE','cases':len(cases),'native_method_case_comparisons':sum(len(c['groups']) for c in cases),'map_summaries':len(maps),'full_KITTI08_export_Gaussians':907488,'default_method':'S0','scientific_quality_PASS':False});print(json.dumps({'status':'BUILT','cases':len(cases),'output':str(O)}),flush=True)
def export_ply(p,b,ids,lab,conf,sup,weight):
 p.parent.mkdir();names=[('x','f4'),('y','f4'),('z','f4')]+[(f'f_dc_{i}','f4') for i in range(3)]+[('opacity','f4')]+[(f'scale_{i}','f4') for i in range(3)]+[(f'rot_{i}','f4') for i in range(4)]+[('gaussian_id','u4'),('semantic_label','i2'),('semantic_confidence','f4'),('support_count','u4'),('total_weight','f8')];out=np.empty(len(ids),dtype=np.dtype(names))
 for i,k in enumerate(['x','y','z']):out[k]=b['means3D'][:,i]
 for i in range(3):out[f'f_dc_{i}']=(b['rgb_colors'][:,i]-.5)/.28209479177387814;out[f'scale_{i}']=b['log_scales'][:,0 if b['log_scales'].shape[1]==1 else i]
 for i in range(4):out[f'rot_{i}']=b['unnorm_rotations'][:,i]
 out['opacity']=b['logit_opacities'].ravel();out['gaussian_id']=ids;out['semantic_label']=lab;out['semantic_confidence']=conf;out['support_count']=sup;out['total_weight']=weight;type_map={'f4':'float','f8':'double','u4':'uint','i2':'short'};head='ply\nformat binary_little_endian 1.0\ncomment CHY Stage3 frozen S0; semantic confidence uncalibrated\nelement vertex '+str(len(ids))+'\n'+'\n'.join('property '+type_map[t]+' '+n for n,t in names)+'\nend_header\n'
 with p.open('wb') as f:f.write(head.encode());f.write(out.tobytes())
 # Independent payload reload: stable IDs, geometry, opacity, scales and rotations.
 raw=np.fromfile(p,dtype=out.dtype,offset=len(head.encode()));assert np.array_equal(raw['gaussian_id'],ids) and np.array_equal(np.column_stack([raw[k] for k in ['x','y','z']]),b['means3D']);assert np.array_equal(raw['opacity'],b['logit_opacities'].ravel());assert np.array_equal(np.column_stack([raw[f'rot_{i}'] for i in range(4)]),b['unnorm_rotations']);assert np.array_equal(raw['semantic_label'],lab) and np.array_equal(raw['support_count'],sup)
 dump(p.with_suffix('.json'),{'status':'PASS_FULL_PLY_RELOAD_ID_AND_ATTRIBUTES','Gaussian_count':len(ids),'sha256':sha(p),'default_method':'frozen S0','RGB_conversion':'SH0 f_dc=(RGB-0.5)/0.28209479177387814','scales':'original isotropic log_scale copied into three axes','opacity':'original logit, not zeroed for unknown','rotations':'original unnormalized quaternion; native renderer normalizes','semantic_unknown':-1,'no_map':'view-dependent, no per-Gaussian no-map class','semantic_confidence':'max accumulated evidence / total, not calibrated probability','field_layout':names,'vertex_bytes':out.dtype.itemsize,'header_bytes':len(head.encode()),'geometry_changed':False})
def graph(o,records):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(14,7));families=[('KITTI08: sparse exposed DEV',[r for r in records if r['dataset']=='KITTI08']),('VKITTI2: dense exposed DEV',[r for r in records if r['dataset'].startswith('Scene')]),('KITTI360: public project holdout',[r for r in records if r['dataset'].startswith('2013')])]
 for col,(name,rows) in enumerate(families):
  mean={g:{k:np.mean([r[k] for r in rows if r['group']==g]) for k in ['coverage','mIoU']} for g in GROUPS}
  for row,key in enumerate(['mIoU','coverage']):
   ax=axes[row,col];values=[(mean[g][key]-mean['S0'][key])*100 for g in GROUPS];bars=ax.bar(GROUPS,values,color=['#276b9a','#dc9444','#34836d','#b85959','#83659c']);ax.axhline(0,color='#546477',lw=.8);ax.margins(y=.3);ax.set_ylabel(('mIoU' if key=='mIoU' else 'Label coverage')+' change (pp)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
   for bar,val in zip(bars,values):ax.annotate(f'{val:+.3f}',(bar.get_x()+bar.get_width()/2,val),xytext=(0,4 if val>=0 else -14),textcoords='offset points',ha='center',fontsize=8)
   if row==0:ax.set_title(name+'\nS0 mIoU '+f"{mean['S0']['mIoU']*100:.3f}%"+', coverage '+f"{mean['S0']['coverage']*100:.3f}%",fontsize=10)
 fig.suptitle('Changes within each frozen protocol; equal-window averages. No cross-dataset ranking.');fig.tight_layout();fig.savefig(o/'quality_coverage_by_protocol.png',dpi=160);plt.close(fig)
def report(o,rows,maps,deps):
 pct=lambda x:f'{x*100:.3f}%';lines=['# 阶段3最终报告：固定地图语义融合与可靠性验证','','负责人CHY；2026-10-07。**工程主线和研究评价交付完成；加权方法的跨窗口质量改善未通过既定判据。默认保留此前冻结的S0。** 04整合已有冻结结果，不运行新推理、不重新融合、不回调阈值；05/06条件扩展不属于04前置。','','## 复用与新增','','复用原固定Gaussian几何、相机/位姿、RGB/opacity、Mask2Former硬标签与标量置信缓存、SplaTAM原生CUDA数值通道渲染。原RAFT旧路径未恢复；当前运行明确使用版本化SGBM预测深度，不能冒充历史RAFT实验。','','本轮新增零语义初始化/稳定ID、源帧去重、帧内alpha质量归一化、S0/S1/S-G/S2/S2-H小型因子对照、原生unknown遮挡与共享RGB/alpha核验、独立GT计分、风险—覆盖/子组、DEV O-Sem、隔离评价与复现封存。多帧语义加权本身不宣称首次提出；本项目贡献是具体驾驶场景固定地图上的受控验证和错误诊断。','','## 入口与结果','','|窗口|方法|主mIoU|标签覆盖|unknown|无地图支持|','|---|---|---:|---:|---:|---:|']
 for r in rows:lines.append('|'+r['dataset']+'|'+r['group']+'|'+'|'.join(pct(r[k]) for k in ['mIoU','coverage','unknown_fraction','no_map_fraction'])+'|')
 lines+=['','KITTI08：历史几何/语义见过全部200帧；180/20只是当前语义融合角色。SemanticKITTI稀疏投影GT有效3798709像素，占整图4.187%，16个冻结候选共通类中的GT存在类计分。不能称稠密整图、逐Gaussian三维准确率或新Test。','','VKITTI2：两个已曝光DEV窗口，几何见26帧，21融合/5评价；官方稠密GT、10个冻结候选共通类。O-Sem输入和评价角色分开且仅为DEV诊断，改善不能当可部署成绩或严格数学上限。','','KITTI-360：两条新的项目隔离公开train/val窗口，21建图/融合、5评价；评价RGB和深度未取用，官方pose包含更广轨迹先验，非官方隐藏Test。只评价固定外部pose条件地图语义，不证明跟踪或全场景泛化。','','三套GT密度、类集及曝光/聚合规则不同，不合并跨数据集mIoU或挑一个绝对分数排名。图中仅在各自协议内展示。','','![质量与覆盖](quality_coverage_by_protocol.png)','','## 假设与是否采用','','VKITTI2四个加权候选均未通过预声明DEV门槛，故在读取KITTI-360结果前保留S0。KITTI-360等驱动主mIoU：S0 49.801%、S-G 50.277%、S2 50.224%；增益未达冻结1个百分点门槛。S2-H额外损失覆盖，不能以条件准确率提高替代主指标。','','原KITTI08新增稀疏DEV证据：S2相对S0 +7.096pp，S-G +6.846pp，说明几何可靠性在这份地图上有效，但不推翻其他冻结窗口的负结果，也不重新选全局赢家。当前帧二维输入43.535%高于全部融合图，输入质量与融合/关联均值得诊断；S2移动原始类条件准确率仅0.171%，不能称可靠动态物体建图。','','unknown和无地图支持都在有效GT主分母中作漏判，只有冻结GT ignore排除。风险分数未经概率校准；同分组完整接受，共同可达覆盖比较，不能伪造无地图类别。类别/帧/场景支持不足明确不可验，不使用像素独立置信区间。','','## 系统交付','','打开本目录index.html（通过本地HTTP入口）：默认S0，可选原冻结比较组。二维展示来自原生数值语义渲染与独立GT数组，调色板只用于显示。固定按已有评价帧首/中/末选择15个展示案例，含81份方法/案例比较；没有按优良mIoU挑帧。','','3D区域展示5张地图各最多12000个均匀索引中心点，用于检查稳定ID、类别、支持计数和置信值，明确不冒充原生Gaussian渲染，不据此计分。KITTI08全907488个Gaussian的S0语义PLY另行导出并重读核验；保留原几何、RGB的SH0等价值、opacity、scale、rotation及unknown。原完整基图/五组状态/融合日志仍由已核验依赖归档恢复。','','## 完成与未完成分别记录','','00输入/曝光审计、01工程核验、02真实GT-DEV、03冻结及项目隔离评价、04系统展示/总报告/复现归档完成。工程完成与质量验收分开；不签发全局科学PASS，Stage2历史正式FAIL保持。','','严格稠密长期稳定/动态遮挡对应、KITTI08稠密边界分组、实时性/在线SLAM/跟踪增益、学习式耦合、动态Gaussian生命周期未验证。05/06无新增实验，之后若重启研究需另审协议和数据隔离；已消费最终窗口不用于调参后再声称独立评价。','','## 可复现资料','','前置核验prerequisites.json；来源asset_provenance.json；依赖bundle_dependencies.json；原始逐窗结果metrics_by_window.csv；源代码source_build_stage04_delivery.py；复现说明复现与交付说明.md；最终文件哈希delivery_file_hashes.json。大资产分包保留，不改写历史run。']
 (o/'阶段3_最终报告_20261007.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (o/'04_前置核验与完成边界.md').write_text('# 04前置核验\n\n输入与曝光、五张唯一几何、五组全量ID对应、已有独立复算收据、S0选择和协议均已核验。四个完整依赖归档重新计算SHA256一致。具体逐项收据见prerequisites.json。\n\n前置PASS_WITH_DECLARED_QUALITY_LIMITS意味着可整合、展示和归档，不等于加权方法科学PASS。稠密长期时序遮挡与05/06扩展不作为本轮收尾前置。\n',encoding='utf-8')
if __name__=='__main__':main()
