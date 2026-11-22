"""Write evidence-based Chinese conclusions and a fixed-view gallery."""
from pathlib import Path
import numpy as np


def number(value,digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def write(out,result,views):
    out=Path(out)
    assert result['status']=='COMPLETE32_REGISTERED_RESULTS'
    assert all(len(result[key])==32 for key in ['trajectory','mapping','semantic','cost'])
    assert len(views)==64 and all(r['mode']=='COMMON_GT' for r in views)
    contrasts=result['formal_contrasts'];lines=['# S23：语义对定位与建图的贡献','',
        '已完成8组开发对照与24组正式验证。正式定位增量以TM−B、TMD−D评价，地图主口径为共同GT参考相机；完整原始数值见四张主表与`全部结果.json`。','',
        '## 主要结果','']
    for contrast in contrasts:
        label=f"{contrast['candidate']}−{contrast['control']}"
        tracking='达到预注册定位联合门' if contrast['tracking_target_pass'] else '未达到预注册定位联合门'
        map_count=sum(r['passed'] for r in contrast['mapping_by_window'])
        lines.append(f"- **{label}：{tracking}。**窗口等权ATE改善{contrast['window_equal_ATE_gain']*100:.2f}%，RPE改善{contrast['window_equal_RPE_gain']*100:.2f}%；地图联合门通过{map_count}/2个窗口。负的改善率表示退化。")
    lines += ['', '定位采用ATE改善≥10%、RPE改善≥5%、任一窗口均值退化≤5%且提案几何失败数不增加的原标准。地图同时检查静态PSNR、全图LPIPS/SSIM/覆盖率、静态深度及近期退化门；逐项通过状态均保存在JSON中。','',
        '## 正式窗口与后端重复','',
        '| 对照 | 窗口 | ATE改善 | RPE改善 | ΔATE均值±标准差/m | ΔRPE均值±标准差/m | 提案失败未增加 | 地图联合门 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for contrast in contrasts:
        maps={r['sequence']:r for r in contrast['mapping_by_window']}
        for window in contrast['windows']:
            a=window['paired_trajectory_differences']['ATE_RMSE_m']
            r=window['paired_trajectory_differences']['RPE_translation_RMSE_m_1frame']
            lines.append(f"| {contrast['candidate']}−{contrast['control']} | {window['sequence']} | {window['gains']['ATE_RMSE_m']*100:.2f}% | {window['gains']['RPE_translation_RMSE_m_1frame']*100:.2f}% | {a['mean']:.4f}±{a['standard_deviation']:.4f} | {r['mean']:.6f}±{r['standard_deviation']:.6f} | {'是' if window['proposal_failures_not_increased'] else '否'} | {'通过' if maps[window['sequence']]['passed'] else '未通过'} |")
    lines += ['', '每个窗口使用后端种子0/1/2，标准差按配对差值、ddof=1计算；两个窗口等权，后端重复不作为三个独立场景。完整地图、语义和开销的配对统计见`逐窗种子配对差值.tsv`。','',
        '## 开发窗八组归因','',
        '| 组 | ATE/m | 单帧平移RPE/m | 静态PSNR/dB | 全图LPIPS | 稀疏mIoU | 算法墙时/s | 耗时可比检查 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    indexed={key:{(r['stage'],r['sequence'],r['seed'],r['group']):r for r in result[key]} for key in ['trajectory','mapping','semantic','cost']}
    for group in ['B','T','M','TM','D','TD','MD','TMD']:
        key=('E2','08',0,group);t=indexed['trajectory'][key];m=indexed['mapping'][key];s=indexed['semantic'][key];c=indexed['cost'][key]
        lines.append(f"| {group} | {number(t['ATE_RMSE_m'])} | {number(t['RPE_translation_RMSE_m_1frame'],6)} | {number(m['static_RGB_PSNR_dB'],3)} | {number(m['LPIPS'])} | {number(s['sparse_mIoU'])} | {number(c['algorithm_seconds'],1)} | {'未见并发' if c['runtime_comparison_eligible'] else c['shared_GPU_status']} |")
    lines += ['', 'T−B检验类别一致性筛选，M−B是闭环系统中的建图开关对照；直接建图贡献另参照E1固定位姿实验。TD−D、MD−D和TMD−D检验动态过滤之外的语义增量。开发窗差值见`8组消融.tsv`；旧E1结果是历史机制诊断，不替代这轮1000帧结果。','',
        '## 动态与语义的实际作用','']
    active=[r for r in result['mapping'] if r['D']]
    informative=sum(r['dynamic_mapping_status']=='INFORMATIVE' for r in active)
    lines.append(f"启用D的{len(active)}组中，{informative}组实际改变了出生候选或地图损失像素；其余组按NOT_INFORMATIVE记录。动态账本中的损失计数为累计像素访问次数，不能当作独立像素或独立样本。匹配输入的有效深度被遮挡数量也不等同于实际删除的对应数量。")
    lines += ['', '语义质量表分别报告共同GT相机下的稀疏mIoU、标签覆盖、静态车辆覆盖/准确率，以及已赋类高斯数。类别图层存在或已赋类比例提高，不单独构成定位与几何建图收益。', '',
        '各类别的独立GT支持数量、预测数量、交并集与缺失预测见`各类语义支持.tsv`；完整19×21混淆计数见`各类语义支持与混淆.json`，额外两列为无类别证据和无渲染覆盖。合并像素的类别IoU只作补充，主表仍使用预登记的逐视图mIoU均值。','',
        '## 运行开销','',
        '| 窗口 | 组 | 实测算法墙时均值/s | 峰值Torch内存均值/GiB | 语义融合均值/s | 可用于速度比较的重复数 |',
        '|---|---|---:|---:|---:|---:|']
    for sequence in ['05','10']:
        for group in ['B','TM','D','TMD']:
            rows=[r for r in result['cost'] if r['stage']=='E3' and r['sequence']==sequence and r['group']==group]
            lines.append(f"| {sequence} | {group} | {np.mean([r['algorithm_seconds'] for r in rows]):.1f} | {np.mean([r['peak_torch_bytes']/2**30 for r in rows]):.3f} | {np.mean([r.get('semantic_fusion',0.) for r in rows]):.2f} | {sum(r['runtime_comparison_eligible'] for r in rows)}/3 |")
    overlap=sum(r['shared_GPU_status']=='OVERLAP_OBSERVED' for r in result['cost'])
    paused=sum(r.get('user_priority_pause_seconds',0.)>0 for r in result['cost'])
    lines += ['', f'资源采样发现{overlap}组存在其他GPU上下文重叠；这些耗时保留为实测墙时，不用于独占GPU速度结论。表中均值与配对TSV保留原始运行数据，作方法速度比较前须按资源标记核对有效重复；不从墙时中估算扣除并发。采样不能排除短于采样间隔的重叠。','',
        f'其中{paused}组含用户优先级暂停：暂停秒数单列，原始墙时保留该等待部分，不估算扣除，也不用于方法速度结论。','',
        '同窗口的预测输入共用一次生成，不在各组间重复相加。语义支持、语义梯度、融合与渲染计时有嵌套，不能把这些小计相加当成总耗时；算法、评价和传输等待分别记录。开销表保留逐组实测，不宣称各方法计算成本相同。','',
        '## 数据与评价口径','',
        '开发窗为KITTI08源3000–3999；正式窗为KITTI05/10各0–999。各组从首帧新建地图，MASS＋H3、3/60历史回放、每帧3000点上限与原验收门固定。每20帧末帧不参与建图或类别融合；评价为20个固定早/近期视图，每组保留共同GT、共同ORB和自选相机三套结果。','',
        '在线输入为双目图像与预测语义/三态状态。KITTI轨迹GT、LiDAR和已有SemanticKITTI点标签只用于评价，未新增人工标注；静态外观与深度只在固定官方投影点支持上比较，未知像素不插值或膨胀。轨迹使用米制P2左目坐标与SE(3)对齐，不修正尺度。','',
        '## 固定典型视图','',
        '全32组各保留逻辑帧19与999，共64张图，均为共同GT参考相机。每张依次显示观测、RGB地图和语义地图；这是预先指定的早/近期帧，不按结果挑选。完整索引见[典型视图索引](典型视图索引.md)。','']
    for logical in [19,999]:
        for group in ['B','TM']:
            row=next(r for r in views if r['stage']=='E2' and r['group']==group and r['logical_frame']==logical)
            lines += [f"### 开发窗{group}，源帧{row['source_frame']}",'',f"![{group} common GT source {row['source_frame']}]({row['path']})",'']
    (out/'结果说明.md').write_text('\n'.join(lines),encoding='utf-8')
    gallery=['# 固定典型视图索引','', '| 阶段 | 窗口 | 后端种子 | 组 | 逻辑帧 | 源帧 | 图像 |','|---|---|---:|---|---:|---:|---|']
    for row in views:
        gallery.append(f"| {row['stage']} | {row['sequence']} | {row['seed']} | {row['group']} | {row['logical_frame']} | {row['source_frame']} | [观测／RGB地图／语义地图]({row['path']}) |")
    (out/'典型视图索引.md').write_text('\n'.join(gallery)+'\n',encoding='utf-8')
