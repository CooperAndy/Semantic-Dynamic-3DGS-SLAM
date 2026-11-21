# 08复现核查与运行边界

## 1. 最小交付

08轻量完整包保存验收裁决、原始关键证据、源码、环境、19份P6收据、独立registry和SHA。大模型、200个浮点渲染、训练/评估图和日志在两份07依赖包中，见`ARCHIVE_DEPENDENCIES.json`。原KITTI/SGBM/VKITTI2/IMS和预训练权重不在轻量包里，按原输入路径和原G1/G2/模型清单另准备。

## 2. 本地安全重验

解压08包后，在有Python与jsonschema的环境执行：

```powershell
python .\verify_stage08_bundle.py --root .
```

此命令核对归档SHA、238份原证据、已注册manifest SHA和P6 schema/语义；默认不传`--registry`，避免把已注册ID当新注册。无推理、调参或评分计算。P6验证通过不等于M3通过。

Windows的JSON含原服务器绝对路径，以保存当时溯源；本地验证按包内相对路径寻找原件，不假装原绝对路径在Windows存在。

## 3. 服务器原执行记录

`evidence/07_*/remaining_engineering_20261001_v1/jobs.json`、`.command.json`、`.exit`、`.seconds`和`.log`保存15项原命令/退出码/成本。各P6收据只在08命名空间中回溯登记，没有重新执行这些命令。

07 source快照、configs、selected maps、Train160桥及历史侧车引用都须匹配原SHA；PRNG seed20260929、192×640、Train160、mapping10/opacity LR0/训练补点.9及评估sil.5/空洞≤.1%保持。旧原生指标与新float/full/static指标不同，不能把两者直接相减。

若今后重做工程复现，必须另建全新输出目录/run ID，复制配置后改`workdir/run_name/scene_path`和桥路径，先重新冻结；不能直接执行旧命令覆盖现有目录。相同配置四次profile地图与原地图数量有微小差异，不能承诺逐位复现。08此次没有在干净主机重装环境或完整重新训练，因此只声明归档重验通过。

## 4. 消费的正式测试

Scene20三次消费记录均保存，第三次正式模型FAIL。禁止用上述复现名义重新跑Scene20。前两次没有正式评分；clone预测完成不代表正式评分完成。旧KITTI seq07 FAIL也不改写。06按用户要求暂停。

## 5. SHA和源版本

`SHA256SUMS.json`覆盖最终包内普通文件；校验日志在包外避免自引用哈希。原P6validator/schema副本的哈希由包清单锁定。P5无Git工作树时记NOT_A_GIT_WORKTREE，使用源码SHA与实际helper diff；ORB元信息若来自真实Git则按source_receipts记录，不编造版本。

## 6. 科学范围

阅读`stage2_acceptance_v2.md/json`：08审查/归档完成，M3仍未完成。KITTI08历史曝光、物体运动/离场GT、时序转换验收、完整track-to-Gaussian与候选晋升/动态清理机制仍是缺口。
