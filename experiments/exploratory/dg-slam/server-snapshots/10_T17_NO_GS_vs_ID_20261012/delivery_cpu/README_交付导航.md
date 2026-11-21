# T17 完整DEV轨迹对照

请先读[T17结果与决定](T17_完整轨迹对照结果与决定.md)，再查[预登记](T17_完整轨迹对照预登记.md)、[协议](protocol.json)及[执行锁](execution_lock.json)。

已曝光DEV的0–199帧，种子1729/2718/31415，各自配对NO_GS_REFINE与ID，共6次，全数正常退出。两个方法均保留高斯建图和作者离线补全。这里的完整轨迹指完整200帧窗口，不是整条TUM序列。

主结果：精度上NO_GS更好，ID没有达到登记收益门槛；NO_GS平均总耗时较ID高6.03%，超过自身5%时间守卫，故综合决定为INCONCLUSIVE_NEITHER_QUALIFIED。保留全部数据，不追加种子或改门槛。

运行产物在runs；成功工程冒烟在smoke；两次失败短冒烟及撤回补丁在receipts/smoke_attempt0、smoke_attempt1。它们是工程记录，不是质量样本。60帧短冒烟不足以完成12关键帧初始化，已在工程附录v2中改为200帧无GT冒烟。最终源码与原T14相同。

冻结评价器保留在scripts/evaluate.py。独立包装器evaluate_cpu64.py只统一浮点精度，修复登记及诊断在receipts/CPU_eval_precision_*.json；没有GPU补跑，没有指标或门槛变化。six-run原始结果在CPU评分前已封存。

scripts/build_runtime.py、repair_empty_graph.py为历史生成/诊断脚本；repair_empty_graph.py对应已经撤回的尝试。复算评分应使用evaluate_cpu64.py及其哈希回执，保持原登记的Linux目录布局和T14原生依赖。CPU本地复核为independent_audit.py。

完整新源码、worker、输入CSV、地图快照、完整轨迹、相机步骤、日志、GT评价参考、报告和图表包含在交付包中。原生sm120环境、原始RGB/depth/masks及模型权重沿用原有归档，精确哈希由执行锁绑定。T14/T15/T16及队友交接文件未改动。

[交付清单](delivery_manifest.json) · [封存验证](receipts/delivery_validation.json) · [ZIP回执](receipts/bundle_receipt.json)
