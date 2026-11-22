# T0.3 Replica room0 基线

此处归档团队的环境搭建、Replica 下载、SplaTAM 运行和结果提取脚本。上游源码固定在 `da6bbcd24c248dc884ac7f49d62e91b841b26ccc`。使用 NICE-SLAM 预处理版 Replica。默认环境针对 Linux/WSL、Python 3.10、CUDA 12.1 和 PyTorch 2.1.2。8 GB 显存曾在 room0 长序列遇到 OOM。

```bash
cd experiments/replica-room0
bash 01_setup_env.sh
bash 02_download_replica.sh
bash 03_run_replica.sh
bash 04_eval.sh
```

仅需 S23 时，请按 [S23 复现手册](../online-semantic-dynamic/REPRODUCE.md) 单独建环境。T0.3 正式验收为 PSNR >32 dB 且 ATE <0.6 cm。历史服务器结果为 33.15 dB 和 0.29 cm。
