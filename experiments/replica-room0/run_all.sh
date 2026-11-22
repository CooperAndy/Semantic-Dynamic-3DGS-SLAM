#!/usr/bin/env bash
# =============================================================================
# run_all - 一键串联 T0.3 全流程：环境 → 数据 → 运行 → 评估
#
# 用法：
#   bash run_all.sh            # 完整跑全部四步
#   SKIP_SETUP=1 bash run_all.sh   # 跳过环境搭建（已装好时）
#   SKIP_DOWNLOAD=1 bash run_all.sh # 跳过数据下载（已有数据时）
#
# 每一步独立、幂等；中途失败可单独重跑对应的 0X 脚本。
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_common.sh"

START_TS=$(date +%s)
log "########## T0.3 SplaTAM 复现 一键流程开始 ##########"
log "工作区: $WORKSPACE | 环境: $ENV_NAME | 场景: $SCENE"

if [ "${SKIP_SETUP:-0}" = "1" ]; then
  warn "SKIP_SETUP=1 → 跳过 01 环境搭建"
else
  bash "$SCRIPT_DIR/01_setup_env.sh"
fi

if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  warn "SKIP_DOWNLOAD=1 → 跳过 02 数据下载"
else
  bash "$SCRIPT_DIR/02_download_replica.sh"
fi

bash "$SCRIPT_DIR/03_run_replica.sh" "$SCENE"
bash "$SCRIPT_DIR/04_eval.sh" "$SCENE"

END_TS=$(date +%s)
log "########## 全流程结束，总耗时 $(( (END_TS-START_TS)/60 )) 分钟 ##########"
log "请对照 04 输出的验收表确认 PSNR/ATE 是否达标。"
