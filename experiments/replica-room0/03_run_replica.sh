#!/usr/bin/env bash
# =============================================================================
# 03 - 运行 SplaTAM（Replica RGB-D，默认 room0）
#
# 用法： bash 03_run_replica.sh [scene]
#   scene 缺省为 room0；其它场景（office0 等）需对应 config 存在
# 输出日志： $LOG_DIR/run_<scene>.log
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_common.sh"

SCENE="${1:-$SCENE}"
[ "$SCENE" = room0 ] || die "此复现配置固定使用 Replica room0"
activate_env
check_gpu

[ -d "$DATA_DIR/$SCENE" ] || die "未找到数据 $DATA_DIR/$SCENE，请先运行 02_download_replica.sh"

cd "$SPLATAM_DIR"

# 定位 config：优先 configs/replica/splatam.py（官方标准复现入口）
CFG="configs/replica/splatam.py"
[ -f "$CFG" ] || CFG="$(find configs -iname '*replica*' -name '*.py' | head -1 || true)"
[ -n "$CFG" ] && [ -f "$CFG" ] || die "未找到 Replica 配置文件，请检查仓库 configs/ 结构"
log "使用配置: $CFG （默认场景由配置内 scene_name 决定，标准复现为 room0）"

# 关闭 wandb（Weights&Biases 云端日志）：复现不需要它，避免要求登录 API key。
# 双保险：配置里已设 use_wandb=False，这里再禁用 wandb 运行模式兜底其它配置。
export WANDB_MODE=disabled

mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_${SCENE}.log"
log "开始运行 SplaTAM（场景 $SCENE）。"
log "实时日志： tail -f $RUN_LOG"

# tee 保留完整日志，供 04_eval.sh 解析指标
set -o pipefail
python scripts/splatam.py "$CFG" 2>&1 | tee "$RUN_LOG"

log "✅ 03 运行完成。下一步： bash 04_eval.sh $SCENE"
