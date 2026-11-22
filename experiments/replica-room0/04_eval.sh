#!/usr/bin/env bash
# =============================================================================
# 04 - 收集并核对验收指标（PSNR / SSIM / ATE）
#
# SplaTAM 在运行结束时会打印最终的 tracking ATE 与渲染指标；本脚本从运行日志
# 中抽取这些数字，并定位实验输出目录，最后对照 T0.3 验收标准给出结论。
#
# 用法： bash 04_eval.sh [scene]
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_common.sh"

SCENE="${1:-$SCENE}"
RUN_LOG="$LOG_DIR/run_${SCENE}.log"
[ -f "$RUN_LOG" ] || die "未找到运行日志 $RUN_LOG，请先运行 03_run_replica.sh"

echo "================================================================"
log "从运行日志抽取指标： $RUN_LOG"
echo "----------------------------------------------------------------"
# SplaTAM 末尾会输出 PSNR/SSIM/LPIPS/Depth-L1/ATE RMSE 等关键字
grep -Ei 'psnr|ssim|lpips|depth.?l1|ate|rmse|final|average' "$RUN_LOG" | tail -40 \
  || warn "未在日志中匹配到指标关键字，请人工查看日志末尾"
echo "----------------------------------------------------------------"

# 定位实验输出目录（SplaTAM 默认写到 experiments/<group>/<run>/）
EXP_DIR="$(ls -dt "$SPLATAM_DIR"/experiments/*/* 2>/dev/null | head -1 || true)"
if [ -n "$EXP_DIR" ]; then
  log "实验输出目录: $EXP_DIR"
  ls -la "$EXP_DIR" 2>/dev/null | head -20
  # 若存在独立的新视角评估脚本，可额外跑一遍渲染指标
  if [ -f "$SPLATAM_DIR/scripts/eval_novel_view.py" ]; then
    log "（可选）如需重算 NVS 指标： cd $SPLATAM_DIR && python scripts/eval_novel_view.py <config>"
  fi
else
  warn "未找到 experiments/ 输出目录，请确认运行是否正常结束"
fi

echo "================================================================"
cat <<'EOF'
T0.3 验收标准（来自 阶段0_技术执行文档.md 4.5 / 论文参考）：
  ┌────────────┬──────────────┬────────────┐
  │ 指标       │ 论文参考值   │ 可接受范围 │
  ├────────────┼──────────────┼────────────┤
  │ PSNR (dB)  │ ~33 – 36     │ > 32       │  ← T0.3 主验收项（偏差 < 1dB）
  │ ATE (cm)   │ ~0.3 – 0.5   │ < 0.6      │
  └────────────┴──────────────┴────────────┘
请把上面抽取到的 PSNR / ATE 与此表比对：
  - PSNR > 32 且 ATE < 0.6cm  → ✅ 复现通过
  - 偏差较大 → 见 runbook「排错」：优先查 (1) 光栅化内核是否编译成功
    (2) torch/CUDA 版本匹配 (3) 数据集是否为 NICE-SLAM 预处理版
EOF
log "✅ 04 评估完成。请据上表人工判定是否达标，并在 M1 验收清单中打勾。"
