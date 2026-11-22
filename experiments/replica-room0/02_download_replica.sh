#!/usr/bin/env bash
# =============================================================================
# 02 - 下载 Replica 数据集（NICE-SLAM 预处理版，约 14GB）
#
# 与 MonoGS(T0.2) 使用同一份数据，若已下载可通过 DATA_DIR 复用。
# 用法： bash 02_download_replica.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_common.sh"

[ -d "$SPLATAM_DIR" ] || die "未找到 SplaTAM 目录，请先运行 01_setup_env.sh"

# 已存在则跳过（检查标志性场景 room0）
if [ -d "$DATA_DIR/room0" ]; then
  log "检测到 $DATA_DIR/room0 已存在，跳过下载"
  log "如需复用 T0.2(MonoGS) 已下载的数据，可建软链："
  log "  ln -s /你的/Replica路径 \"$DATA_DIR\""
  exit 0
fi

cd "$SPLATAM_DIR"
log "开始下载 Replica（约 14GB，耗时取决于带宽）..."

if [ -f bash_scripts/download_replica.sh ]; then
  log "使用 SplaTAM 内置下载脚本 bash_scripts/download_replica.sh"
  bash bash_scripts/download_replica.sh
else
  warn "未找到内置下载脚本，改用直接下载"
  mkdir -p data
  cd data
  wget -c https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
  log "解压中..."
  unzip -q Replica.zip
fi

[ -d "$DATA_DIR/room0" ] || die "下载/解压后仍未找到 $DATA_DIR/room0，请检查网络或磁盘空间"
log "✅ 02 数据准备完成： $DATA_DIR"
log "   场景: $(ls "$DATA_DIR" | tr '\n' ' ')"
log "下一步： bash 03_run_replica.sh"
