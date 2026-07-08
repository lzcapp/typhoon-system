#!/bin/bash
# 台风路径预测系统容器启动脚本
# 功能: 自动检测并下载Pangu-Weather ONNX模型权重
#
# 环境变量:
#   AUTO_DOWNLOAD_PANGU=1 (默认) 自动下载缺失的模型
#   AUTO_DOWNLOAD_PANGU=0          跳过下载，仅检查状态

set -e

MODEL_DIR="/app/backend/models/pangu"
PANGU_24H="${MODEL_DIR}/pangu_weather_24.onnx"
PANGU_6H="${MODEL_DIR}/pangu_weather_6.onnx"

# Google Drive 文件ID (Pangu-Weather官方仓库)
GDRIVE_24H_ID="1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP"
GDRIVE_6H_ID="1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4"

AUTO_DOWNLOAD="${AUTO_DOWNLOAD_PANGU:-1}"

echo "========================================"
echo "  台风路径预测系统 - 启动检查"
echo "========================================"

# ---- 检查核心依赖 ----
echo ""
echo "[1/3] 核心依赖..."

if [ -f "/app/backend/models/lstm_best.pt" ]; then
    LSTM_SIZE=$(du -h /app/backend/models/lstm_best.pt | cut -f1)
    echo "  ✅ LSTM模型: ${LSTM_SIZE}"
else
    echo "  ⚠️ LSTM模型未找到(需ISC数据训练)"
fi

# ---- 检查可选依赖 ----
echo ""
echo "[2/3] 可选依赖..."

python3 -c "from ecmwf.opendata import Client; print('  ✅ ecmwf-opendata')" 2>/dev/null || \
    echo "  ⚠️ ecmwf-opendata不可用"

python3 -c "import pdbufr; print('  ✅ pdbufr')" 2>/dev/null || \
    echo "  ⚠️ pdbufr不可用"

python3 -c "import eccodes; print('  ✅ eccodes')" 2>/dev/null || \
    echo "  ⚠️ eccodes不可用"

python3 -c "import onnxruntime; print('  ✅ onnxruntime')" 2>/dev/null || \
    echo "  ⚠️ onnxruntime不可用"

# ---- 检查Pangu-Weather模型 ----
echo ""
echo "[3/3] Pangu-Weather ONNX模型..."

NEED_DOWNLOAD=false
MIN_SIZE=100000000  # 100MB

_check_file() {
    local filepath="$1"
    local name="$2"
    if [ -f "$filepath" ]; then
        local fsize
        # macOS 和 Linux stat 语法不同
        fsize=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null || echo "0")
        if [ "$fsize" -gt "$MIN_SIZE" ]; then
            local hsize=$(du -h "$filepath" | cut -f1)
            echo "  ✅ ${name}: ${hsize}"
            return 0
        else
            echo "  ❌ ${name}: 文件不完整(${fsize} bytes)"
            NEED_DOWNLOAD=true
            return 1
        fi
    else
        echo "  ❌ ${name}: 未下载(~1.1GB)"
        NEED_DOWNLOAD=true
        return 1
    fi
}

_check_file "$PANGU_24H" "24h预报模型"
_check_file "$PANGU_6H" "6h预报模型"

# ---- 自动下载 ----
if [ "$NEED_DOWNLOAD" = true ] && [ "$AUTO_DOWNLOAD" = "1" ]; then
    echo ""
    echo "========================================"
    echo "  🔄 自动下载Pangu-Weather模型权重"
    echo "  合计约2.2GB，首次启动需等待"
    echo "  已下载的模型通过volume持久化"
    echo "  后续启动无需重复下载"
    echo "========================================"
    echo ""

    mkdir -p "$MODEL_DIR"

    # 安装gdown
    echo "安装下载工具..."
    pip install --no-cache-dir gdown 2>/dev/null || echo "  ⚠️ gdown安装失败，尝试继续"

    _download_model() {
        local file_id="$1"
        local target="$2"
        local name="$3"

        # 已存在且完整则跳过
        if [ -f "$target" ]; then
            local fsize
            fsize=$(stat -c%s "$target" 2>/dev/null || stat -f%z "$target" 2>/dev/null || echo "0")
            if [ "$fsize" -gt "$MIN_SIZE" ]; then
                echo "  ✅ ${name} 已存在, 跳过"
                return 0
            fi
        fi

        echo "  下载 ${name} (~1.1GB)..."
        python3 << PYEOF
import sys
try:
    import gdown
except ImportError:
    print("  ⚠️ gdown不可用，跳过下载")
    sys.exit(0)

import os
url = 'https://drive.google.com/uc?id=${file_id}'
target = '${target}'
min_size = ${MIN_SIZE}

try:
    gdown.download(url, target, quiet=False)
    if os.path.exists(target) and os.path.getsize(target) > min_size:
        size_gb = os.path.getsize(target) / (1024**3)
        print(f'  ✅ ${name}下载完成: {size_gb:.2f} GB')
    else:
        print(f'  ❌ ${name}下载失败或文件不完整')
        if os.path.exists(target):
            os.remove(target)
except Exception as e:
    print(f'  ❌ ${name}下载失败: {e}')
    if os.path.exists(target):
        os.remove(target)
PYEOF
    }

    _download_model "$GDRIVE_24H_ID" "$PANGU_24H" "24h模型"
    _download_model "$GDRIVE_6H_ID" "$PANGU_6H" "6h模型"

    # 下载结果摘要
    echo ""
    echo "下载结果:"
    _check_file "$PANGU_24H" "24h模型" || true
    _check_file "$PANGU_6H" "6h模型" || true

elif [ "$NEED_DOWNLOAD" = true ] && [ "$AUTO_DOWNLOAD" != "1" ]; then
    echo ""
    echo "⚠️ AUTO_DOWNLOAD_PANGU=0, 跳过自动下载"
    echo "手动下载: python backend/models/pangu/download_models.py"
fi

# ---- 启动应用 ----
echo ""
echo "========================================"
echo "  🚀 启动台风路径预测系统..."
echo "========================================"
echo ""

cd /app/backend
exec python3 app.py
