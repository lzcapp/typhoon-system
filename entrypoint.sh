#!/bin/bash
# 台风路径预测系统容器启动脚本
# 设计原则: 先启动应用, 后台下载模型, 不阻塞健康检查
#
# 环境变量:
#   AUTO_DOWNLOAD_PANGU=1 (默认) 后台自动下载缺失的模型
#   AUTO_DOWNLOAD_PANGU=0          跳过下载
#   HF_ENDPOINT=https://hf-mirror.com  (默认, 国内镜像)

set -u

# 导出 HuggingFace 镜像端点 (国内可访问)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL_DIR="/app/backend/models/pangu"
PANGU_24H="${MODEL_DIR}/pangu_weather_24.onnx"
PANGU_6H="${MODEL_DIR}/pangu_weather_6.onnx"

AUTO_DOWNLOAD="${AUTO_DOWNLOAD_PANGU:-1}"

echo "========================================"
echo "  台风路径预测系统 - 启动"
echo "  HuggingFace镜像: ${HF_ENDPOINT}"
echo "========================================"

# ---- 快速检查依赖状态 (仅打印, 不阻塞) ----
echo ""
echo "[依赖检查]"

if [ -f "/app/backend/models/lstm_best.pt" ]; then
    LSTM_SIZE=$(du -h /app/backend/models/lstm_best.pt 2>/dev/null | cut -f1)
    echo "  ✅ LSTM模型: ${LSTM_SIZE}"
else
    echo "  ℹ️ LSTM模型未找到(需ISC数据训练)"
fi

python3 -c "import onnxruntime; print('  ✅ onnxruntime')" 2>/dev/null || echo "  ℹ️ onnxruntime不可用"

# ---- 快速检查 Pangu 模型 ----
echo ""
echo "[Pangu-Weather模型检查]"

_check_file() {
    local filepath="$1"
    local name="$2"
    if [ -f "$filepath" ]; then
        local fsize
        fsize=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null || echo "0")
        if [ "$fsize" -gt 100000000 ]; then
            local hsize=$(du -h "$filepath" 2>/dev/null | cut -f1)
            echo "  ✅ ${name}: ${hsize}"
            return 0
        else
            echo "  ⚠️ ${name}: 文件不完整(${fsize} bytes), 将后台重新下载"
            return 1
        fi
    else
        echo "  ℹ️ ${name}: 未下载(~1.18GB), 将后台自动下载"
        return 1
    fi
}

PANGU_READY=true
_check_file "$PANGU_24H" "24h预报模型" || PANGU_READY=false
_check_file "$PANGU_6H" "6h预报模型" || PANGU_READY=false

# ---- 立即启动 Flask 应用 (不等待下载) ----
echo ""
echo "========================================"
echo "  🚀 启动台风路径预测系统..."
echo "========================================"
echo ""

# 如果模型未就绪且开启了自动下载, 在后台启动下载线程
if [ "$PANGU_READY" = false ] && [ "$AUTO_DOWNLOAD" = "1" ]; then
    echo "  📥 后台启动 Pangu 模型下载 (不阻塞应用启动)..."
    echo "  下载源: ${HF_ENDPOINT} (国内镜像)"
    echo "  模型大小: 约2.4GB, 下载完成后自动生效"
    echo ""

    # 后台运行下载脚本, 不等待完成
    nohup python3 /app/backend/pangu_downloader.py --auto > /app/data/pangu_download.log 2>&1 &
    echo "  下载进程 PID: $!"
    echo "  日志: /app/data/pangu_download.log"
fi

# ---- 启动应用 ----
cd /app/backend
exec python3 app.py
