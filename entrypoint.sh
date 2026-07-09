#!/bin/bash
# 台风路径预测系统容器启动脚本
# 设计原则: 只启动Flask应用, 不做任何下载/阻塞操作
# Pangu模型下载通过前端API手动触发或huggingface_hub库自动处理

set -u

echo "========================================"
echo "  台风路径预测系统 - 启动"
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
python3 -c "import huggingface_hub; print('  ✅ huggingface_hub')" 2>/dev/null || echo "  ℹ️ huggingface_hub不可用"

# ---- 快速检查 Pangu 模型 (仅打印状态, 不下载) ----
echo ""
echo "[Pangu-Weather模型检查]"

PANGU_24H="/app/backend/models/pangu/pangu_weather_24.onnx"
PANGU_6H="/app/backend/models/pangu/pangu_weather_6.onnx"

for f in "$PANGU_24H" "$PANGU_6H"; do
    if [ -f "$f" ]; then
        FSIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "0")
        if [ "$FSIZE" -gt 100000000 ]; then
            HSIZE=$(du -h "$f" 2>/dev/null | cut -f1)
            echo "  ✅ $(basename $f): ${HSIZE}"
        else
            echo "  ⚠️ $(basename $f): 文件不完整(${FSIZE} bytes)"
        fi
    else
        echo "  ℹ️ $(basename $f): 未下载"
    fi
done

echo ""
echo "========================================"
echo "  🚀 启动Flask应用..."
echo "  Pangu模型可通过前端界面手动下载"
echo "========================================"
echo ""

# ---- 直接启动应用, 不做任何下载 ----
cd /app/backend
exec python3 app.py
