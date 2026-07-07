#!/bin/bash
# 台风系统生产部署脚本

set -e

echo "=== 台风路径预测系统 - 生产部署 ==="

# 1. 安装Python依赖
echo "[1] 安装Python依赖..."
pip install flask flask-cors requests numpy torch apscheduler

# 2. 颖取初始数据（近10年）
echo "[2] 缓存近10年历史数据..."
python -c "
from scheduler import fetch_historical_data
fetch_historical_data()
print('历史数据缓存完成')
"

# 3. 训练LSTM模型（首次）
echo "[3] 训练LSTM模型..."
python -c "
from scheduler import auto_train_if_needed
auto_train_if_needed(9999)  # 强制首次训练
print('LSTM训练完成')
"

# 4. 计算初始预测
echo "[4] 计算活跃台风预测..."
python -c "
from scheduler import compute_active_predictions
compute_active_predictions()
print('预测计算完成')
"

echo "=== 部署完成 ==="
echo "启动方式:"
echo "  Flask模式: python app.py"
echo "  独立调度:  python scheduler.py"
echo "  systemd:   systemctl start typhoon-system"
