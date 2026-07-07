#!/bin/bash
# 台风系统启动脚本 (systemd使用)
cd /opt/typhoon-system/backend
exec python app.py
