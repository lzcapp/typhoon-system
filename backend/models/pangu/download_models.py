#!/usr/bin/env python3
"""
Pangu-Weather ONNX 模型权重下载脚本（兼容入口）

此文件已迁移到 backend/pangu_downloader.py（支持重试 + fallback + 状态追踪）
请使用: python backend/pangu_downloader.py

Docker部署时，容器启动脚本(entrypoint.sh)会后台自动下载缺失的模型。
前端Web界面也提供手动下载按钮。

下载源: HuggingFace 国内镜像 (hf-mirror.com), 无需翻墙
许可证: BY-NC-SA 4.0 (非商业用途)
"""

import os
import sys

# 添加 backend 目录到 path，以便导入 pangu_downloader
_backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, _backend_dir)

from pangu_downloader import check_models, download_pangu_models, PANGU_MODEL_DIR

if __name__ == '__main__':
    if '--check' in sys.argv:
        check_models()
    else:
        print("Pangu-Weather ONNX 模型下载")
        print(f"目标目录: {PANGU_MODEL_DIR}")
        print()
        result = download_pangu_models()
        if result['success']:
            print("\n✅ 所有模型下载成功！")
        else:
            print("\n❌ 下载未完全成功，请查看上方日志")
