#!/usr/bin/env python3
"""
Pangu-Weather 模型下载入口脚本
实际下载逻辑在 pangu_downloader.py 中
"""
import sys
import os

# 添加 backend 目录到 path (本脚本在 models/pangu/ 下)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_dir)

from pangu_downloader import check_models, download_pangu_models

if __name__ == '__main__':
    if '--check' in sys.argv:
        check_models()
    else:
        result = download_pangu_models()
        if result['success']:
            print("\n✅ 所有模型下载成功！")
        else:
            print("\n❌ 下载未完全成功")
            from pangu_downloader import MODEL_FILES
            print("\n手动下载地址:")
            for filename, info in MODEL_FILES.items():
                print(f"  {filename}: {info['baidu_url']} (提取码: {info['baidu_code']})")
            print(f"  放置到: {info['local_path']}")
