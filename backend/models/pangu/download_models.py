#!/usr/bin/env python3
"""
Pangu-Weather ONNX 模型权重下载脚本

下载地址(Google Drive官方):
- pangu_weather_24.onnx: https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP
- pangu_weather_6.onnx: https://drive.google.com/file/d/1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4

Docker部署时，容器启动脚本(entrypoint.sh)会自动检测并下载缺失的模型，
无需手动运行此脚本。此脚本仅用于手动下载或检查模型状态。

使用:
  python download_models.py          # 自动下载
  python download_models.py --check  # 检查模型状态

许可证: BY-NC-SA 4.0 (非商业用途)
"""

import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
PANGU_24H = os.path.join(MODEL_DIR, 'pangu_weather_24.onnx')
PANGU_6H = os.path.join(MODEL_DIR, 'pangu_weather_6.onnx')

# Google Drive 文件ID (Pangu-Weather官方仓库确认)
GDRIVE_24H_ID = '1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP'
GDRIVE_6H_ID = '1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4'

MIN_SIZE = 100 * 1024 * 1024  # 100MB (最小合法大小，实际约1.1GB)


def check_models():
    """检查模型文件状态"""
    print("=" * 50)
    print("Pangu-Weather 模型状态检查")
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {
        'pangu_weather_24.onnx (24h预报)': PANGU_24H,
        'pangu_weather_6.onnx (6h预报)': PANGU_6H,
    }

    all_ready = True
    for name, path in models.items():
        if os.path.exists(path) and os.path.getsize(path) > MIN_SIZE:
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(f"  ✅ {name}: {size_gb:.2f} GB")
        else:
            print(f"  ❌ {name}: 未下载或文件不完整")
            all_ready = False

    if all_ready:
        print("\n✅ 所有模型已就绪, Pangu-Weather推理功能可用")
        _check_model_shapes()
    else:
        print("\n❌ 模型不完整, 需要下载")
        print("\n下载方式:")
        print("  1. 自动: python download_models.py")
        print("  2. Docker: 容器启动时自动检测并下载")
        print("  3. 手动: 从Google Drive下载到 models/pangu/ 目录")

    return all_ready


def _check_model_shapes():
    """检查ONNX模型输入输出形状"""
    try:
        import onnxruntime as ort
        for model_path, model_name in [(PANGU_24H, '24h'), (PANGU_6H, '6h')]:
            if not os.path.exists(model_path):
                continue
            print(f"\n  --- {model_name}模型元数据 ---")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            for inp in session.get_inputs():
                print(f"  输入: {inp.name} → shape={inp.shape}, dtype={inp.type}")
            for out in session.get_outputs():
                print(f"  输出: {out.name} → shape={out.shape}, dtype={out.type}")
    except ImportError:
        print("  ⚠️ onnxruntime未安装: pip install onnxruntime")
    except Exception as e:
        print(f"  ⚠️ 模型检查失败: {e}")


def download_with_gdown():
    """从Google Drive自动下载模型"""
    try:
        import gdown
    except ImportError:
        print("需要安装gdown: pip install gdown")
        print("或运行: pip install gdown && python download_models.py")
        return False

    os.makedirs(MODEL_DIR, exist_ok=True)

    models_to_download = [
        ('pangu_weather_24.onnx', GDRIVE_24H_ID, PANGU_24H),
        ('pangu_weather_6.onnx', GDRIVE_6H_ID, PANGU_6H),
    ]

    for name, file_id, target_path in models_to_download:
        if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
            size_gb = os.path.getsize(target_path) / (1024 ** 3)
            print(f"✅ {name} 已存在 ({size_gb:.2f} GB), 跳过下载")
            continue

        print(f"\n下载 {name} (~1.1GB)...")
        url = f"https://drive.google.com/uc?id={file_id}"

        try:
            gdown.download(url, target_path, quiet=False)
            if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
                size_gb = os.path.getsize(target_path) / (1024 ** 3)
                print(f"✅ {name} 下载完成 ({size_gb:.2f} GB)")
            else:
                print(f"❌ {name} 下载失败或文件不完整")
                if os.path.exists(target_path):
                    os.remove(target_path)
                return False
        except Exception as e:
            print(f"❌ {name} 下载失败: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)
            return False

    return check_models()


if __name__ == '__main__':
    if '--check' in sys.argv:
        check_models()
    else:
        print("Pangu-Weather ONNX模型自动下载")
        print("=" * 50)
        print(f"目标目录: {MODEL_DIR}")
        print("源: Google Drive (官方)")
        print("大小: 约2.2GB (两个模型各约1.1GB)")
        print("许可证: BY-NC-SA 4.0 (非商业用途)")
        print("")
        success = download_with_gdown()
        if not success:
            print("\n自动下载失败，请手动下载:")
            print(f"  24h: https://drive.google.com/file/d/{GDRIVE_24H_ID}/view")
            print(f"  6h:  https://drive.google.com/file/d/{GDRIVE_6H_ID}/view")
            print(f"  放置到: {MODEL_DIR}/")
