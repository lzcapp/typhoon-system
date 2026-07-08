#!/usr/bin/env python3
"""
Pangu-Weather ONNX 模型权重下载脚本

下载地址:
- Google Drive: https://drive.google.com/drive/folders/1Rjhf3In8aAnEpYfUkC2j-hLvH0c1c2PX
  - pangu_weather_24.onnx (~1.1GB)
  - pangu_weather_6.onnx (~1.1GB)

由于Google Drive下载较复杂，本脚本提供:
1. 自动从Google Drive下载(gdown)
2. 从备用镜像下载(如果有)
3. 手动下载指引

使用:
  python download_models.py          # 自动下载
  python download_models.py --check  # 检查模型状态
"""

import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'pangu')
PANGU_24H = os.path.join(MODEL_DIR, 'pangu_weather_24.onnx')
PANGU_6H = os.path.join(MODEL_DIR, 'pangu_weather_6.onnx')

# Google Drive文件ID (Pangu-Weather官方)
# 24h模型: https://drive.google.com/file/d/1Rjhf3In8aAnEpYfUkC2j-hLvH0c1c2PX
# 注意: 具体文件ID需要从Google Drive分享链接中获取
GDRIVE_24H_ID = '1Rjhf3In8aAnEpYfUkC2j-hLvH0c1c2PX'  # 需确认
GDRIVE_6H_ID = ''  # 需确认

# 备用下载URL(HuggingFace镜像, 如果有人上传了)
HUGGINGFACE_URL = ''  # 暂无


def check_models():
    """检查模型文件状态"""
    print("=" * 50)
    print("Pangu-Weather 模型状态检查")
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {
        'pangu_weather_24.onnx': PANGU_24H,
        'pangu_weather_6.onnx': PANGU_6H,
    }

    all_ready = True
    for name, path in models.items():
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(f"✅ {name}: {size_gb:.2f} GB")
        else:
            print(f"❌ {name}: 未下载")
            all_ready = False

    if all_ready:
        print("\n✅ 所有模型已就绪, Pangu-Weather推理功能可用")
        # 检查ONNX模型输入形状
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(PANGU_24H, providers=['CPUExecutionProvider'])
            inputs = session.get_inputs()
            for inp in inputs:
                print(f"   模型输入: {inp.name} → shape={inp.shape}")
            outputs = session.get_outputs()
            for out in outputs:
                print(f"   模型输出: {out.name} → shape={out.shape}")
        except ImportError:
            print("   ⚠️ onnxruntime未安装: pip install onnxruntime")
        except Exception as e:
            print(f"   ⚠️ 模型检查失败: {e}")
    else:
        print("\n❌ 模型不完整, 需要下载")
        print_download_instructions()

    return all_ready


def print_download_instructions():
    """打印手动下载指引"""
    print("\n" + "=" * 50)
    print("手动下载指引")
    print("=" * 50)
    print("""
1. 从Google Drive下载ONNX模型权重:
   https://drive.google.com/drive/folders/1Rjhf3In8aAnEpYfUkC2j-hLvH0c1c2PX

   需下载两个文件:
   - pangu_weather_24.onnx (~1.1GB, 24h预报模型)
   - pangu_weather_6.onnx (~1.1GB, 6h预报模型)

2. 放置到以下目录:
   {model_dir}/

3. 或使用gdown自动下载:
   pip install gdown
   gdown https://drive.google.com/uc?id=<FILE_ID>

4. Docker部署:
   模型文件应放在宿主机并挂载到容器, 或打包到镜像中
   注意: 两个模型合计约2.2GB, 会增加镜像体积

推荐做法:
   - 生产环境: 将模型打包到Docker镜像(构建时下载)
   - 开发环境: 手动下载到 models/pangu/ 目录
""".format(model_dir=MODEL_DIR))


def download_with_gdown():
    """尝试使用gdown从Google Drive下载"""
    try:
        import gdown
    except ImportError:
        print("需要安装gdown: pip install gdown")
        print_download_instructions()
        return False

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(PANGU_24H):
        print("下载 pangu_weather_24.onnx...")
        try:
            url = f"https://drive.google.com/uc?id={GDRIVE_24H_ID}"
            gdown.download(url, PANGU_24H, quiet=False)
        except Exception as e:
            print(f"24h模型下载失败: {e}")
            return False

    if not os.path.exists(PANGU_6H):
        print("下载 pangu_weather_6.onnx...")
        try:
            url = f"https://drive.google.com/uc?id={GDRIVE_6H_ID}"
            gdown.download(url, PANGU_6H, quiet=False)
        except Exception as e:
            print(f"6h模型下载失败: {e}")
            return False

    return check_models()


if __name__ == '__main__':
    if '--check' in sys.argv:
        check_models()
    else:
        print("尝试自动下载Pangu-Weather ONNX模型...")
        success = download_with_gdown()
        if not success:
            print("\n自动下载失败, 请参考手动下载指引")
