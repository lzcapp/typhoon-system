#!/usr/bin/env python3
"""
Pangu-Weather ONNX 模型权重下载器

下载策略:
  1. huggingface_hub 库 (通过 HF_ENDPOINT 环境变量走国内镜像 API, 不走 resolve 重定向)
  2. 如果库不可用, 返回手动下载说明

关键: huggingface_hub 库使用 /api/ 端点获取文件元数据,
      不直接访问 /resolve/ 端点(会被 hf-mirror.com 308重定向到 huggingface.co)

模型来源: NickGeneva/earth_ai (HuggingFace)
  - pangu_weather_24.onnx (1.18 GB)
  - pangu_weather_6.onnx  (1.18 GB)

许可证: BY-NC-SA 4.0 (非商业用途)

手动下载备用地址:
  - 24h模型 百度网盘: https://pan.baidu.com/s/179q2gkz2BrsOR6g3yfTVQg?pwd=eajy
  - 6h模型  百度网盘: https://pan.baidu.com/s/1q7IB7tNjqIwoGC7KVMPn4w?pwd=vxq3
"""

import json
import os
import sys
import time
import subprocess

PANGU_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'pangu')
PANGU_24H = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_24.onnx')
PANGU_6H = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_6.onnx')

# HuggingFace 仓库
HF_REPO_ID = 'NickGeneva/earth_ai'
HF_REPO_SUBDIR = 'pangu'

# 国内镜像 (huggingface_hub 库会使用此环境变量)
HF_MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

# 模型文件信息
MODEL_FILES = {
    'pangu_weather_24.onnx': {
        'local_path': PANGU_24H,
        'name': '24h预报模型',
        'repo_path': f'{HF_REPO_SUBDIR}/pangu_weather_24.onnx',
        'baidu_url': 'https://pan.baidu.com/s/179q2gkz2BrsOR6g3yfTVQg?pwd=eajy',
        'baidu_code': 'eajy',
    },
    'pangu_weather_6.onnx': {
        'local_path': PANGU_6H,
        'name': '6h预报模型',
        'repo_path': f'{HF_REPO_SUBDIR}/pangu_weather_6.onnx',
        'baidu_url': 'https://pan.baidu.com/s/1q7IB7tNjqIwoGC7KVMPn4w?pwd=vxq3',
        'baidu_code': 'vxq3',
    },
}

MIN_SIZE = 100 * 1024 * 1024  # 100MB (实际约1.18GB)
MAX_RETRIES = 3
RETRY_DELAY = 10  # 秒

# 下载状态文件
STATUS_FILE = os.path.join(PANGU_MODEL_DIR, 'download_status.json')

# 手动下载说明
MANUAL_DOWNLOAD_INFO = {
    'models': {
        'pangu_weather_24.onnx': {
            'name': '24h预报模型',
            'size': '~1.18 GB',
            'baidu_url': 'https://pan.baidu.com/s/179q2gkz2BrsOR6g3yfTVQg?pwd=eajy',
            'baidu_code': 'eajy',
            'google_drive_id': '1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP',
        },
        'pangu_weather_6.onnx': {
            'name': '6h预报模型',
            'size': '~1.18 GB',
            'baidu_url': 'https://pan.baidu.com/s/1q7IB7tNjqIwoGC7KVMPn4w?pwd=vxq3',
            'baidu_code': 'vxq3',
            'google_drive_id': '1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4',
        },
    },
    'target_dir': PANGU_MODEL_DIR,
    'instructions': [
        '1. 从百度网盘下载两个 .onnx 文件',
        '2. 将文件放到容器映射目录: ./backend/models/pangu/',
        '3. 重启容器或点击前端"检查模型"按钮',
    ],
}


def _write_status(status, detail='', progress=0):
    """写入下载状态文件"""
    os.makedirs(PANGU_MODEL_DIR, exist_ok=True)
    data = {
        'status': status,  # idle / downloading / success / failed
        'detail': detail,
        'progress': progress,  # 0-100
        'source': HF_MIRROR,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': {},
    }
    for fname, info in MODEL_FILES.items():
        path = info['local_path']
        data['models'][fname] = {
            'path': path,
            'exists': os.path.exists(path),
            'size_mb': round(os.path.getsize(path) / (1024 * 1024), 1) if os.path.exists(path) else 0,
            'ready': os.path.exists(path) and os.path.getsize(path) > MIN_SIZE,
        }
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get_download_status():
    """获取当前下载状态（供 API 调用）"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    _write_status('idle', '尚未开始下载')
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'status': 'idle', 'detail': '未知', 'models': {}}


def get_manual_download_info():
    """获取手动下载说明（供 API 调用）"""
    # 更新模型状态
    info = MANUAL_DOWNLOAD_INFO.copy()
    models = {}
    for fname, minfo in MANUAL_DOWNLOAD_INFO['models'].items():
        path = os.path.join(PANGU_MODEL_DIR, fname)
        exists = os.path.exists(path)
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 1) if exists else 0
        models[fname] = {
            **minfo,
            'exists': exists,
            'size_mb': size_mb,
            'ready': exists and size_mb > 100,
        }
    info['models'] = models
    return info


def _ensure_hf_hub():
    """确保 huggingface_hub 已安装"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("  huggingface_hub 未安装，尝试安装...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'huggingface_hub'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120
            )
            import huggingface_hub
            return True
        except Exception as e:
            print(f"  huggingface_hub 安装失败: {e}")
            return False


def _download_with_hf_hub(repo_path, target_path, name):
    """使用 huggingface_hub 库下载（通过 HF_ENDPOINT 走国内镜像 API）

    关键: huggingface_hub 库使用 /api/ 端点获取文件元数据,
    不直接访问 /resolve/ 端点(会被 hf-mirror.com 重定向到 huggingface.co)
    """
    if not _ensure_hf_hub():
        return False

    from huggingface_hub import hf_hub_download

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] huggingface_hub 下载尝试 {attempt}/{MAX_RETRIES}...")
        print(f"  [{name}] HF_ENDPOINT={HF_MIRROR}")
        try:
            # hf_hub_download 会自动使用 HF_ENDPOINT 环境变量
            # 它通过 API 获取文件元数据, 然后下载
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_path,
                local_dir=PANGU_MODEL_DIR,
            )

            if downloaded_path and os.path.exists(downloaded_path):
                # hf_hub_download 下载到 local_dir/repo_path, 可能需要移动
                if downloaded_path != target_path:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    # 复制而不是移动(hf_hub可能缓存)
                    import shutil
                    shutil.copy2(downloaded_path, target_path)

                if os.path.getsize(target_path) > MIN_SIZE:
                    size_gb = os.path.getsize(target_path) / (1024 ** 3)
                    print(f"  [{name}] ✅ 下载成功: {size_gb:.2f} GB")
                    return True
                else:
                    size = os.path.getsize(target_path)
                    print(f"  [{name}] 下载不完整 ({size} bytes)")
                    if os.path.exists(target_path):
                        os.remove(target_path)
            else:
                print(f"  [{name}] hf_hub_download 返回路径不存在: {downloaded_path}")
        except Exception as e:
            err_msg = str(e)[:200]
            print(f"  [{name}] huggingface_hub 下载异常: {err_msg}")
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass

        if attempt < MAX_RETRIES:
            print(f"  [{name}] {RETRY_DELAY}秒后重试...")
            time.sleep(RETRY_DELAY)

    return False


def _download_model(repo_path, target_path, name):
    """下载单个模型文件"""
    # 已存在且完整则跳过
    if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
        size_gb = os.path.getsize(target_path) / (1024 ** 3)
        print(f"  [{name}] 已存在 ({size_gb:.2f} GB), 跳过")
        return True

    # 只使用 huggingface_hub (走 API, 不走 resolve 重定向)
    print(f"  [{name}] 使用 huggingface_hub 下载...")
    if _download_with_hf_hub(repo_path, target_path, name):
        return True

    print(f"  [{name}] ❌ huggingface_hub 下载失败")
    print(f"  [{name}] 请手动下载:")
    info = MODEL_FILES.get(os.path.basename(target_path), {})
    if info.get('baidu_url'):
        print(f"  [{name}]   百度网盘: {info['baidu_url']} (提取码: {info.get('baidu_code', '')})")
    print(f"  [{name}]   放置到: {target_path}")
    return False


def download_pangu_models():
    """下载所有缺失的 Pangu-Weather 模型

    Returns:
        dict: {'success': bool, 'detail': str, 'models': {...}}
    """
    os.makedirs(PANGU_MODEL_DIR, exist_ok=True)

    _write_status('downloading', f'开始下载 Pangu-Weather 模型权重 (源: {HF_MIRROR})...', 0)

    results = {}
    all_success = True

    model_list = list(MODEL_FILES.items())
    for i, (filename, info) in enumerate(model_list):
        _write_status('downloading', f'下载 {info["name"]} ({i+1}/{len(model_list)})...',
                       int((i / len(model_list)) * 100))

        success = _download_model(info['repo_path'], info['local_path'], info['name'])
        results[filename] = {
            'success': success,
            'path': info['local_path'],
            'size_gb': round(os.path.getsize(info['local_path']) / (1024**3), 2) if success else 0,
        }
        if not success:
            all_success = False

    if all_success:
        _write_status('success', '所有模型下载完成', 100)
    else:
        failed = [k for k, v in results.items() if not v['success']]
        _write_status('failed', f'部分模型下载失败: {", ".join(failed)}，请手动下载', 0)

    return {'success': all_success, 'detail': '完成' if all_success else '部分失败', 'models': results}


def check_models():
    """检查模型文件状态"""
    print("=" * 50)
    print("Pangu-Weather 模型状态检查")
    print("=" * 50)
    print(f"下载源: {HF_MIRROR} (国内镜像)")
    print(f"仓库: {HF_REPO_ID}")
    print(f"目录: {PANGU_MODEL_DIR}")
    print()

    os.makedirs(PANGU_MODEL_DIR, exist_ok=True)

    all_ready = True
    for filename, info in MODEL_FILES.items():
        path = info['local_path']
        if os.path.exists(path) and os.path.getsize(path) > MIN_SIZE:
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(f"  ✅ {filename} ({info['name']}): {size_gb:.2f} GB")
        else:
            print(f"  ❌ {filename} ({info['name']}): 未下载或文件不完整")
            print(f"     百度网盘: {info['baidu_url']} (提取码: {info['baidu_code']})")
            all_ready = False

    if all_ready:
        print("\n✅ 所有模型已就绪, Pangu-Weather推理功能可用")
    else:
        print("\n❌ 模型不完整, 需要下载")
        print("下载方式:")
        print(f"  1. 自动: python pangu_downloader.py --auto")
        print(f"  2. 前端: 在Web界面点击下载按钮")
        print(f"  3. 手动: 从百度网盘下载并放到 {PANGU_MODEL_DIR}/")

    return all_ready


if __name__ == '__main__':
    if '--check' in sys.argv:
        check_models()
    elif '--auto' in sys.argv:
        print("=" * 50)
        print("Pangu-Weather 模型自动检测下载")
        print(f"下载源: {HF_MIRROR} (国内镜像)")
        print(f"目标目录: {PANGU_MODEL_DIR}")
        print("=" * 50)
        print()

        all_ready = True
        for filename, info in MODEL_FILES.items():
            path = info['local_path']
            if not (os.path.exists(path) and os.path.getsize(path) > MIN_SIZE):
                all_ready = False
                break

        if all_ready:
            print("✅ 所有模型已存在，跳过下载")
            _write_status('success', '模型已就绪', 100)
        else:
            result = download_pangu_models()
            if result['success']:
                print("\n✅ 下载成功！")
            else:
                print("\n⚠️ 部分下载失败，请手动下载")
                print("\n手动下载地址:")
                for filename, info in MODEL_FILES.items():
                    print(f"  {filename}: {info['baidu_url']} (提取码: {info['baidu_code']})")
                print(f"  放置到: {PANGU_MODEL_DIR}/")
                sys.exit(0)  # 不返回错误码
    else:
        print("=" * 50)
        print("Pangu-Weather ONNX 模型下载器")
        print("=" * 50)
        print(f"下载源: {HF_MIRROR} (国内HuggingFace镜像)")
        print(f"仓库: {HF_REPO_ID}")
        print(f"目录: {PANGU_MODEL_DIR}")
        print(f"大小: 约 2.4GB (两个模型各约 1.18GB)")
        print(f"许可证: BY-NC-SA 4.0 (非商业用途)")
        print()
        print("备用手动下载:")
        for filename, info in MODEL_FILES.items():
            print(f"  {filename}: {info['baidu_url']} (提取码: {info['baidu_code']})")
        print()

        result = download_pangu_models()
        if result['success']:
            print("\n✅ 所有模型下载成功！")
        else:
            print("\n❌ 下载未完全成功，请使用手动下载")
