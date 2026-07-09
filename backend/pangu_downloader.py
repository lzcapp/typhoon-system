#!/usr/bin/env python3
"""
Pangu-Weather ONNX 模型权重下载器

国内可访问下载源（HuggingFace 镜像）:
  主源: hf-mirror.com (国内 HF 镜像, 无需翻墙)
  备源: huggingface.co (国际源, 备用)

下载方式（自动 fallback）:
1. huggingface_hub 包 (支持断点续传, 推荐)
2. requests 直接下载 (绕过包依赖)
3. wget 系统命令 (最后兜底)

模型来源: NickGeneva/earth_ai (HuggingFace)
  - pangu_weather_24.onnx (1.18 GB)
  - pangu_weather_6.onnx  (1.18 GB)

许可证: BY-NC-SA 4.0 (非商业用途)
"""

import json
import os
import sys
import time
import subprocess

PANGU_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'pangu')
PANGU_24H = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_24.onnx')
PANGU_6H = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_6.onnx')

# HuggingFace 仓库 (NickGeneva/earth_ai, 包含全部4个Pangu ONNX模型)
HF_REPO_ID = 'NickGeneva/earth_ai'
HF_REPO_SUBDIR = 'pangu'

# 国内镜像 (hf-mirror.com 是 HuggingFace 的国内镜像, 无需翻墙)
HF_MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

# 模型文件名
MODEL_FILES = {
    'pangu_weather_24.onnx': {
        'local_path': PANGU_24H,
        'name': '24h预报模型',
        'repo_path': f'{HF_REPO_SUBDIR}/pangu_weather_24.onnx',
    },
    'pangu_weather_6.onnx': {
        'local_path': PANGU_6H,
        'name': '6h预报模型',
        'repo_path': f'{HF_REPO_SUBDIR}/pangu_weather_6.onnx',
    },
}

MIN_SIZE = 100 * 1024 * 1024  # 100MB (实际约1.18GB)
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# 下载状态文件（供前端 API 查询）
STATUS_FILE = os.path.join(PANGU_MODEL_DIR, 'download_status.json')


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


def _get_hf_url(repo_path):
    """构建 HuggingFace 镜像下载 URL"""
    return f'{HF_MIRROR}/{HF_REPO_ID}/resolve/main/{repo_path}'


def _download_with_hf_hub(repo_path, target_path, name):
    """方法1: 使用 huggingface_hub 包下载（支持断点续传, 推荐）"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"  [{name}] huggingface_hub 未安装，尝试安装...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'huggingface_hub'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
            )
            from huggingface_hub import hf_hub_download
        except Exception as e:
            print(f"  [{name}] huggingface_hub 安装失败: {e}")
            return False

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] huggingface_hub 下载尝试 {attempt}/{MAX_RETRIES}...")
        try:
            # hf_hub_download 会自动使用 HF_ENDPOINT 环境变量
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_path,
                local_dir=PANGU_MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download 下载到 local_dir/filename, 移动到目标路径
            if downloaded_path and os.path.exists(downloaded_path):
                if downloaded_path != target_path:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    os.replace(downloaded_path, target_path)

                if os.path.getsize(target_path) > MIN_SIZE:
                    size_gb = os.path.getsize(target_path) / (1024 ** 3)
                    print(f"  [{name}] ✅ huggingface_hub 下载成功: {size_gb:.2f} GB")
                    return True
                else:
                    size = os.path.getsize(target_path)
                    print(f"  [{name}] 下载不完整 ({size} bytes)")
                    if os.path.exists(target_path):
                        os.remove(target_path)
            else:
                print(f"  [{name}] hf_hub_download 返回路径不存在: {downloaded_path}")
        except Exception as e:
            print(f"  [{name}] huggingface_hub 下载异常: {e}")
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass

        if attempt < MAX_RETRIES:
            print(f"  [{name}] {RETRY_DELAY}秒后重试...")
            time.sleep(RETRY_DELAY)

    return False


def _download_with_requests(repo_path, target_path, name):
    """方法2: 使用 requests 直接下载（从 HF 镜像）"""
    try:
        import requests
    except ImportError:
        print(f"  [{name}] requests 不可用")
        return False

    url = _get_hf_url(repo_path)
    print(f"  [{name}] requests 下载: {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] requests 下载尝试 {attempt}/{MAX_RETRIES}...")

        try:
            # 先 HEAD 请求检查文件大小
            head_resp = requests.head(url, timeout=30, allow_redirects=True)
            total = int(head_resp.headers.get('Content-Length', 0))
            if total > 0:
                print(f"  [{name}] 文件大小: {total / (1024**3):.2f} GB")

            # 流式下载
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total = total or int(response.headers.get('Content-Length', 0))
            downloaded = 0
            last_report = 0

            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # 每 200MB 输出一次进度
                        if downloaded - last_report >= 200 * 1024 * 1024:
                            pct = (downloaded / total * 100) if total > 0 else 0
                            print(f"  [{name}] 进度: {downloaded / (1024**2):.0f}MB / {total / (1024**2):.0f}MB ({pct:.0f}%)")
                            last_report = downloaded

            if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
                size_gb = os.path.getsize(target_path) / (1024 ** 3)
                print(f"  [{name}] ✅ requests 下载成功: {size_gb:.2f} GB")
                return True
            else:
                size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
                print(f"  [{name}] 下载不完整 ({size} bytes)")
                if os.path.exists(target_path):
                    os.remove(target_path)

        except Exception as e:
            print(f"  [{name}] requests 下载异常: {e}")
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass

        if attempt < MAX_RETRIES:
            print(f"  [{name}] {RETRY_DELAY}秒后重试...")
            time.sleep(RETRY_DELAY)

    return False


def _download_with_wget(repo_path, target_path, name):
    """方法3: 使用 wget 系统命令（最后兜底）"""
    url = _get_hf_url(repo_path)
    print(f"  [{name}] wget 下载: {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] wget 下载尝试 {attempt}/{MAX_RETRIES}...")
        try:
            result = subprocess.run(
                ['wget', '--no-check-certificate', '-q', '--show-progress',
                 '-O', target_path, url],
                timeout=900,  # 15分钟超时
                capture_output=False
            )
            if result.returncode == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
                size_gb = os.path.getsize(target_path) / (1024 ** 3)
                print(f"  [{name}] ✅ wget 下载成功: {size_gb:.2f} GB")
                return True
            else:
                print(f"  [{name}] wget 下载失败 (exit={result.returncode})")
                if os.path.exists(target_path):
                    os.remove(target_path)
        except FileNotFoundError:
            print(f"  [{name}] wget 命令不存在")
            return False
        except subprocess.TimeoutExpired:
            print(f"  [{name}] wget 下载超时")
            if os.path.exists(target_path):
                os.remove(target_path)
        except Exception as e:
            print(f"  [{name}] wget 下载异常: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

        if attempt < MAX_RETRIES:
            print(f"  [{name}] {RETRY_DELAY}秒后重试...")
            time.sleep(RETRY_DELAY)

    return False


def _download_model(repo_path, target_path, name):
    """下载单个模型文件（自动 fallback）"""
    # 已存在且完整则跳过
    if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
        size_gb = os.path.getsize(target_path) / (1024 ** 3)
        print(f"  [{name}] 已存在 ({size_gb:.2f} GB), 跳过")
        return True

    methods = [
        ('huggingface_hub', _download_with_hf_hub),
        ('requests', _download_with_requests),
        ('wget', _download_with_wget),
    ]

    for method_name, method in methods:
        print(f"  [{name}] 尝试 {method_name}...")
        if method(repo_path, target_path, name):
            return True
        print(f"  [{name}] {method_name} 失败，尝试下一个方法...")

    print(f"  [{name}] ❌ 所有下载方法均失败")
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
        _write_status('failed', f'部分模型下载失败: {", ".join(failed)}', 0)

    return {'success': all_success, 'detail': '完成' if all_success else '部分失败', 'models': results}


def check_models():
    """检查模型文件状态"""
    print("=" * 50)
    print("Pangu-Weather 模型状态检查")
    print("=" * 50)
    print(f"下载源: {HF_MIRROR}")
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
            all_ready = False

    if all_ready:
        print("\n✅ 所有模型已就绪, Pangu-Weather推理功能可用")
    else:
        print("\n❌ 模型不完整, 需要下载")
        print("下载方式:")
        print(f"  1. 自动: python pangu_downloader.py --auto")
        print(f"  2. Docker: 容器启动时后台自动下载")
        print(f"  3. 前端: 在Web界面点击下载按钮")
        print(f"  4. 手动下载:")
        for filename, info in MODEL_FILES.items():
            url = _get_hf_url(info['repo_path'])
            print(f"     {filename}: {url}")

    return all_ready


if __name__ == '__main__':
    if '--check' in sys.argv:
        check_models()
    elif '--auto' in sys.argv:
        # 容器启动模式：自动检查并下载缺失的模型
        print("=" * 50)
        print("Pangu-Weather 模型自动检测下载")
        print(f"下载源: {HF_MIRROR} (国内镜像)")
        print(f"目标目录: {PANGU_MODEL_DIR}")
        print("=" * 50)
        print()

        # 先检查状态
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
                print("\n⚠️ 部分下载失败，可在前端手动重试")
                sys.exit(0)  # 不返回错误码，不阻止容器启动
    else:
        # 交互模式
        print("=" * 50)
        print("Pangu-Weather ONNX 模型下载器")
        print("=" * 50)
        print(f"下载源: {HF_MIRROR} (国内HuggingFace镜像)")
        print(f"仓库: {HF_REPO_ID}")
        print(f"目录: {PANGU_MODEL_DIR}")
        print(f"大小: 约 2.4GB (两个模型各约 1.18GB)")
        print(f"许可证: BY-NC-SA 4.0 (非商业用途)")
        print(f"下载方式: huggingface_hub → requests → wget (自动 fallback)")
        print(f"重试次数: 每个方法最多 {MAX_RETRIES} 次")
        print()

        result = download_pangu_models()
        if result['success']:
            print("\n✅ 所有模型下载成功！")
        else:
            print("\n❌ 下载未完全成功")
            print("\n手动下载地址:")
            for filename, info in MODEL_FILES.items():
                url = _get_hf_url(info['repo_path'])
                print(f"  {filename}: {url}")
            print(f"  放置到: {PANGU_MODEL_DIR}/")
