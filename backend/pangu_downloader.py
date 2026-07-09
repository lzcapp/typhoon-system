#!/usr/bin/env python3
"""
Pangu-Weather ONNX 模型权重下载器

支持三种下载方式（自动 fallback）:
1. gdown (Google Drive 专用工具，支持大文件病毒扫描确认)
2. requests + Google Drive API (直接下载，绕过 gdown 限制)
3. wget (系统命令，最后兜底)

特性:
- 最多重试 3 次
- 下载状态文件（供前端 API 查询进度）
- 支持 CLI 模式（entrypoint.sh 调用）和 import 模式（app.py 调用）

使用:
  python pangu_downloader.py --auto      # 容器启动时自动下载
  python pangu_downloader.py --check     # 仅检查状态
  python pangu_downloader.py             # 交互式下载
  from pangu_downloader import download_pangu_models, get_download_status  # import 模式
"""

import json
import os
import sys
import time
import subprocess

PANGU_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'pangu')
PANGU_24H = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_24.onnx')
PANGU_6H = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_6.onnx')

# Google Drive 文件ID (Pangu-Weather官方仓库确认)
# https://github.com/198808xc/Pangu-Weather
GDRIVE_24H_ID = '1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP'
GDRIVE_6H_ID = '1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4'

MIN_SIZE = 100 * 1024 * 1024  # 100MB (实际约1.1GB)
MAX_RETRIES = 3
RETRY_DELAY = 10  # 秒

# 下载状态文件（供前端 API 查询）
STATUS_FILE = os.path.join(PANGU_MODEL_DIR, 'download_status.json')


def _write_status(status, detail='', progress=0):
    """写入下载状态文件"""
    os.makedirs(PANGU_MODEL_DIR, exist_ok=True)
    data = {
        'status': status,  # idle / downloading / success / failed
        'detail': detail,
        'progress': progress,  # 0-100
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': {
            'pangu_weather_24.onnx': {
                'path': PANGU_24H,
                'exists': os.path.exists(PANGU_24H),
                'size_mb': round(os.path.getsize(PANGU_24H) / (1024 * 1024), 1) if os.path.exists(PANGU_24H) else 0,
                'ready': os.path.exists(PANGU_24H) and os.path.getsize(PANGU_24H) > MIN_SIZE,
            },
            'pangu_weather_6.onnx': {
                'path': PANGU_6H,
                'exists': os.path.exists(PANGU_6H),
                'size_mb': round(os.path.getsize(PANGU_6H) / (1024 * 1024), 1) if os.path.exists(PANGU_6H) else 0,
                'ready': os.path.exists(PANGU_6H) and os.path.getsize(PANGU_6H) > MIN_SIZE,
            },
        },
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
    # 没有状态文件，生成一个当前状态的快照
    _write_status('idle', '尚未开始下载')
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'status': 'idle', 'detail': '未知', 'models': {}}


def _download_with_gdown(file_id, target_path, name):
    """方法1: 使用 gdown 下载（支持 Google Drive 大文件病毒扫描确认）"""
    try:
        import gdown
    except ImportError:
        print(f"  [{name}] gdown 未安装，尝试安装...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'gdown'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import gdown
        except Exception as e:
            print(f"  [{name}] gdown 安装失败: {e}")
            return False

    url = f'https://drive.google.com/uc?id={file_id}'

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] gdown 下载尝试 {attempt}/{MAX_RETRIES}...")
        try:
            # fuzzy=True 处理 Google Drive URL 重定向和确认页面
            gdown.download(url, target_path, quiet=False, fuzzy=True)
            if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
                size_gb = os.path.getsize(target_path) / (1024 ** 3)
                print(f"  [{name}] ✅ gdown 下载成功: {size_gb:.2f} GB")
                return True
            else:
                size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
                print(f"  [{name}] gdown 下载不完整 ({size} bytes)")
                if os.path.exists(target_path):
                    os.remove(target_path)
        except Exception as e:
            print(f"  [{name}] gdown 下载异常: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

        if attempt < MAX_RETRIES:
            print(f"  [{name}] {RETRY_DELAY}秒后重试...")
            time.sleep(RETRY_DELAY)

    return False


def _download_with_requests(file_id, target_path, name):
    """方法2: 使用 requests 直接下载（绕过 gdown 限制）

    Google Drive 大文件下载流程:
    1. 第一次请求获取 confirm token
    2. 带 cookie 重新请求获取实际文件
    """
    try:
        import requests
    except ImportError:
        print(f"  [{name}] requests 不可用")
        return False

    session = requests.Session()
    base_url = 'https://drive.google.com/uc'

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] requests 下载尝试 {attempt}/{MAX_RETRIES}...")

        try:
            # Step 1: 发起下载请求
            response = session.get(base_url, params={'id': file_id, 'export': 'download'},
                                   stream=True, timeout=30)

            # 检查是否需要病毒扫描确认（大文件）
            confirm_token = None
            content_type = response.headers.get('Content-Type', '')

            # Google Drive 返回 HTML 页面表示需要确认
            if 'text/html' in content_type:
                # 从 cookie 中获取 confirm token
                for cookie in session.cookies:
                    if cookie.name.startswith('download_warning'):
                        confirm_token = cookie.value
                        break

                if confirm_token:
                    print(f"  [{name}] 需要 Google Drive 病毒扫描确认，自动处理...")
                    response = session.get(
                        base_url,
                        params={'id': file_id, 'export': 'download', 'confirm': confirm_token},
                        stream=True, timeout=30
                    )
                else:
                    # 有些情况需要从 HTML 中提取 confirm token
                    html = response.text
                    if 'confirm=' in html:
                        # 尝试从 HTML 中提取
                        import re
                        match = re.search(r'confirm=([a-zA-Z0-9_-]+)', html)
                        if match:
                            confirm_token = match.group(1)
                            response = session.get(
                                base_url,
                                params={'id': file_id, 'export': 'download', 'confirm': confirm_token},
                                stream=True, timeout=30
                            )

            # 检查响应
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                print(f"  [{name}] 仍返回 HTML 页面，Google Drive 可能限制了下载")
                continue

            total = int(response.headers.get('Content-Length', 0))
            print(f"  [{name}] 开始下载, 文件大小: {total / (1024**3):.2f} GB")

            downloaded = 0
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # 进度显示（每 100MB 输出一次）
                        if downloaded % (100 * 1024 * 1024) == 0:
                            pct = (downloaded / total * 100) if total > 0 else 0
                            print(f"  [{name}] 进度: {downloaded / (1024**2):.0f}MB / {total / (1024**2):.0f}MB ({pct:.0f}%)")

            if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
                size_gb = os.path.getsize(target_path) / (1024 ** 3)
                print(f"  [{name}] ✅ requests 下载成功: {size_gb:.2f} GB")
                return True
            else:
                size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
                print(f"  [{name}] requests 下载不完整 ({size} bytes)")
                if os.path.exists(target_path):
                    os.remove(target_path)

        except Exception as e:
            print(f"  [{name}] requests 下载异常: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

        if attempt < MAX_RETRIES:
            print(f"  [{name}] {RETRY_DELAY}秒后重试...")
            time.sleep(RETRY_DELAY)

    return False


def _download_with_wget(file_id, target_path, name):
    """方法3: 使用 wget 系统命令（最后兜底）"""
    url = f'https://drive.google.com/uc?id={file_id}&export=download'

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] wget 下载尝试 {attempt}/{MAX_RETRIES}...")
        try:
            result = subprocess.run(
                ['wget', '--no-check-certificate', '-q', '-O', target_path, url],
                timeout=600,
                capture_output=True
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


def _download_model(file_id, target_path, name, method_index=0):
    """下载单个模型文件（自动 fallback）

    method_index:
      0 = gdown → requests → wget
      1 = requests → wget (跳过 gdown)
      2 = wget only
    """
    # 已存在且完整则跳过
    if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_SIZE:
        size_gb = os.path.getsize(target_path) / (1024 ** 3)
        print(f"  [{name}] 已存在 ({size_gb:.2f} GB), 跳过")
        return True

    methods = [_download_with_gdown, _download_with_requests, _download_with_wget]
    methods = methods[method_index:]

    for i, method in enumerate(methods):
        method_name = method.__name__
        print(f"  [{name}] 使用 {method_name}...")
        if method(file_id, target_path, name):
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

    _write_status('downloading', '开始下载 Pangu-Weather 模型权重...', 0)

    models = [
        ('pangu_weather_24.onnx', GDRIVE_24H_ID, PANGU_24H, '24h预报模型'),
        ('pangu_weather_6.onnx', GDRIVE_6H_ID, PANGU_6H, '6h预报模型'),
    ]

    results = {}
    all_success = True

    for i, (filename, file_id, target_path, name) in enumerate(models):
        _write_status('downloading', f'下载 {name} ({i+1}/{len(models)})...',
                       int((i / len(models)) * 100))

        success = _download_model(file_id, target_path, name)
        results[filename] = {
            'success': success,
            'path': target_path,
            'size_gb': round(os.path.getsize(target_path) / (1024**3), 2) if success else 0,
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

    os.makedirs(PANGU_MODEL_DIR, exist_ok=True)

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
    else:
        print("\n❌ 模型不完整, 需要下载")
        print("下载方式:")
        print("  1. 自动: python pangu_downloader.py")
        print("  2. Docker: 容器启动时自动检测并下载")
        print("  3. 前端: 在Web界面点击下载按钮")

    return all_ready


if __name__ == '__main__':
    if '--check' in sys.argv:
        check_models()
    elif '--auto' in sys.argv:
        # 容器启动模式：自动检查并下载缺失的模型
        print("Pangu-Weather 模型自动检测下载")
        print(f"目标目录: {PANGU_MODEL_DIR}")
        print()

        # 先检查状态
        all_ready = True
        for name, path in [('24h模型', PANGU_24H), ('6h模型', PANGU_6H)]:
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
        print("Pangu-Weather ONNX 模型下载器")
        print("=" * 50)
        print(f"目标目录: {PANGU_MODEL_DIR}")
        print("源: Google Drive (官方)")
        print("大小: 约 2.2GB (两个模型各约 1.1GB)")
        print("许可证: BY-NC-SA 4.0 (非商业用途)")
        print("下载方式: gdown → requests → wget (自动 fallback)")
        print(f"重试次数: 每个方法最多 {MAX_RETRIES} 次")
        print()

        result = download_pangu_models()
        if result['success']:
            print("\n✅ 所有模型下载成功！")
        else:
            print("\n❌ 下载未完全成功")
            print("\n手动下载地址:")
            print(f"  24h: https://drive.google.com/file/d/{GDRIVE_24H_ID}/view")
            print(f"  6h:  https://drive.google.com/file/d/{GDRIVE_6H_ID}/view")
            print(f"  放置到: {PANGU_MODEL_DIR}/")
