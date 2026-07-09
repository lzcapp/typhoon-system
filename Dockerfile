FROM python:3.12-slim

LABEL maintainer="typhoon-system"
LABEL description="台风路径预测系统 - AI预测+地图可视化"

# 安装系统依赖（含eccodes C库 + wget用于模型下载）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 设置工作目录
WORKDIR /app

# 先安装 PyTorch CPU 版本（体积远小于 GPU 版本，约 200MB vs 1.5GB）
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip

# 安装核心Python依赖（必须成功）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# 逐个安装可选Python依赖（每个独立失败，不影响其他）
# P1: ECMWF BUFR台风轨迹 - ecmwf-opendata（纯Python，一般可成功）
RUN pip install --no-cache-dir "ecmwf-opendata>=0.3" || \
    echo "⚠️ ecmwf-opendata安装失败，ECMWF BUFR下载不可用" \
    && rm -rf /root/.cache/pip

# P1: BUFR解析 - eccodes（需要eccodeslib wheel，可能失败）
RUN pip install --no-cache-dir "eccodes>=1.7" || \
    echo "⚠️ eccodes安装失败，BUFR解析将退化为DISS HTTP模式" \
    && rm -rf /root/.cache/pip

# P1: BUFR解析高级接口 - pdbufr（依赖eccodes）
RUN pip install --no-cache-dir "pdbufr>=0.10" || \
    echo "⚠️ pdbufr安装失败，BUFR解析将使用eccodes底层接口" \
    && rm -rf /root/.cache/pip

# P2: Pangu-Weather ONNX推理 - onnxruntime（需要平台对应wheel）
RUN pip install --no-cache-dir "onnxruntime>=1.17" || \
    echo "⚠️ onnxruntime安装失败，Pangu-Weather推理不可用" \
    && rm -rf /root/.cache/pip

# P2: HuggingFace Hub - 用于从国内镜像下载Pangu模型
RUN pip install --no-cache-dir "huggingface_hub>=0.20" || \
    echo "⚠️ huggingface_hub安装失败，Pangu模型自动下载不可用(可手动下载)" \
    && rm -rf /root/.cache/pip

# 复制应用代码和启动脚本
COPY backend/ /app/backend/
COPY static/ /app/static/
COPY entrypoint.sh /app/entrypoint.sh

# 创建数据目录(含ECMWF BUFR缓存和GRIB2缓存)
RUN mkdir -p /app/data/isc /app/data/nii /app/data/predictions /app/data/hashes \
    /app/data/ecmwf_bufr /app/data/grib2_cache /app/backend/models/pangu/initial_data

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com

# 暴露端口
EXPOSE 8088

# 健康检查(应用立即启动, 不等待模型下载)
# start_period=60s 给Flask+scheduler初始化足够时间, retries=5 容忍偶尔慢响应
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8088/api/data/status || exit 1

# 使用entrypoint脚本启动(只启动Flask, 不做下载)
ENTRYPOINT ["/app/entrypoint.sh"]
