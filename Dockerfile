FROM python:3.12-slim

LABEL maintainer="typhoon-system"
LABEL description="台风路径预测系统 - AI预测+地图可视化"

# 安装系统依赖（含eccodes C库，用于ECMWF BUFR解析）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
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

# 安装其余Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# 复制应用代码
COPY backend/ /app/backend/
COPY static/ /app/static/

# 创建数据目录(含ECMWF BUFR缓存和GRIB2缓存)
RUN mkdir -p /app/data/isc /app/data/nii /app/data/predictions /app/data/hashes \
    /app/data/ecmwf_bufr /app/data/grib2_cache /app/backend/models/pangu/initial_data

# 环境变量
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8088

# 健康检查
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8088/api/data/status || exit 1

# 启动命令
WORKDIR /app/backend
CMD ["python", "app.py"]
