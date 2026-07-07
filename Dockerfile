FROM python:3.13-slim

LABEL maintainer="typhoon-system"
LABEL description="台风路径预测系统 - AI预测+地图可视化"

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 设置工作目录
WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY backend/ /app/backend/
COPY static/ /app/static/

# 创建数据目录
RUN mkdir -p /app/data/isc /app/data/nii /app/data/predictions /app/data/hashes /app/models/pangu

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
