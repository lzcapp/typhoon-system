# 台风路径预测系统 - 1Panel 部署指南

## 前置条件

- 服务器已安装 1Panel（如未安装，参考 [1Panel 官方文档](https://1panel.cn/docs/)）
- 1Panel 中已安装 Docker 和 Docker Compose（1Panel 应用商店自带）
- 服务器有 ≥2GB 内存（LSTM 推理需要）
- 服务器可访问外网（抓取 ISC/Open-Meteo 数据）

---

## 方式一：拉取预构建镜像（最简单，推荐）

镜像已通过 GitHub Actions 自动构建并发布到 GHCR 和 DockerHub，无需本地构建。

### 第 1 步：创建目录

```bash
mkdir -p /opt/typhoon-system/data /opt/typhoon-system/models /opt/typhoon-system/static
```

### 第 2 步：在 1Panel 中创建编排

1. 打开 1Panel → **容器** → **编排** → **创建编排**
2. 名称：`typhoon-system`
3. 工作目录：`/opt/typhoon-system`
4. 粘贴以下内容：

```yaml
version: '3.8'

services:
  typhoon-system:
    # GHCR（GitHub Container Registry，推荐）
    image: ghcr.io/lzcapp/typhoon-system:latest
    # DockerHub（取消注释下行，注释上行即可切换）
    # image: seeleo/typhoon-system:latest
    container_name: typhoon-system
    restart: always
    ports:
      - "8088:8088"
    volumes:
      - /opt/typhoon-system/data:/app/data
      - /opt/typhoon-system/models:/app/backend/models
      - /opt/typhoon-system/static:/app/static
    environment:
      - TZ=Asia/Shanghai
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8088/api/data/status"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 30s
```

5. 点击 **创建** → 自动拉取镜像并启动

> 💡 拉取约 500MB 镜像，首次启动后自动初始化数据+训练

### 第 3 步：验证

浏览器访问 `http://你的服务器IP:8088`

---

## 方式二：本地构建部署

适合想修改代码后自行构建的场景。

### 第 1 步：获取项目文件

```bash
# 克隆仓库
git clone https://github.com/seeleo/typhoon-system.git /opt/typhoon-system
```

或通过 SCP/1Panel文件管理上传。

### 第 2 步：本地构建并启动

```bash
cd /opt/typhoon-system
docker compose up -d --build
```

> ⚠️ **首次构建约 5-10 分钟**：需下载 PyTorch CPU 版（约 200MB）

或使用 1Panel 编排：创建编排 → 粘贴 `docker-compose.yml` 内容，`context` 改为 `/opt/typhoon-system`。

### 第 3 步：验证> ⚠️ **首次构建较慢**：PyTorch CPU 版下载约 200MB，整体镜像构建需 5-10 分钟

### 第 3 步：验证部署

1. 在 1Panel → 容器 中查看 `typhoon-system` 容器状态是否为 **运行中**
2. 浏览器访问：`http://你的服务器IP:8088`
3. 应看到台风路径预测地图界面

---

## 方式三：命令行部署（适合熟练用户）

适合想在服务器上直接用命令行操作的场景：

```bash
cd /opt/typhoon-system

# 构建并启动
docker compose up -d --build

# 查看日志
docker compose logs -f typhoon-system

# 查看容器状态
docker compose ps

# 停止服务
docker compose down

# 重启（重新构建）
docker compose up -d --build
```

---

## 部署后初始化

容器首次启动后，需要等待自动调度器完成初始化：

1. **数据缓存**：启动后约 1-2 分钟，调度器自动抓取当前年度 ISC 数据
2. **历史数据补充**：`fetch_historical_data()` 会缓存近 10 年数据，约 5-10 分钟
3. **LSTM 训练**：数据量足够后自动触发训练（需 ≥50 个数据点），约 2-5 分钟
4. **预测计算**：如有活跃台风，自动计算并缓存所有方法预测

### 手动触发初始化（可选）

如想加速初始化，SSH 到服务器后：

```bash
# 进入容器
docker exec -it typhoon-system bash

# 手动缓存历史数据
python -c "from scheduler import fetch_historical_data; fetch_historical_data()"

# 手动训练 LSTM
python -c "from scheduler import auto_train_if_needed; auto_train_if_needed(9999)"

# 手动计算预测
python -c "from scheduler import compute_active_predictions; compute_active_predictions()"
```

---

## 端口配置

默认使用 **8088** 端口。如需修改：

1. 修改 `docker-compose.yml` 中的 ports 行：
   ```yaml
   ports:
     - "你想要的端口:8088"
   ```
2. 在 1Panel 编排中修改对应行
3. 重启容器

---

## 反向代理配置（可选）

如想通过域名访问（如 `typhoon.yourdomain.com`）：

1. 在 1Panel → **网站** → **创建网站**
2. 选择 **反向代理**
3. 主域名填写你的域名
4. 代理地址填写：`http://127.0.0.1:8088`
5. SSL 证书可选配置

---

## 数据持久化说明

所有持久化数据通过 Docker volumes 映射到宿主机：

| 容器路径 | 宿主机路径 | 内容 |
|---------|----------|------|
| `/app/data/isc/` | `/opt/typhoon-system/data/isc/` | ISC 台风数据缓存 |
| `/app/data/nii/` | `/opt/typhoon-system/data/nii/` | NII 历史数据 |
| `/app/data/predictions/` | `/opt/typhoon-system/data/predictions/` | 预测结果缓存 |
| `/app/data/hashes/` | `/opt/typhoon-system/data/hashes/` | MD5 哈希变化检测 |
| `/app/backend/models/` | `/opt/typhoon-system/models/` | LSTM 权重 + Pangu ONNX |
| `/app/static/` | `/opt/typhoon-system/static/` | 前端页面 |

容器重建/升级时这些数据不会丢失。

---

## 日常运维

### 查看日志
```bash
docker compose logs -f --tail 100
```
或在 1Panel → 容器 → typhoon-system → **日志**

### 检查系统状态
浏览器访问 `http://服务器IP:8088/api/data/status` 查看：
- 缓存年份列表
- 活跃台风信息
- LSTM 模型状态

### 更新代码
```bash
cd /opt/typhoon-system
# 上传更新后的文件
docker compose up -d --build  # 重新构建并启动
```

### 数据备份
```bash
tar czf typhoon-backup-$(date +%Y%m%d).tar.gz \
    /opt/typhoon-system/data/ \
    /opt/typhoon-system/models/
```

---

## 可选：启用 Pangu-Weather 预测

如需启用盘古大模型本地推理：

1. 下载 ONNX 权重文件：
   - [pangu_weather_24.onnx](https://drive.google.com/drive/folders/1Rjhf3In8aAnEpYfUkC2j-hLvH0c1c2PX)
   - [pangu_weather_6.onnx](同上)

2. 放到宿主机目录：
   ```bash
   mkdir -p /opt/typhoon-system/models/pangu
   # 将两个 .onnx 文件复制到该目录
   ```

3. 安装 onnxruntime（进入容器）：
   ```bash
   docker exec -it typhoon-system pip install onnxruntime
   ```

4. 重启容器使模块检测生效

> ⚠️ Pangu ONNX 推理需要 ERA5/GFS 初始场数据，配置较复杂，建议仅作为可选高级功能。

---

## 故障排查

| 问题 | 排查方法 |
|------|---------|
| 容器启动失败 | `docker compose logs` 查看错误日志 |
| 端口访问不通 | 检查防火墙是否放行 8088 端口：`firewall-cmd --add-port=8088/tcp` |
| 数据抓取失败 | 检查服务器能否访问 `data.istrongcloud.com` 和 `api.open-meteo.com` |
| LSTM 训练失败 | 查看日志中训练相关报错；确认 data/isc/ 中有足够历史数据 |
| 预测结果偏差大 | 属正常现象，综合融合(Kalman)方法已整合多源数据，结果更可靠 |

---

## 1Panel 防火墙放行

1. 1Panel → **主机** → **防火墙**
2. 添加规则：端口 `8088`，协议 `TCP`，策略 `允许`
3. 或直接放行端口：
   ```bash
   # CentOS/RHEL
   firewall-cmd --permanent --add-port=8088/tcp
   firewall-cmd --reload
   # Ubuntu
   ufw allow 8088/tcp
   ```
