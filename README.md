# 🌪️ 台风路径 AI 预测系统

实时台风追踪 + 10 种 AI 预测方法 + Leaflet 深色地图可视化

数据源基于 [TropicalCyclone-Data-Parser](https://github.com/lzcapp/TropicalCyclone-Data-Parser)，整合 ISC（深圳气象创新研究院）与 NII（日本信息研究所）历史台风数据。

## 预测方法

| 方法 | 来源 | 说明 |
|------|------|------|
| 趋势外推 | 本地计算 | 线性趋势 + 加速度 |
| 物理模型 | 本地计算 | 科里奥利力 + 引导气流 |
| GFS 数值预报 | Open-Meteo API | 美国GFS全球模型 |
| ECMWF AIFS | Open-Meteo API | ECMWF AI预报系统 |
| CMA GRAPES | Open-Meteo API | 中国气象局15km分辨率 |
| GraphCast | Open-Meteo API | DeepMind AI天气模型 |
| 历史类比 | 本地计算 | DTW相似路径匹配 |
| LSTM 深度学习 | 本地训练 | 8特征+8步窗口递推预测 |
| 盘古 Pangu-Weather | ONNX本地推理 | 华为盘古天气大模型（可选） |
| **Kalman 融合** | 本地计算 | **动态权重融合所有方法** |

Kalman 融合权重：Pangu 35% / AIFS 25% / LSTM 20% / CMA 10% / GFS 8% / 其他

## 快速部署

### Docker 一键启动（推荐）

```bash
# 从 GHCR 拉取预构建镜像
docker compose -f docker-compose.deploy.yml up -d

# 或本地构建
docker compose up -d --build
```

访问 `http://localhost:8088`

### 1Panel 部署

详见 [DEPLOY_1PANEL.md](DEPLOY_1PANEL.md)

## 功能特性

- 🗺️ **深色主题地图** — Leaflet.js，台风轨迹按等级变色
- 🔮 **10 种预测路径** — 含 AI 大模型和数值预报
- 📊 **置信度指标** — 预测偏差实时标注
- 🔄 **自动数据获取** — 每小时检测 ISC 数据变化
- 🧠 **LSTM 自动训练** — 数据变化时增量训练
- 📡 **活跃台风自动预测** — 每 30 分钟刷新
- 💾 **本地数据缓存** — 近 10 年历史数据持久化

## 项目结构

```
typhoon-system/
├── backend/
│   ├── app.py              # Flask 后端服务（端口 8088）
│   ├── lstm_predictor.py   # LSTM 深度学习预测模块
│   ├── pangu_predictor.py  # Pangu-Weather ONNX 推理模块
│   └── scheduler.py        # APScheduler 自动化调度引擎
├── static/
│   └── index.html          # 前端可视化界面
├── Dockerfile              # Docker 构建（PyTorch CPU版）
├── docker-compose.yml      # 本地构建部署
├── docker-compose.deploy.yml  # 预构建镜像部署
├── .github/workflows/
│   └── docker-build.yml    # GitHub Actions CI/CD
└── DEPLOY_1PANEL.md        # 1Panel 部署指南
```

## LSTM 模型详情

- **架构**: LSTM[128] → Dropout → LSTM[64] → Dropout → Dense[32] → Dense[2]
- **输入**: 8 维特征 (lat, lng, pressure, wind_speed, move_speed, move_dir_sin/cos, month)
- **窗口**: 8 步滑动窗口 (≈24h)
- **训练结果**: 80 个台风，4347 个序列，best_val_loss = 4.22e-05，估计误差 ≈36km
- **DTW 优化**: 动态时间规整筛选相似路径训练集

## 可选：启用 Pangu-Weather

1. 下载 ONNX 权重到 `backend/models/pangu/`:
   - [pangu_weather_24.onnx](https://drive.google.com/drive/folders/1Rjhf3In8aAnEpYfUkC2j-hLvH0c1c2PX)
   - [pangu_weather_6.onnx](同上)
2. 安装 `onnxruntime`
3. 准备 ERA5/GFS 初始场数据

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| TZ | Asia/Shanghai | 时区 |
| PYTHONUNBUFFERED | 1 | Python 无缓冲输出 |

## API 端点

| 端点 | 说明 |
|------|------|
| `/api/typhoons/{year}` | 获取指定年份台风列表 |
| `/api/typhoons/detail/{id}` | 台风详情 |
| `/api/typhoons/predict/{id}` | 预测路径 |
| `/api/data/status` | 数据缓存状态 |
| `/api/lstm/status` | LSTM 模型状态 |
| `/api/lstm/train` | 触发 LSTM 训练 |
| `/api/scheduler/status` | 调度器状态 |
| `/api/prediction-methods` | 可用预测方法列表 |

## License

MIT

## 致谢

- [ISC](https://data.istrongcloud.com/) — 台风数据
- [NII Digital Typhoon](https://agora.ex.nii.ac.jp/digital-typhoon/) — 历史数据
- [Open-Meteo](https://open-meteo.com/) — 天气预报 API
- [Pangu-Weather](https://nature.com/articles/s41586-022-05238-y) — 华为盘古大模型
- [TropicalCyclone-Data-Parser](https://github.com/lzcapp/TropicalCyclone-Data-Parser) — 数据解析参考
