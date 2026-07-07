"""
台风系统自动化调度引擎

功能:
1. 定时自动获取数据 (每小时检查ISC新数据)
2. 数据变化检测 (对比本地缓存与远程数据)
3. 变化时自动触发:
   - LSTM 模型增量训练
   - 活跃台风预测计算
   - 预测结果缓存供前端快速查询
4. 服务器启动时自动初始化

部署方式:
- 嵌入Flask进程: 由APScheduler在后台运行
- 独立进程: python scheduler.py (适合生产环境分离部署)

配置:
  SCHEDULE_INTERVAL_DATA_FETCH = 3600  # 数据获取间隔(秒)
  SCHEDULE_INTERVAL_PREDICTION = 1800  # 预测计算间隔(秒)
  SCHEDULE_INTERVAL_TRAINING = 86400   # 训练间隔(秒, 每天1次)
  AUTO_TRAIN_THRESHOLD = 5             # 新数据点数超过此阈值才触发重训练
"""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta

# ============================================================
# 配置
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ISC_DIR = os.path.join(DATA_DIR, 'isc')
PREDICTION_CACHE_DIR = os.path.join(DATA_DIR, 'predictions')
HASH_DIR = os.path.join(DATA_DIR, 'hashes')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

FETCH_INTERVAL = 3600      # 数据获取间隔(秒), 1小时
PREDICTION_INTERVAL = 1800  # 预测计算间隔(秒), 30分钟
TRAINING_INTERVAL = 86400   # 训练间隔(秒), 24小时
AUTO_TRAIN_THRESHOLD = 50   # 新增数据点超过此数触发重训练
MAX_PREDICTION_HOURS = 120  # 最大预测时长

for d in [ISC_DIR, PREDICTION_CACHE_DIR, HASH_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 数据获取
# ============================================================

import requests as req_lib

HEADERS = {
    'Connection': 'Keep-Alive',
    'Accept': 'text/html, application/xhtml+xml, */*',
    'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def fetch_current_data():
    """获取当前月份和前后月份的ISC数据"""
    now = datetime.now()
    months_to_fetch = []

    # 当前月 + 前3个月 + 后1个月(预报)
    for offset in [-3, -2, -1, 0, 1]:
        dt = now + timedelta(days=offset * 30)
        ym = dt.strftime('%Y%m')
        months_to_fetch.append(ym)

    changes_detected = 0
    new_points_count = 0

    for ym in months_to_fetch:
        local_file = os.path.join(ISC_DIR, f'{ym}.json')
        url = f'https://data.istrongcloud.com/v2/data/complex/{ym}.json'

        # 获取远程数据
        try:
            response = req_lib.get(url, headers=HEADERS, timeout=20)
            if response.status_code != 200:
                continue
            remote_data = response.json()
        except Exception as e:
            print(f"[Scheduler] Fetch error {ym}: {e}")
            continue

        # 检测数据变化
        remote_hash = hashlib.md5(json.dumps(remote_data, sort_keys=True).encode()).hexdigest()
        hash_file = os.path.join(HASH_DIR, f'{ym}.hash')

        old_hash = ''
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                old_hash = f.read().strip()

        if remote_hash == old_hash and os.path.exists(local_file):
            # 数据无变化
            continue

        # 数据有变化!
        changes_detected += 1

        # 计算新增数据点数
        old_points = 0
        if os.path.exists(local_file):
            try:
                with open(local_file, 'r') as f:
                    old_data = json.load(f)
                for t in old_data:
                    old_points += len(t.get('points', []))
            except:
                old_points = 0

        new_points = 0
        for t in remote_data:
            new_points += len(t.get('points', []))
        new_points_count += (new_points - old_points)

        # 更新本地数据
        with open(local_file, 'w') as f:
            json.dump(remote_data, f)

        # 更新哈希
        with open(hash_file, 'w') as f:
            f.write(remote_hash)

        print(f"[Scheduler] 数据更新 {ym}: {new_points}点 (变化+{new_points - old_points})")

    return changes_detected, new_points_count


def fetch_historical_data():
    """补充缓存历史数据（启动时运行一次）"""
    current_year = datetime.now().year

    # 确保近10年数据已缓存
    for year in range(current_year - 9, current_year + 1):
        for month in range(1, 13):
            ym = f"{year}{str(month).zfill(2)}"
            local_file = os.path.join(ISC_DIR, f'{ym}.json')

            if os.path.exists(local_file):
                continue

            url = f'https://data.istrongcloud.com/v2/data/complex/{ym}.json'
            try:
                response = req_lib.get(url, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    with open(local_file, 'w') as f:
                        json.dump(data, f)
                    # 同时记录哈希
                    h = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
                    with open(os.path.join(HASH_DIR, f'{ym}.hash'), 'w') as f:
                        f.write(h)
            except:
                pass  # 部分历史数据可能不可用，忽略

    # 检查缓存状态
    cached_count = len([f for f in os.listdir(ISC_DIR) if f.endswith('.json')])
    print(f"[Scheduler] 历史数据缓存完成: {cached_count}个月文件")


# ============================================================
# 自动预测计算
# ============================================================

def compute_active_predictions():
    """为所有活跃台风计算预测并缓存结果"""
    now = datetime.now()
    ym = now.strftime('%Y%m')
    local_file = os.path.join(ISC_DIR, f'{ym}.json')

    if not os.path.exists(local_file):
        print("[Scheduler] 无当前月数据，跳过预测")
        return

    with open(local_file, 'r') as f:
        data = json.load(f)

    # 找活跃台风
    active_typhoons = [t for t in data if t.get('is_current') == 1]

    # 也检查前月
    prev_ym = (now - timedelta(days=30)).strftime('%Y%m')
    prev_file = os.path.join(ISC_DIR, f'{prev_ym}.json')
    if os.path.exists(prev_file):
        with open(prev_file, 'r') as f:
            prev_data = json.load(f)
        for t in prev_data:
            if t.get('is_current') == 1 and t['tfbh'] not in [a['tfbh'] for a in active_typhoons]:
                active_typhoons.append(t)

    if not active_typhoons:
        print("[Scheduler] 无活跃台风，跳过预测")
        return

    print(f"[Scheduler] 发现 {len(active_typhoons)} 个活跃台风，开始计算预测")

    # 导入预测模块（延迟导入避免循环依赖）
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from app import TyphoonPredictor, normalize_isc_data
    from lstm_predictor import is_lstm_ready, lstm_predict, WINDOW_SIZE

    for t in active_typhoons:
        tfid = t.get('tfbh', '')
        if not tfid:
            continue

        # 准备标准化数据
        norm_data = normalize_isc_data([t])
        if not norm_data:
            continue
        typhoon = norm_data[0]
        points = typhoon.get('points', [])
        if len(points) < 2:
            continue

        last_point = points[-1]

        # 为每个预测时长计算结果
        for hours in [24, 48, 72, 120]:
            predictions = {}

            # LSTM预测
            if is_lstm_ready() and len(points) >= WINDOW_SIZE:
                lstm_result = lstm_predict(points, hours)
                if lstm_result:
                    predictions['lstm'] = lstm_result

            # GFS预报
            gfs_pred = TyphoonPredictor._gfs_forecast(last_point, hours)
            if gfs_pred:
                predictions['gfs'] = gfs_pred

            # AIFS预报
            aifs_pred = TyphoonPredictor._aifs_prediction(last_point, hours)
            if aifs_pred:
                predictions['aifs'] = aifs_pred

            # CMA预报
            cma_pred = TyphoonPredictor._cma_prediction(last_point, hours)
            if cma_pred:
                predictions['cma'] = cma_pred

            # Kalman融合
            if len(predictions) >= 2:
                ensemble = TyphoonPredictor._kalman_ensemble(predictions, hours, last_point)
                if ensemble:
                    predictions['ensemble'] = ensemble

            # 机构预报
            forecasts = t.get('forecast', {})
            if forecasts:
                typhoon['forecasts'] = forecasts

            # 缓存结果
            cache_file = os.path.join(PREDICTION_CACHE_DIR, f'{tfid}_{hours}h.json')
            result = {
                'typhoon_id': tfid,
                'name_cn': typhoon.get('name_cn', ''),
                'name_en': typhoon.get('name_en', ''),
                'hours': hours,
                'base_time': last_point.get('time', ''),
                'base_lat': last_point.get('lat', 0),
                'base_lng': last_point.get('lng', 0),
                'predictions': predictions,
                'computed_at': now.isoformat(),
                'active': True,
            }

            with open(cache_file, 'w') as f:
                json.dump(result, f)

            method_count = len(predictions)
            print(f"[Scheduler] 预测缓存 {tfid} {hours}h: {method_count}种方法")

    print(f"[Scheduler] 预测计算完成: {len(active_typhoons)}个活跃台风")


# ============================================================
# 自动训练
# ============================================================

def auto_train_if_needed(new_points_count):
    """检查是否需要重新训练LSTM模型"""
    from lstm_predictor import LSTMTrainer, is_lstm_ready

    if new_points_count < AUTO_TRAIN_THRESHOLD:
        print(f"[Scheduler] 新增{new_points_count}点 < 阈值{AUTO_TRAIN_THRESHOLD}, 跳过训练")
        return

    # 获取上次训练时间
    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    last_train_time = 0
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        last_train_time = history.get('last_train_timestamp', 0)

    # 至少间隔12小时才重训练
    if time.time() - last_train_time < 43200:
        print("[Scheduler] 训练间隔不足12h, 跳过")
        return

    print(f"[Scheduler] 新增{new_points_count}点数据, 开始增量训练LSTM")

    trainer = LSTMTrainer()
    current_year = datetime.now().year
    years = list(range(current_year - 9, current_year + 1))

    X, Y, typhoon_count = trainer.load_training_data(years)
    if X is None or typhoon_count < 10:
        print("[Scheduler] 训练数据不足, 跳过")
        return

    history = trainer.train(X, Y, epochs=40, batch_size=32, lr=0.001, val_ratio=0.15)

    # 记录训练时间
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            full_history = json.load(f)
        full_history['last_train_timestamp'] = time.time()
        full_history['last_train_points'] = new_points_count
        with open(history_path, 'w') as f:
            json.dump(full_history, f)

    best_loss = history.get('best_val_loss', 0)
    error_km = math.sqrt(best_loss) * 50 * 111
    print(f"[Scheduler] LSTM增量训练完成: 误差≈{round(error_km)}km, {history.get('epochs_trained', 0)}轮")


# ============================================================
# 调度器主循环
# ============================================================

def run_scheduler_loop():
    """独立进程调度器主循环（适合生产环境）"""
    print("=" * 60)
    print("[Scheduler] 台风系统自动化调度引擎启动")
    print("=" * 60)

    # 初始化: 缓存历史数据
    print("[Scheduler] 初始化: 缓存历史数据...")
    fetch_historical_data()

    # 初始化: 训练LSTM（如果模型不存在）
    from lstm_predictor import is_lstm_ready, LSTMTrainer
    if not is_lstm_ready():
        print("[Scheduler] LSTM模型不存在, 首次训练...")
        trainer = LSTMTrainer()
        X, Y, count = trainer.load_training_data()
        if X is not None:
            trainer.train(X, Y, epochs=80)
            print("[Scheduler] LSTM首次训练完成")

    # 初始化: 计算活跃台风预测
    print("[Scheduler] 初始化: 计算活跃台风预测...")
    compute_active_predictions()

    # 主循环
    last_fetch_time = time.time()
    last_pred_time = time.time()
    last_train_time = time.time()

    print("[Scheduler] 进入主循环 (数据获取1h / 预测30min / 训练24h)")

    while True:
        now = time.time()

        # 定时数据获取
        if now - last_fetch_time >= FETCH_INTERVAL:
            print("[Scheduler] === 定时数据获取 ===")
            changes, new_points = fetch_current_data()
            last_fetch_time = now

            if changes > 0:
                # 数据变化 → 立即触发预测计算
                print(f"[Scheduler] 数据变化! {changes}个月更新, {new_points}新点")
                compute_active_predictions()

                # 检查是否需要重训练
                auto_train_if_needed(new_points)

        # 定时预测计算（即使数据没变化也定期刷新）
        elif now - last_pred_time >= PREDICTION_INTERVAL:
            print("[Scheduler] === 定时预测刷新 ===")
            compute_active_predictions()
            last_pred_time = now

        # 定时训练（每天检查一次）
        elif now - last_train_time >= TRAINING_INTERVAL:
            # 不一定真的训练，只是检查一下
            print("[Scheduler] === 定时训练检查 ===")
            auto_train_if_needed(AUTO_TRAIN_THRESHOLD + 1)  # 强制检查
            last_train_time = now

        # 等待30秒
        time.sleep(30)


# ============================================================
# Flask嵌入版本
# ============================================================

def setup_flask_scheduler(app):
    """在Flask应用中嵌入APScheduler后台任务"""
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(daemon=True)

    # 每小时获取数据
    scheduler.add_job(
        func=_scheduled_data_fetch,
        trigger='interval',
        seconds=FETCH_INTERVAL,
        id='data_fetch',
        name='定时数据获取',
    )

    # 每30分钟刷新预测
    scheduler.add_job(
        func=_scheduled_prediction_refresh,
        trigger='interval',
        seconds=PREDICTION_INTERVAL,
        id='prediction_refresh',
        name='定时预测刷新',
    )

    # 每天3:00检查训练
    scheduler.add_job(
        func=_scheduled_training_check,
        trigger='cron',
        hour=3,
        minute=0,
        id='training_check',
        name='定时训练检查',
    )

    scheduler.start()
    print("[Scheduler] APScheduler已启动 (嵌入Flask)")
    return scheduler


def _scheduled_data_fetch():
    """APScheduler: 定时数据获取"""
    try:
        changes, new_points = fetch_current_data()
        if changes > 0:
            print(f"[APScheduler] 数据变化: {changes}月, {new_points}点 → 触发预测")
            compute_active_predictions()
            auto_train_if_needed(new_points)
    except Exception as e:
        print(f"[APScheduler] 数据获取异常: {e}")


def _scheduled_prediction_refresh():
    """APScheduler: 定时预测刷新"""
    try:
        compute_active_predictions()
    except Exception as e:
        print(f"[APScheduler] 预测刷新异常: {e}")


def _scheduled_training_check():
    """APScheduler: 每天3点训练检查"""
    try:
        auto_train_if_needed(AUTO_TRAIN_THRESHOLD + 1)
    except Exception as e:
        print(f"[APScheduler] 训练检查异常: {e}")


# ============================================================
# 预测缓存读取 API
# ============================================================

def get_cached_prediction(typhoon_id, hours):
    """从缓存中读取已计算的预测结果（前端快速查询）"""
    cache_file = os.path.join(PREDICTION_CACHE_DIR, f'{typhoon_id}_{hours}h.json')

    if not os.path.exists(cache_file):
        return None

    with open(cache_file, 'r') as f:
        data = json.load(f)

    # 检查缓存时效（超过1小时则标记为过期但仍可用）
    computed_at = data.get('computed_at', '')
    try:
        compute_time = datetime.fromisoformat(computed_at)
        age_minutes = (datetime.now() - compute_time).total_seconds() / 60
        data['cache_age_minutes'] = round(age_minutes)
        data['cache_fresh'] = age_minutes < 60
    except:
        data['cache_age_minutes'] = 999
        data['cache_fresh'] = False

    return data


# ============================================================
# CLI入口
# ============================================================

if __name__ == '__main__':
    run_scheduler_loop()
