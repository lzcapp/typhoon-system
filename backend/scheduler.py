"""
台风系统自动化调度引擎 v2

架构: 后台持续更新 → 缓存预测结果 → 前端秒级读取

功能:
1. 定时自动获取数据 (每小时检查ISC新数据)
2. 数据变化检测 (对比本地缓存与远程数据)
3. 变化时自动触发:
   - ECMWF BUFR官方轨迹获取
   - 活跃台风全方法预测计算 (trend/physics/gfs/ecmwf/aifs/cma/ecmwf_bufr/pangu/ensemble)
   - 预测结果缓存供前端快速查询
   - LSTM模型增量训练
4. 用户选择台风时触发按需缓存计算
5. 服务器启动时自动初始化

部署方式:
- 嵌入Flask进程: 由APScheduler在后台运行
- 独立进程: python scheduler.py (适合生产环境分离部署)

配置:
  FETCH_INTERVAL = 3600      # 数据获取间隔(秒)
  PREDICTION_INTERVAL = 1800  # 预测计算间隔(秒)
  TRAINING_INTERVAL = 86400   # 训练间隔(秒, 每天1次)
  PREDICTION_HOURS = [24, 48, 72, 120, 168, 240]  # 缓存的预测时长
"""

import hashlib
import json
import math
import os
import threading
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
PREDICTION_HOURS = [24, 48, 72, 120, 168, 240]  # 缓存所有时长
CACHE_STALE_MINUTES = 30    # 缓存超过30分钟视为过时

# ★ 后台计算使用重型方法（因为后台计算不怕等待）
# 'all' = 全部方法(含Pangu等重型模型)，'all_fast' = 快速方法(不含Pangu)
BACKGROUND_METHOD = 'all'   # 后台默认使用所有方法（包括Pangu-Weather）
BACKGROUND_STALE_MINUTES = 120  # 后台重型计算结果缓存2小时视为过时（更长的新鲜期）

for d in [ISC_DIR, PREDICTION_CACHE_DIR, HASH_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 运行状态记录
# ============================================================

_scheduler_state = {
    'last_data_fetch': None,
    'last_prediction': None,
    'last_training': None,
    'last_ecmwf_bufr': None,
    'last_method': '',
    'active_typhoon_count': 0,
    'cached_predictions_count': 0,
    'on_demand_queue_size': 0,
    'startup_init_done': False,
    'errors': [],
}

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

    _scheduler_state['last_data_fetch'] = datetime.now().isoformat()
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
# ECMWF BUFR 数据获取
# ============================================================

def fetch_ecmwf_bufr_data():
    """定期获取ECMWF BUFR官方台风轨迹数据"""
    try:
        from ecmwf_bufr_fetcher import get_ecmwf_active_storms, fetch_ecmwf_tracks_for_typhoon
        # 获取ECMWF追踪的所有活跃热带气旋
        active_storms = get_ecmwf_active_storms()
        if active_storms and active_storms.get('storms'):
            storm_count = len(active_storms['storms'])
            print(f"[Scheduler] ECMWF BUFR: 发现{storm_count}个活跃热带气旋")

            # 为每个风暴获取详细轨迹数据并缓存
            for storm in active_storms['storms']:
                storm_id = storm.get('id', '')
                name = storm.get('name', '')
                try:
                    tracks = fetch_ecmwf_tracks_for_typhoon(storm_id)
                    if tracks:
                        print(f"[Scheduler] ECMWF BUFR: 获取{name or storm_id}轨迹成功")
                except Exception as e:
                    print(f"[Scheduler] ECMWF BUFR: 获取{name or storm_id}轨迹失败: {e}")
        else:
            print("[Scheduler] ECMWF BUFR: 无活跃热带气旋数据")
        _scheduler_state['last_ecmwf_bufr'] = datetime.now().isoformat()
        return active_storms
    except ImportError:
        print("[Scheduler] ECMWF BUFR: ecmwf_bufr_fetcher模块未安装，跳过")
        _scheduler_state['errors'].append("BUFR: 模块未安装")
        return None
    except Exception as e:
        print(f"[Scheduler] ECMWF BUFR获取异常: {e}")
        _scheduler_state['errors'].append(f"BUFR: {str(e)[:80]}")
        return None


# ============================================================
# 自动预测计算 (核心 - 使用predict_path统一入口)
# ============================================================

def _get_typhoon_data(tfid):
    """根据台风编号获取完整的原始数据+标准化数据"""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from app import normalize_isc_data

    year = int(tfid[:4])
    typhoon_raw = None

    # 搜索全年月份数据
    for month in range(1, 13):
        ym = f"{year}{str(month).zfill(2)}"
        month_file = os.path.join(ISC_DIR, f'{ym}.json')
        if not os.path.exists(month_file):
            continue
        try:
            with open(month_file, 'r') as f:
                month_data = json.load(f)
            for t in month_data:
                if t.get('tfbh') == tfid or t.get('ident') == tfid:
                    typhoon_raw = t
                    break
            if typhoon_raw:
                break
        except:
            continue

    if not typhoon_raw:
        return None, None

    # 标准化数据
    norm_data = normalize_isc_data([typhoon_raw])
    if not norm_data:
        return typhoon_raw, None

    return typhoon_raw, norm_data[0]


def compute_predictions_for_typhoon(tfid, hours_list=None, method='all'):
    """
    为单个台风计算预测并缓存结果
    使用 predict_path 统一入口，包含所有方法

    Args:
        tfid: 台风编号
        hours_list: 预测时长列表，默认 PREDICTION_HOURS
        method: 预测方法，默认 'all' (全部方法)
    """
    if hours_list is None:
        hours_list = PREDICTION_HOURS

    typhoon_raw, typhoon = _get_typhoon_data(tfid)
    if not typhoon:
        print(f"[Scheduler] 台风 {tfid} 数据未找到，跳过")
        return 0

    points = typhoon.get('points', [])
    if len(points) < 2:
        print(f"[Scheduler] 台风 {tfid} 数据点不足，跳过")
        return 0

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from app import TyphoonPredictor
    from lstm_predictor import is_lstm_ready, lstm_predict, WINDOW_SIZE

    now = datetime.now()
    last_point = points[-1]
    cached_count = 0

    for hours in hours_list:
        # ★ 核心: 使用 predict_path(method='all') 统一入口
        # 这样包含 trend/physics/gfs/ecmwf/aifs/cma/ecmwf_bufr/pangu 等所有方法
        result = TyphoonPredictor.predict_path(typhoon, hours=hours, method=method)

        if not result or not result.get('predictions'):
            # all方法失败时回退到 ensemble
            result = TyphoonPredictor.predict_path(typhoon, hours=hours, method='ensemble')

        if not result or not result.get('predictions'):
            # ensemble失败时回退到 trend
            result = TyphoonPredictor.predict_path(typhoon, hours=hours, method='trend')

        if result and result.get('predictions'):
            # 补充元信息
            result['typhoon_id'] = tfid
            result['name_cn'] = typhoon.get('name_cn', '')
            result['name_en'] = typhoon.get('name_en', '')
            result['hours'] = hours
            result['base_time'] = last_point.get('time', '')
            result['base_lat'] = last_point.get('lat', 0)
            result['base_lng'] = last_point.get('lng', 0)
            result['computed_at'] = now.isoformat()
            result['cache_source'] = 'scheduler'

            # 缓存结果
            cache_file = os.path.join(PREDICTION_CACHE_DIR, f'{tfid}_{hours}h.json')
            with open(cache_file, 'w') as f:
                json.dump(result, f)

            method_count = len(result.get('predictions', {}))
            cached_count += 1
            print(f"[Scheduler] 缓存 {tfid} {hours}h: {method_count}种方法")

    return cached_count


def compute_active_predictions(method=None):
    """为所有活跃台风计算预测并缓存结果（后台自动执行，不等用户选择）"""
    if method is None:
        method = BACKGROUND_METHOD  # ★ 后台使用重型方法（含Pangu）

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
        try:
            with open(prev_file, 'r') as f:
                prev_data = json.load(f)
            for t in prev_data:
                if t.get('is_current') == 1 and t['tfbh'] not in [a['tfbh'] for a in active_typhoons]:
                    active_typhoons.append(t)
        except:
            pass

    # ★ 也检查前后2个月（更全面的活跃台风检测）
    for offset in [-2, 2]:
        other_ym = (now + timedelta(days=offset * 30)).strftime('%Y%m')
        other_file = os.path.join(ISC_DIR, f'{other_ym}.json')
        if os.path.exists(other_file):
            try:
                with open(other_file, 'r') as f:
                    other_data = json.load(f)
                for t in other_data:
                    if t.get('is_current') == 1 and t['tfbh'] not in [a['tfbh'] for a in active_typhoons]:
                        active_typhoons.append(t)
            except:
                pass

    _scheduler_state['active_typhoon_count'] = len(active_typhoons)

    if not active_typhoons:
        print("[Scheduler] 无活跃台风，跳过预测")
        _scheduler_state['last_prediction'] = now.isoformat()
        return

    active_ids = [t.get('tfbh', '') for t in active_typhoons if t.get('tfbh')]
    print(f"[Scheduler] 发现 {len(active_typhoons)} 个活跃台风: {active_ids}")
    print(f"[Scheduler] 使用方法: {method} (后台重型计算)")

    total_cached = 0
    for t in active_typhoons:
        tfid = t.get('tfbh', '')
        if not tfid:
            continue
        # ★ 后台计算所有时长 + 重型方法
        cached = compute_predictions_for_typhoon(tfid, method=method)
        total_cached += cached

    _scheduler_state['cached_predictions_count'] = len(
        [f for f in os.listdir(PREDICTION_CACHE_DIR) if f.endswith('.json')]
    )
    _scheduler_state['last_prediction'] = now.isoformat()
    _scheduler_state['last_method'] = method
    print(f"[Scheduler] 预测计算完成: {len(active_typhoons)}台风, {total_cached}缓存文件, 方法={method}")


def compute_on_demand(tfid, hours=168, method='all'):
    """
    按需计算：当用户选择一个没有缓存的台风时触发后台计算
    使用线程池避免阻塞Flask请求

    Args:
        tfid: 台风编号
        hours: 预测时长
        method: 预测方法
    """
    # 检查缓存是否已存在且新鲜
    cache_file = os.path.join(PREDICTION_CACHE_DIR, f'{tfid}_{hours}h.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            computed_at = data.get('computed_at', '')
            compute_time = datetime.fromisoformat(computed_at)
            age_minutes = (datetime.now() - compute_time).total_seconds() / 60
            if age_minutes < CACHE_STALE_MINUTES:
                print(f"[Scheduler] 按需计算跳过: {tfid} {hours}h 缓存新鲜({round(age_minutes)}分钟)")
                return  # 缓存新鲜，无需重算
        except:
            pass

    # 在后台线程中计算，避免阻塞前端请求
    _scheduler_state['on_demand_queue_size'] = (
        _scheduler_state.get('on_demand_queue_size', 0) + 1
    )

    def _background_compute():
        try:
            cached = compute_predictions_for_typhoon(tfid, hours_list=[hours], method=method)
            print(f"[Scheduler] 按需计算完成: {tfid} {hours}h, {cached}缓存")
        except Exception as e:
            print(f"[Scheduler] 按需计算异常: {tfid} {e}")
            _scheduler_state['errors'].append(f"on_demand {tfid}: {str(e)[:80]}")
        finally:
            _scheduler_state['on_demand_queue_size'] = (
                _scheduler_state.get('on_demand_queue_size', 0) - 1
            )

    thread = threading.Thread(target=_background_compute, daemon=True)
    thread.start()
    print(f"[Scheduler] 按需计算已启动: {tfid} {hours}h (后台线程)")


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
    _scheduler_state['last_training'] = datetime.now().isoformat()
    print(f"[Scheduler] LSTM增量训练完成: 误差≈{round(error_km)}km, {history.get('epochs_trained', 0)}轮")


# ============================================================
# 调度器主循环
# ============================================================

def run_scheduler_loop():
    """独立进程调度器主循环（适合生产环境）"""
    print("=" * 60)
    print("[Scheduler] 台风系统自动化调度引擎 v2 启动")
    print("  后台持续计算 → 缓存结果 → 前端秒级读取")
    print("=" * 60)

    # 初始化: 缓存历史数据
    print("[Scheduler] 初始化: 缓存历史数据...")
    fetch_historical_data()

    # 初始化: 获取ECMWF BUFR数据
    print("[Scheduler] 初始化: 获取ECMWF BUFR官方轨迹...")
    fetch_ecmwf_bufr_data()

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
    print("[Scheduler] 初始化: 全方法预测计算...")
    compute_active_predictions()

    # 主循环
    last_fetch_time = time.time()
    last_pred_time = time.time()
    last_train_time = time.time()
    last_bufr_time = time.time()

    print("[Scheduler] 进入主循环 (数据1h/预测30min/BUFR6h/训练24h)")

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

        # 定时获取ECMWF BUFR (每6小时)
        elif now - last_bufr_time >= 6 * 3600:
            print("[Scheduler] === 定时ECMWF BUFR获取 ===")
            fetch_ecmwf_bufr_data()
            # BUFR数据更新后也刷新预测(因为ecmwf_bufr方法依赖此数据)
            compute_active_predictions()
            last_bufr_time = now

        # 定时训练（每天检查一次）
        elif now - last_train_time >= TRAINING_INTERVAL:
            print("[Scheduler] === 定时训练检查 ===")
            auto_train_if_needed(AUTO_TRAIN_THRESHOLD + 1)  # 强制检查
            last_train_time = now

        # 等待30秒
        time.sleep(30)


# ============================================================
# Flask嵌入版本
# ============================================================

def startup_init():
    """
    ★ 启动时立即初始化：获取数据 + 计算所有活跃台风预测
    在后台线程中运行，不阻塞Flask启动
    """
    print("=" * 60)
    print("[Scheduler] ★ 启动初始化开始（后台线程）")
    print("  自动计算所有活跃台风 → 缓存 → 前端秒级读取")
    print("=" * 60)

    # Step 1: 获取当前数据
    print("[Scheduler] 初始化 Step 1: 获取当前数据...")
    try:
        fetch_current_data()
    except Exception as e:
        print(f"[Scheduler] 数据获取异常: {e}")

    # Step 2: 获取ECMWF BUFR数据（如可用）
    print("[Scheduler] 初始化 Step 2: 获取ECMWF BUFR官方轨迹...")
    try:
        fetch_ecmwf_bufr_data()
    except Exception as e:
        print(f"[Scheduler] BUFR获取异常: {e}")

    # Step 3: LSTM模型检查（如未训练则首次训练）
    print("[Scheduler] 初始化 Step 3: LSTM模型检查...")
    try:
        from lstm_predictor import is_lstm_ready, LSTMTrainer
        if not is_lstm_ready():
            print("[Scheduler] LSTM模型不存在, 首次训练...")
            trainer = LSTMTrainer()
            X, Y, count = trainer.load_training_data()
            if X is not None:
                trainer.train(X, Y, epochs=80)
                print("[Scheduler] LSTM首次训练完成")
    except Exception as e:
        print(f"[Scheduler] LSTM训练异常: {e}")

    # Step 4: ★ 核心步骤 — 计算所有活跃台风预测（使用重型方法）
    print("[Scheduler] 初始化 Step 4: 全活跃台风预测计算（重型方法）...")
    try:
        compute_active_predictions(method=BACKGROUND_METHOD)
    except Exception as e:
        print(f"[Scheduler] 活跃台风预测异常: {e}")

    print("=" * 60)
    print("[Scheduler] ★ 启动初始化完成！前端可立即读取缓存")
    print("=" * 60)
    _scheduler_state['startup_init_done'] = True


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

    # 每30分钟刷新预测（使用重型方法）
    scheduler.add_job(
        func=_scheduled_prediction_refresh,
        trigger='interval',
        seconds=PREDICTION_INTERVAL,
        id='prediction_refresh',
        name='定时预测刷新(重型方法)',
    )

    # 每6小时获取ECMWF BUFR
    scheduler.add_job(
        func=_scheduled_bufr_fetch,
        trigger='interval',
        seconds=6 * 3600,
        id='bufr_fetch',
        name='定时ECMWF BUFR获取',
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
    print("[Scheduler] APScheduler v2已启动 (数据1h/预测30min/BUFR6h/训练24h)")

    # ★ 启动后立即在后台线程初始化（不等第一个定时周期）
    init_thread = threading.Thread(target=startup_init, daemon=True)
    init_thread.start()
    print("[Scheduler] 启动初始化已在后台线程启动，约1-5分钟后缓存可用")

    return scheduler


def _scheduled_data_fetch():
    """APScheduler: 定时数据获取"""
    try:
        changes, new_points = fetch_current_data()
        if changes > 0:
            print(f"[APScheduler] 数据变化: {changes}月, {new_points}点 → 触发重型方法预测")
            compute_active_predictions(method=BACKGROUND_METHOD)
            auto_train_if_needed(new_points)
    except Exception as e:
        print(f"[APScheduler] 数据获取异常: {e}")
        _scheduler_state['errors'].append(f"data_fetch: {str(e)[:80]}")


def _scheduled_prediction_refresh():
    """APScheduler: 定时预测刷新（重型方法）"""
    try:
        compute_active_predictions(method=BACKGROUND_METHOD)
    except Exception as e:
        print(f"[APScheduler] 预测刷新异常: {e}")
        _scheduler_state['errors'].append(f"prediction: {str(e)[:80]}")


def _scheduled_bufr_fetch():
    """APScheduler: 定时ECMWF BUFR获取"""
    try:
        fetch_ecmwf_bufr_data()
        # BUFR更新后刷新预测
        compute_active_predictions()
    except Exception as e:
        print(f"[APScheduler] BUFR获取异常: {e}")
        _scheduler_state['errors'].append(f"bufr: {str(e)[:80]}")


def _scheduled_training_check():
    """APScheduler: 每天3点训练检查"""
    try:
        auto_train_if_needed(AUTO_TRAIN_THRESHOLD + 1)
    except Exception as e:
        print(f"[APScheduler] 训练检查异常: {e}")
        _scheduler_state['errors'].append(f"training: {str(e)[:80]}")


# ============================================================
# 预测缓存读取 API
# ============================================================

def get_cached_prediction(typhoon_id, hours, method=None):
    """
    从缓存中读取已计算的预测结果（前端快速查询）

    Args:
        typhoon_id: 台风编号
        hours: 预测时长
        method: 如果指定，只返回该方法的预测（可选）

    Returns:
        dict with predictions + cache metadata, or None if no cache
    """
    cache_file = os.path.join(PREDICTION_CACHE_DIR, f'{typhoon_id}_{hours}h.json')

    if not os.path.exists(cache_file):
        return None

    with open(cache_file, 'r') as f:
        data = json.load(f)

    # 检查缓存时效
    computed_at = data.get('computed_at', '')
    try:
        compute_time = datetime.fromisoformat(computed_at)
        age_minutes = (datetime.now() - compute_time).total_seconds() / 60
        data['cache_age_minutes'] = round(age_minutes)
        data['cache_fresh'] = age_minutes < CACHE_STALE_MINUTES
    except:
        data['cache_age_minutes'] = 999
        data['cache_fresh'] = False

    # 如果指定了方法，只返回该方法的预测
    if method and method != 'all':
        all_predictions = data.get('predictions', {})
        if method in all_predictions:
            data['predictions'] = {method: all_predictions[method]}
        elif method == 'ensemble' and 'ensemble' in all_predictions:
            data['predictions'] = {'ensemble': all_predictions['ensemble']}
        elif method == 'trend' and 'trend' in all_predictions:
            data['predictions'] = {'trend': all_predictions['trend']}

    return data


def get_scheduler_status():
    """获取调度引擎运行状态"""
    pred_files = [f for f in os.listdir(PREDICTION_CACHE_DIR) if f.endswith('.json')] \
        if os.path.exists(PREDICTION_CACHE_DIR) else []
    data_files = [f for f in os.listdir(ISC_DIR) if f.endswith('.json')] \
        if os.path.exists(ISC_DIR) else []
    hash_files = [f for f in os.listdir(HASH_DIR) if f.endswith('.hash')] \
        if os.path.exists(HASH_DIR) else []

    # 最近一次预测时间
    latest_pred_time = ''
    if pred_files:
        latest_file = os.path.join(PREDICTION_CACHE_DIR, max(pred_files))
        try:
            with open(latest_file, 'r') as f:
                d = json.load(f)
            latest_pred_time = d.get('computed_at', '')
        except:
            pass

    from lstm_predictor import is_lstm_ready

    # Pangu-Weather 状态
    pangu_ready = False
    try:
        from pangu_predictor import is_pangu_ready
        pangu_ready = is_pangu_ready()
    except:
        pass

    # ECMWF BUFR 状态
    bufr_available = False
    try:
        from ecmwf_bufr_fetcher import is_bufr_available
        bufr_available = is_bufr_available()
    except:
        pass

    # ★ 活跃台风缓存覆盖情况
    active_coverage = get_active_cache_coverage()

    return {
        'scheduler_running': True,
        'version': 'v2',
        'architecture': '后台持续计算 → 缓存 → 前端秒级读取',
        'background_method': BACKGROUND_METHOD,
        'data_files_count': len(data_files),
        'hash_files_count': len(hash_files),
        'cached_predictions_count': len(pred_files),
        'cached_prediction_files': pred_files[:10],
        'latest_prediction_time': latest_pred_time,
        'lstm_model_ready': is_lstm_ready(),
        'pangu_model_ready': pangu_ready,
        'ecmwf_bufr_available': bufr_available,
        'active_typhoon_count': _scheduler_state.get('active_typhoon_count', 0),
        'active_cache_coverage': active_coverage,
        'on_demand_queue_size': _scheduler_state.get('on_demand_queue_size', 0),
        'last_data_fetch': _scheduler_state.get('last_data_fetch', ''),
        'last_prediction': _scheduler_state.get('last_prediction', ''),
        'last_method': _scheduler_state.get('last_method', ''),
        'last_ecmwf_bufr': _scheduler_state.get('last_ecmwf_bufr', ''),
        'last_training': _scheduler_state.get('last_training', ''),
        'startup_init_done': _scheduler_state.get('startup_init_done', False),
        'recent_errors': _scheduler_state.get('errors', [])[-5:],
        'auto_fetch_interval': f'{FETCH_INTERVAL}秒 (1小时)',
        'auto_prediction_interval': f'{PREDICTION_INTERVAL}秒 (30分钟)',
        'auto_prediction_method': f'{BACKGROUND_METHOD} (后台重型方法)',
        'auto_bufr_interval': '6小时',
        'auto_training_schedule': '每天3:00AM',
        'prediction_hours': PREDICTION_HOURS,
        'cache_stale_threshold': f'{CACHE_STALE_MINUTES}分钟',
        'background_stale_threshold': f'{BACKGROUND_STALE_MINUTES}分钟(重型计算)',
    }


def get_active_cache_coverage():
    """★ 查询所有活跃台风的缓存覆盖情况（前端批量预加载用）"""
    now = datetime.now()
    ym = now.strftime('%Y%m')
    local_file = os.path.join(ISC_DIR, f'{ym}.json')

    active_typhoons = []

    # 搜索当前月 + 前后月份的活跃台风
    for offset in [-2, -1, 0, 1, 2]:
        month_dt = now + timedelta(days=offset * 30)
        month_ym = month_dt.strftime('%Y%m')
        month_file = os.path.join(ISC_DIR, f'{month_ym}.json')
        if not os.path.exists(month_file):
            continue
        try:
            with open(month_file, 'r') as f:
                month_data = json.load(f)
            for t in month_data:
                if t.get('is_current') == 1 and t.get('tfbh') not in [a['tfbh'] for a in active_typhoons]:
                    active_typhoons.append(t)
        except:
            pass

    # 构建缓存覆盖信息
    coverage = []
    for t in active_typhoons:
        tfid = t.get('tfbh', '')
        if not tfid:
            continue
        cached_hours = []
        for h in PREDICTION_HOURS:
            cache_file = os.path.join(PREDICTION_CACHE_DIR, f'{tfid}_{h}h.json')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    computed_at = data.get('computed_at', '')
                    age_min = (datetime.now() - datetime.fromisoformat(computed_at)).total_seconds() / 60
                    cached_hours.append({
                        'hours': h,
                        'fresh': age_min < BACKGROUND_STALE_MINUTES,
                        'age_minutes': round(age_min),
                        'computed_at': computed_at,
                        'methods': list(data.get('predictions', {}).keys()),
                    })
                except:
                    cached_hours.append({'hours': h, 'fresh': False, 'age_minutes': 999})
        coverage.append({
            'tfid': tfid,
            'name_cn': t.get('name_cn', ''),
            'name_en': t.get('name_en', ''),
            'cached_hours': cached_hours,
            'total_cached': len(cached_hours),
            'total_expected': len(PREDICTION_HOURS),
        })

    return coverage


# ============================================================
# CLI入口
# ============================================================

if __name__ == '__main__':
    run_scheduler_loop()
