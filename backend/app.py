"""
台风路径预测系统 - 后端服务
数据源: ISC (istrongcloud.com) + NII (日本信息研究所)
"""

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests as req_lib

# LSTM深度学习预测模块
from lstm_predictor import lstm_predict, is_lstm_ready, LSTMTrainer, WINDOW_SIZE
from pangu_predictor import pangu_predict, is_pangu_ready
from ecmwf_bufr_fetcher import fetch_ecmwf_tracks_for_typhoon, get_ecmwf_active_storms, is_bufr_available

# 海岸线 & 登陆检测
from coastline import detect_landfall_from_segments, detect_landfall, get_coastline_geojson

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)  # 允许跨域访问

# ============================================================
# 数据抓取与解析模块
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ISC_DIR = os.path.join(DATA_DIR, 'isc')
NII_DIR = os.path.join(DATA_DIR, 'nii')

CACHE_DURATION = 3600  # 1小时缓存
_cache = {}

headers = {
    'Connection': 'Keep-Alive',
    'Accept': 'text/html, application/xhtml+xml, */*',
    'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def ensure_dirs():
    for d in [ISC_DIR, NII_DIR]:
        os.makedirs(d, exist_ok=True)


def fetch_isc_typhoon_data(year_month):
    """从 ISC 获取指定年月的台风数据"""
    cache_key = f"isc_{year_month}"
    if cache_key in _cache and time.time() - _cache[cache_key]['ts'] < CACHE_DURATION:
        return _cache[cache_key]['data']

    # 先尝试本地文件
    local_file = os.path.join(ISC_DIR, f'{year_month}.json')
    if os.path.exists(local_file):
        with open(local_file, 'r') as f:
            data = json.load(f)
        _cache[cache_key] = {'data': data, 'ts': time.time()}
        return data

    # 从远程获取
    url = f'https://data.istrongcloud.com/v2/data/complex/{year_month}.json'
    try:
        response = req_lib.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            # 缓存到本地
            with open(local_file, 'w') as f:
                json.dump(data, f)
            _cache[cache_key] = {'data': data, 'ts': time.time()}
            return data
    except Exception as e:
        print(f"ISC fetch error for {year_month}: {e}")
    return []


def fetch_isc_year_list(year):
    """获取指定年份所有台风列表"""
    typhoons = []
    for month in range(1, 13):
        number = f"{year}{str(month).zfill(2)}"
        data = fetch_isc_typhoon_data(number)
        for t in data:
            typhoons.append(t)
    return typhoons


def fetch_isc_current_typhoons():
    """获取当前活跃台风"""
    now = datetime.now()
    year_month = now.strftime('%Y%m')
    data = fetch_isc_typhoon_data(year_month)
    current = [t for t in data if t.get('is_current') == 1]
    # 也检查前一个月
    prev = now - timedelta(days=30)
    prev_ym = prev.strftime('%Y%m')
    prev_data = fetch_isc_typhoon_data(prev_ym)
    for t in prev_data:
        if t.get('is_current') == 1 and t['tfbh'] not in [c['tfbh'] for c in current]:
            current.append(t)
    return current


def fetch_nii_typhoon_geojson(tid):
    """从 NII 获取单个台风的 GeoJSON 数据"""
    cache_key = f"nii_{tid}"
    if cache_key in _cache and time.time() - _cache[cache_key]['ts'] < CACHE_DURATION * 24:
        return _cache[cache_key]['data']

    local_file = os.path.join(NII_DIR, f'{tid}.json')
    if os.path.exists(local_file):
        with open(local_file, 'r') as f:
            data = json.load(f)
        _cache[cache_key] = {'data': data, 'ts': time.time()}
        return data

    url = f'https://agora.ex.nii.ac.jp/digital-typhoon/geojson/wnp/{tid}.en.json'
    try:
        response = req_lib.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            with open(local_file, 'w') as f:
                json.dump(data, f)
            _cache[cache_key] = {'data': data, 'ts': time.time()}
            return data
    except Exception as e:
        print(f"NII fetch error for {tid}: {e}")
    return None


def fetch_nii_typhoon_list():
    """从 NII 获取台风列表"""
    url = 'https://agora.ex.nii.ac.jp/cgi-bin/dt/search_name2.pl?lang=en&basin=wnp&smp=1&sdp=1&emp=12&edp=31'
    try:
        response = req_lib.get(url, timeout=15)
        response.encoding = 'utf8'
        html = response.text
        ids = re.findall(r'<a href="/digital-typhoon/summary/wnp/s/(.*?)">', html, re.S)
        return ids
    except Exception as e:
        print(f"NII list fetch error: {e}")
        return []


def normalize_isc_data(raw_data):
    """标准化 ISC 数据为统一格式"""
    result = []
    for typhoon in raw_data:
        normalized = {
            'id': typhoon.get('tfbh', ''),
            'name_cn': typhoon.get('name', ''),
            'name_en': typhoon.get('ename', ''),
            'is_current': typhoon.get('is_current', 0),
            'begin_time': typhoon.get('begin_time', ''),
            'end_time': typhoon.get('end_time', ''),
            'landfalls': typhoon.get('land', []),
            'source': 'ISC',
            'points': [],
            'forecasts': {}
        }

        for p in typhoon.get('points', []):
            point = {
                'time': p.get('time', ''),
                'lat': p.get('lat', 0),
                'lng': p.get('lng', 0),
                'pressure': p.get('pressure', 0),
                'wind_speed': p.get('speed', 0),  # m/s
                'power': p.get('power', 0),  # 等级
                'category': p.get('strong', ''),
                'move_dir': p.get('move_dir', ''),
                'move_speed': p.get('move_speed', 0),  # km/h
                'radius7': p.get('radius7'),
                'radius10': p.get('radius10'),
                'radius12': p.get('radius12'),
            }
            normalized['points'].append(point)

            # 处理预报数据
            if p.get('forecast'):
                for fc in p['forecast']:
                    agency = fc.get('sets', '')
                    if agency not in normalized['forecasts']:
                        normalized['forecasts'][agency] = {
                            'forecast_time': p.get('time', ''),
                            'points': []
                        }
                    for fp in fc.get('points', []):
                        normalized['forecasts'][agency]['points'].append({
                            'time': fp.get('time', ''),
                            'lat': fp.get('lat', 0),
                            'lng': fp.get('lng', 0),
                            'pressure': fp.get('pressure'),
                            'wind_speed': fp.get('speed', 0),
                            'power': fp.get('power', 0),
                            'category': fp.get('strong', ''),
                        })

        result.append(normalized)
    return result


def normalize_nii_data(geojson_data, tid):
    """标准化 NII GeoJSON 数据为统一格式"""
    if not geojson_data or 'features' not in geojson_data:
        return None

    props = geojson_data.get('properties', {})
    normalized = {
        'id': tid,
        'name_en': props.get('name', ''),
        'name_cn': '',
        'is_current': 0,
        'begin_time': '',
        'end_time': '',
        'landfalls': [],
        'source': 'NII',
        'points': [],
        'forecasts': {}
    }

    for feature in geojson_data['features']:
        coords = feature['geometry']['coordinates']
        fprops = feature['properties']
        point = {
            'time': fprops.get('display_time', ''),
            'lat': coords[1],
            'lng': coords[0],
            'pressure': fprops.get('pressure', 0),
            'wind_speed': round(fprops.get('wind', 0) * 1.852, 1) if fprops.get('wind') else 0,  # kt -> m/s rough
            'power': 0,
            'category': _nii_class_to_category(fprops.get('class', 0)),
            'move_dir': '',
            'move_speed': 0,
            'radius7': None,
            'radius10': None,
            'radius12': None,
        }
        normalized['points'].append(point)

    if normalized['points']:
        normalized['begin_time'] = normalized['points'][0]['time']
        normalized['end_time'] = normalized['points'][-1]['time']

    return normalized


def _nii_class_to_category(cls):
    """NII等级转分类名称"""
    mapping = {
        2: '热带低压(TD)',
        3: '热带风暴(TS)',
        4: '强热带风暴(STS)',
        5: '台风(TY)/超强台风(Super TY)',
    }
    return mapping.get(cls, '')


# ============================================================
# 台风路径预测模块
# ============================================================

class TyphoonPredictor:
    """台风路径预测引擎 - 多方法融合"""

    # 西北太平洋典型引导气流参数
    STEERING_FLOW_PARAMS = {
        'summer': {'beta_drift_n': 1.5, 'beta_drift_w': 0.8},  # 夏季偏向西北
        'autumn': {'beta_drift_n': 1.2, 'beta_drift_w': 1.5},  # 秋季偏西
        'winter': {'beta_drift_n': 0.8, 'beta_drift_w': 2.0},  # 冬季强西移
    }

    @staticmethod
    def predict_path(typhoon_data, hours=72, method='ensemble'):
        """
        预测台风未来路径
        methods: 'trend' (趋势外推), 'physics' (物理模型),
                 'gfs' (GFS数值预报), 'gfs_graphcast' (GFS GraphCast AI),
                 'analog' (历史相似法), 'ensemble' (综合), 'all' (全部方法)
        """
        points = typhoon_data.get('points', [])
        if len(points) < 2:
            return {'error': '数据点不足，无法预测', 'predictions': []}

        predictions = {}
        last_point = points[-1]

        # ★★★ 机构速度校准：从机构预报提取共识移动速度 ★★★
        # 核心思路：AI方法（trend/physics）只能从过去数据推断速度，
        # 但NWP模型可以预测未来加速。当机构预报可用时，
        # 用机构共识速度校准AI预测速度，避免AI方法系统性偏慢。
        agency_speed_calibration = None
        forecasts = typhoon_data.get('forecasts', {})
        if forecasts:
            agency_lat_rates = []
            agency_lng_rates = []
            for agency, fc_data in forecasts.items():
                fcpts = fc_data.get('points', [])
                if len(fcpts) >= 2:
                    try:
                        t0 = datetime.fromisoformat(fcpts[0]['time'].replace('Z', '+00:00'))
                        t_last = datetime.fromisoformat(fcpts[-1]['time'].replace('Z', '+00:00'))
                        dt_total = (t_last - t0).total_seconds() / 3600
                        if dt_total > 12:  # 至少12小时才有意义
                            dlat = fcpts[-1]['lat'] - fcpts[0]['lat']
                            dlng = fcpts[-1]['lng'] - fcpts[0]['lng']
                            agency_lat_rates.append(dlat / dt_total)
                            agency_lng_rates.append(dlng / dt_total)
                    except:
                        pass
            if agency_lat_rates:
                # 机构共识速度（度/小时）
                agency_speed_calibration = {
                    'lat_rate': sum(agency_lat_rates) / len(agency_lat_rates),
                    'lng_rate': sum(agency_lng_rates) / len(agency_lng_rates),
                    'n_agencies': len(agency_lat_rates),
                }

        # 1. 趋势外推（本地计算，秒级）—— 用机构速度校准
        if method in ['trend', 'ensemble', 'all']:
            predictions['trend'] = TyphoonPredictor._trend_extrapolation(
                points, hours, agency_speed_calibration
            )

        # 2. 物理模型（本地计算，秒级）—— 用机构速度校准
        if method in ['physics', 'ensemble', 'all']:
            predictions['physics'] = TyphoonPredictor._physics_model(
                points, hours, agency_speed_calibration
            )

        # 3. NWP涡旋追踪 —— 并发调用多个NWP模型（GFS、ECMWF等）
        #    串行调用每个耗时10-20秒，并发后总耗时等于最慢的单次调用
        nwp_tasks = []
        if method in ['gfs', 'ensemble', 'all']:
            nwp_tasks.append(('gfs', 'gfs', 'GFS'))
        if method in ['ecmwf', 'ensemble', 'all']:
            nwp_tasks.append(('ecmwf', 'ecmwf_ifs04', 'ECMWF IFS'))
        if method in ['aifs', 'all']:
            nwp_tasks.append(('aifs', 'aifs', 'AIFS'))
        if method in ['cma', 'all']:
            nwp_tasks.append(('cma', 'cma_grapes_global', 'CMA GRAPES'))

        if nwp_tasks:
            with ThreadPoolExecutor(max_workers=min(len(nwp_tasks), 4)) as pool:
                future_map = {}
                for key, model, label in nwp_tasks:
                    future_map[pool.submit(
                        TyphoonPredictor._nwp_vortex_track, last_point, hours, model, label
                    )] = key
                for future in as_completed(future_map):
                    key = future_map[future]
                    try:
                        result = future.result()
                        if result:
                            predictions[key] = result
                    except Exception as e:
                        print(f"NWP concurrent error ({key}): {e}")

        # 4. GraphCast AI预报（仅单独调用时使用）
        if method in ['gfs_graphcast', 'all']:
            gc_pred = TyphoonPredictor._gfs_graphcast_prediction(last_point, hours)
            if gc_pred:
                predictions['gfs_graphcast'] = gc_pred

        # 5. 历史相似路径类比法（仅单独调用或all时使用，ensemble跳过）
        if method in ['analog', 'all']:
            analog_pred = TyphoonPredictor._analog_prediction(typhoon_data, hours)
            if analog_pred:
                predictions['analog'] = analog_pred

        # 6. LSTM深度学习预测
        if method in ['lstm', 'ensemble', 'all']:
            lstm_pred = TyphoonPredictor._lstm_prediction(points, hours)
            if lstm_pred:
                predictions['lstm'] = lstm_pred

        # 7. Pangu-Weather盘古大模型（可选）
        if method in ['pangu', 'all']:
            pangu_pred = TyphoonPredictor._pangu_prediction(typhoon_data, hours)
            if pangu_pred:
                predictions['pangu'] = pangu_pred

        # 8. ECMWF BUFR官方台风轨迹预报（最高精度数据源）
        #    直接获取ECMWF官方的台风路径预测，无需自建NWP
        if method in ['ecmwf_bufr', 'ensemble', 'all']:
            bufr_tracks = fetch_ecmwf_tracks_for_typhoon(
                last_point.get('lat', 0), last_point.get('lng', 0),
                hours=hours
            )
            if bufr_tracks:
                # HRES确定性预报(全球最准)
                if 'ecmwf_bufr' in bufr_tracks:
                    predictions['ecmwf_bufr'] = bufr_tracks['ecmwf_bufr']
                # ENS集合均值(不确定性更低)
                if 'ecmwf_bufr_ens' in bufr_tracks:
                    predictions['ecmwf_bufr_ens'] = bufr_tracks['ecmwf_bufr_ens']

        # ★ 关键：在ensemble融合之前，先纳入机构预报
        if typhoon_data.get('forecasts'):
            for agency, fc_data in typhoon_data['forecasts'].items():
                predictions[f'forecast_{agency}'] = fc_data['points']

        # 综合融合 (Kalman滤波加权，含机构预报锚定)
        if method in ['ensemble', 'all'] and len(predictions) >= 2:
            ensemble_pred = TyphoonPredictor._kalman_ensemble(predictions, hours, last_point)
            if ensemble_pred:
                predictions['ensemble'] = ensemble_pred

        # ============================================================
        # 登陆点检测（增强版：线段相交法 + 距离法回退）
        # ============================================================
        landfalls = {}
        for method_name, pred_points in predictions.items():
            if method_name.startswith('forecast_'):
                continue
            # 1. 精确的线段相交法检测
            lf = detect_landfall_from_segments(pred_points, margin_deg=0.4)
            if lf:
                landfalls[method_name] = lf
                continue
            # 2. 距离法回退兜底（增大 margin 检测近岸路径）
            lf = detect_landfall(pred_points, margin_deg=0.5)
            if lf:
                landfalls[method_name] = lf

        return {
            'typhoon_id': typhoon_data.get('id', ''),
            'method': method,
            'base_time': points[-1]['time'] if points else '',
            'base_lat': points[-1]['lat'] if points else 0,
            'base_lng': points[-1]['lng'] if points else 0,
            'predictions': predictions,
            'landfalls': landfalls,
            'available_methods': list(predictions.keys()),
            'confidence': TyphoonPredictor._calculate_confidence(points, hours),
        }

    # ============================================================
    # 方法3: GFS数值预报引导气流法
    # ============================================================
    @staticmethod
    def _gfs_steering_prediction(last_point, hours):
        """通过Open-Meteo API获取GFS模型500/700hPa引导气流，预测台风移动方向"""
        lat = last_point.get('lat', 0)
        lng = last_point.get('lng', 0)

        if lat == 0 and lng == 0:
            return None

        # 台风被500-700hPa的环境气流引导，查询这两个层的风场
        # 在台风中心及周围取多点计算环境引导气流
        offsets = [(0, 0), (2, 2), (2, -2), (-2, 2), (-2, -2)]  # 中心+四角2度偏移
        all_wind_data = []

        for dlat, dlng in offsets:
            query_lat = lat + dlat
            query_lng = lng + dlng

            url = (
                f'https://api.open-meteo.com/v1/gfs?'
                f'latitude={query_lat}&longitude={query_lng}'
                f'&hourly=wind_speed_700hPa,wind_direction_700hPa,'
                f'wind_speed_500hPa,wind_direction_500hPa,'
                f'pressure_msl,temperature_2m'
                f'&forecast_days={min(hours // 24 + 1, 16)}'
                f'&cell_selection=nearest'
                f'&timeformat=iso8601'
            )

            try:
                response = req_lib.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    all_wind_data.append({
                        'offset': (dlat, dlng),
                        'data': data,
                    })
            except Exception as e:
                print(f"GFS API error for ({query_lat}, {query_lng}): {e}")

        if not all_wind_data:
            return None

        # 计算引导气流（700hPa和500hPa加权平均）
        # 台风主要被700hPa（深层对流层中层）气流引导
        predictions = []
        try:
            base_time_str = last_point.get('time', '')
            base_time = datetime.fromisoformat(base_time_str.replace('Z', '+00:00'))
        except:
            base_time = datetime.now()

        hourly_times = all_wind_data[0]['data'].get('hourly', {}).get('time', [])
        if not hourly_times:
            return None

        # 找到基准时间对应的索引
        base_idx = 0
        for i, t in enumerate(hourly_times):
            try:
                ht = datetime.fromisoformat(t)
                if ht >= base_time:
                    base_idx = i
                    break
            except:
                continue

        dt = 6  # 6小时步长
        current_lat = lat
        current_lng = lng

        for step in range(1, hours // dt + 1):
            target_idx = base_idx + step * dt // 1  # GFS是逐小时
            if target_idx >= len(hourly_times):
                break

            # 计算引导气流：周围多点的700hPa和500hPa风场加权
            steering_u = 0  # 东向分量 (m/s)
            steering_v = 0  # 北向分量 (m/s)
            valid_points = 0

            for wd in all_wind_data:
                hourly = wd['data'].get('hourly', {})
                times = hourly.get('time', [])
                if target_idx >= len(times):
                    continue

                # 700hPa引导气流 (权重0.6，更直接影响台风移动)
                ws700 = hourly.get('wind_speed_700hPa', [])
                wd700 = hourly.get('wind_direction_700hPa', [])

                # 500hPa引导气流 (权重0.4，反映深层气流)
                ws500 = hourly.get('wind_speed_500hPa', [])
                wd500 = hourly.get('wind_direction_500hPa', [])

                if target_idx < len(ws700) and ws700[target_idx] is not None:
                    # 风向转u/v分量 (气象风向：从北0°顺时针)
                    dir700 = wd700[target_idx] if target_idx < len(wd700) else 0
                    u700 = ws700[target_idx] * math.sin(math.radians(dir700)) / 3.6  # km/h -> m/s
                    v700 = -ws700[target_idx] * math.cos(math.radians(dir700)) / 3.6
                    steering_u += u700 * 0.6
                    steering_v += v700 * 0.6
                    valid_points += 0.6

                if target_idx < len(ws500) and ws500[target_idx] is not None:
                    dir500 = wd500[target_idx] if target_idx < len(wd500) else 0
                    u500 = ws500[target_idx] * math.sin(math.radians(dir500)) / 3.6
                    v500 = -ws500[target_idx] * math.cos(math.radians(dir500)) / 3.6
                    steering_u += u500 * 0.4
                    steering_v += v500 * 0.4
                    valid_points += 0.4

            if valid_points > 0:
                steering_u /= valid_points
                steering_v /= valid_points

            # 转换风速到经纬度移动速率
            # 1度纬度 ≈ 111km, 1度经度 ≈ 111*cos(lat)km
            lat_km = 111.0
            lng_km = 111.0 * math.cos(math.radians(current_lat))

            # 6小时移动量 (m/s * 6h * 3600s / 1000 = km)
            move_lat_km = steering_v * dt * 3600 / 1000
            move_lng_km = steering_u * dt * 3600 / 1000

            # 加入Beta漂移修正 (约1-2度/天向北)
            beta_n_per_6h = 0.06  # ~1.5度/天 / 4步
            beta_w_per_6h = -0.04

            pred_lat = current_lat + move_lat_km / lat_km + beta_n_per_6h
            pred_lng = current_lng + move_lng_km / lng_km + beta_w_per_6h

            # 转向因子：高纬度时偏向东北
            if pred_lat > 25:
                recurvature = min((pred_lat - 25) / 15, 1.0) * 0.3
                pred_lng += recurvature

            # 估算气压和风速（从GFS海平面气压）
            pred_pressure = last_point.get('pressure', 1000)
            pred_wind = last_point.get('wind_speed', 0)

            current_lat = pred_lat
            current_lng = pred_lng

            pred_time = (base_time + timedelta(hours=step * dt)).isoformat()

            # 置信度：GFS是最权威的NWP模型，基础置信度较高
            h = step * dt
            decay = math.exp(-h / (hours * 2.5))

            predictions.append({
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': TyphoonPredictor._intensity_category(pred_pressure, pred_wind),
                'confidence': round(0.92 * decay, 2),
                'method_desc': '基于GFS 500/700hPa引导气流',
            })

        return predictions if predictions else None

    # ============================================================
    # 方法3b: NWP涡旋追踪法（替代单点引导气流法）
    # ============================================================
    @staticmethod
    def _nwp_vortex_track(last_point, hours, model='gfs', model_label='GFS'):
        """NWP涡旋追踪：查询模型MSLP气压场网格，追踪最低气压中心位置。
        这是业务预报中追踪台风的标准方法——不是查一个点的风来推方向，
        而是直接看模型预报的气压场中低压中心在哪里。

        model: 'gfs', 'ecmwf_ifs04', 'cma_grapes_global', 'aifs'
        """
        lat = last_point.get('lat', 0)
        lng = last_point.get('lng', 0)
        if lat == 0 and lng == 0:
            return None

        # 构建查询网格：以台风为中心，±3度范围，步长3度 = 3x3网格
        grid_size = 3  # ±3度
        grid_step = 3  # 3度间距
        lats_grid = []
        lngs_grid = []
        for dlat in range(-grid_size, grid_size + 1, grid_step):
            for dlng in range(-grid_size, grid_size + 1, grid_step):
                lats_grid.append(round(lat + dlat, 1))
                lngs_grid.append(round(lng + dlng, 1))

        # Open-Meteo支持多位置查询（逗号分隔）
        lat_str = ','.join(str(l) for l in lats_grid)
        lng_str = ','.join(str(l) for l in lngs_grid)
        forecast_days = min(hours // 24 + 1, 16)

        # 构建API URL
        if model == 'gfs':
            base_url = 'https://api.open-meteo.com/v1/gfs'
            model_param = ''
        else:
            base_url = 'https://api.open-meteo.com/v1/forecast'
            model_param = f'&models={model}'

        url = (
            f'{base_url}?latitude={lat_str}&longitude={lng_str}'
            f'&hourly=pressure_msl'
            f'{model_param}'
            f'&forecast_days={forecast_days}'
            f'&cell_selection=nearest'
            f'&timeformat=iso8601'
        )

        try:
            response = req_lib.get(url, timeout=10)
            if response.status_code != 200:
                print(f"NWP vortex track API error ({model_label}): {response.status_code}")
                return None
            data = response.json()
        except Exception as e:
            print(f"NWP vortex track error ({model_label}): {e}")
            return None

        # Open-Meteo返回多位置数据为list
        if not isinstance(data, list):
            data = [data]

        if len(data) != len(lats_grid):
            print(f"NWP vortex track: grid mismatch {len(data)} vs {len(lats_grid)}")
            return None

        # 提取每个网格点的时间序列
        grid_data = []
        for i, loc_data in enumerate(data):
            hourly = loc_data.get('hourly', {})
            times = hourly.get('time', [])
            pressures = hourly.get('pressure_msl', [])
            grid_data.append({
                'lat': lats_grid[i],
                'lng': lngs_grid[i],
                'times': times,
                'pressures': pressures,
            })

        if not grid_data or not grid_data[0]['times']:
            return None

        try:
            base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
        except:
            base_time = datetime.now()

        # 找到基准时间索引
        times = grid_data[0]['times']
        base_idx = 0
        for i, t in enumerate(times):
            try:
                ht = datetime.fromisoformat(t)
                if ht >= base_time:
                    base_idx = i
                    break
            except:
                continue

        # 在每个预报时刻用气压加权质心法追踪台风中心
        # 不只取最低气压点（粗网格下会卡住），而是用所有低气压点加权平均
        dt = 6
        predictions = []
        prev_lat = lat
        prev_lng = lng
        last_pressure = last_point.get('pressure', 1000)

        for step in range(1, hours // dt + 1):
            target_idx = base_idx + step * dt
            if target_idx >= len(times):
                break

            # 收集所有网格点的气压值
            grid_pressures = []
            for gd in grid_data:
                if target_idx < len(gd['pressures']) and gd['pressures'][target_idx] is not None:
                    p = gd['pressures'][target_idx]
                    dist = math.sqrt((gd['lat'] - prev_lat)**2 + (gd['lng'] - prev_lng)**2)
                    if dist <= 8:  # 限制搜索范围
                        grid_pressures.append({
                            'lat': gd['lat'],
                            'lng': gd['lng'],
                            'pressure': p,
                            'dist': dist,
                        })

            if not grid_pressures:
                break

            # 找最低气压
            min_p = min(g['pressure'] for g in grid_pressures)

            # 气压加权质心：气压越低权重越大
            # w = (1015 - p)^2 / (dist + 0.5)
            # 这样低气压点权重高，且离上一个位置近的点权重高
            weighted_lat = 0
            weighted_lng = 0
            weight_sum = 0

            for g in grid_pressures:
                dp = 1015 - g['pressure']
                if dp > 0:
                    w = dp * dp / (g['dist'] + 0.5)
                    weighted_lat += g['lat'] * w
                    weighted_lng += g['lng'] * w
                    weight_sum += w

            if weight_sum > 0:
                pred_lat = weighted_lat / weight_sum
                pred_lng = weighted_lng / weight_sum
            else:
                # 所有气压都>=1015，用最低气压点
                min_g = min(grid_pressures, key=lambda x: x['pressure'])
                pred_lat = min_g['lat']
                pred_lng = min_g['lng']

            pred_pressure = min_p
            pred_wind = TyphoonPredictor._pressure_to_wind(pred_pressure)

            h = step * dt
            decay = math.exp(-h / (hours * 2.5))
            pred_time = (base_time + timedelta(hours=h)).isoformat()

            predictions.append({
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': TyphoonPredictor._intensity_category(pred_pressure, pred_wind),
                'confidence': round(0.90 * decay, 2),
                'method_desc': f'{model_label}涡旋追踪(气压加权质心)',
            })

            prev_lat = pred_lat
            prev_lng = pred_lng
            last_pressure = pred_pressure

        return predictions if predictions else None

    # ============================================================
    # 方法4: GFS GraphCast AI预报
    # ============================================================
    @staticmethod
    def _gfs_graphcast_prediction(last_point, hours):
        """通过Open-Meteo API获取GraphCast(DeepMind AI)模型预报，
        GraphCast是目前最先进的AI天气预测模型，在多项指标上超越传统NWP"""
        lat = last_point.get('lat', 0)
        lng = last_point.get('lng', 0)

        if lat == 0 and lng == 0:
            return None

        # GraphCast提供6小时间隔的全球预报
        url = (
            f'https://api.open-meteo.com/v1/gfs?'
            f'latitude={lat}&longitude={lng}'
            f'&hourly=pressure_msl,wind_speed_10m,wind_direction_10m,'
            f'wind_speed_80m,wind_direction_80m,temperature_2m,'
            f'relative_humidity_2m,cape'
            f'&models=gfs_graphcast'
            f'&forecast_days={min(hours // 24 + 1, 16)}'
            f'&cell_selection=nearest'
            f'&timeformat=iso8601'
        )

        try:
            response = req_lib.get(url, timeout=10)
            if response.status_code != 200:
                return None
            data = response.json()
        except Exception as e:
            print(f"GraphCast API error: {e}")
            return None

        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        if not times:
            return None

        try:
            base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
        except:
            base_time = datetime.now()

        # 找基准索引
        base_idx = 0
        for i, t in enumerate(times):
            try:
                ht = datetime.fromisoformat(t)
                if ht >= base_time:
                    base_idx = i
                    break
            except:
                continue

        predictions = []
        current_lat = lat
        current_lng = lng

        pressures = hourly.get('pressure_msl', [])
        wind_speeds = hourly.get('wind_speed_10m', [])
        wind_dirs = hourly.get('wind_direction_10m', [])

        dt = 6
        for step in range(1, hours // dt + 1):
            target_idx = base_idx + step * dt
            if target_idx >= len(times):
                break

            # 使用GFS GraphCast的风场来推断台风移动方向
            # 台风近似跟随10m层环境风场移动（简化假设）
            if target_idx < len(wind_speeds) and wind_speeds[target_idx] is not None:
                ws = wind_speeds[target_idx]
                wd_val = wind_dirs[target_idx] if target_idx < len(wind_dirs) and wind_dirs[target_idx] else 0

                # 环境风分量（但这只是局地风，需要更大尺度引导）
                # 简化：使用80m风作为引导气流近似
                # 但GraphCast不提供气压层变量，所以用10m+80m合成
                u_env = ws * math.sin(math.radians(wd_val)) / 3.6  # km/h -> m/s
                v_env = -ws * math.cos(math.radians(wd_val)) / 3.6

                # 台风移速通常为环境风的50-70%（Franklin原则）
                typhoon_move_u = u_env * 0.6
                typhoon_move_v = v_env * 0.6

                move_lat_km = typhoon_move_v * dt * 3600 / 1000
                move_lng_km = typhoon_move_u * dt * 3600 / 1000

                pred_lat = current_lat + move_lat_km / 111.0
                pred_lng = current_lng + move_lng_km / (111.0 * math.cos(math.radians(current_lat)))

                # Beta漂移修正
                pred_lat += 0.06
                pred_lng -= 0.04
            else:
                # 无法获取风场数据时跳过
                break

            pred_pressure = pressures[target_idx] if target_idx < len(pressures) and pressures[target_idx] else last_point.get('pressure', 1000)
            # GraphCast的10m风速是环境风速，不是台风中心风速
            # 台风中心风速需要从气压估算
            pred_wind = TyphoonPredictor._pressure_to_wind(pred_pressure)

            current_lat = pred_lat
            current_lng = pred_lng

            h = step * dt
            decay = math.exp(-h / (hours * 3))
            pred_time = (base_time + timedelta(hours=step * dt)).isoformat()

            predictions.append({
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': TyphoonPredictor._intensity_category(pred_pressure, pred_wind),
                'confidence': round(0.88 * decay, 2),
                'method_desc': 'DeepMind GraphCast AI天气模型',
            })

        return predictions if predictions else None

    # ============================================================
    # 方法4b: ECMWF AIFS AI预报系统
    # ============================================================
    @staticmethod
    def _aifs_prediction(last_point, hours):
        """通过Open-Meteo API获取ECMWF AIFS预报。
        AIFS是ECMWF的AI预报系统(2025年2月正式运行)，ECMWF声称其
        性能超越GraphCast和其他AI天气模型。使用IFS/AIFS的500/700/850hPa
        多层次引导气流和海平面气压场来追踪台风移动。"""
        lat = last_point.get('lat', 0)
        lng = last_point.get('lng', 0)

        if lat == 0 and lng == 0:
            return None

        # AIFS通过ECMWF API获取，使用多层次数据计算引导气流
        # 同时获取500/700/850hPa风场以获得更精确的深层平均引导气流
        url = (
            f'https://api.open-meteo.com/v1/forecast?'
            f'latitude={lat}&longitude={lng}'
            f'&hourly=pressure_msl,wind_speed_10m,wind_direction_10m,'
            f'wind_speed_850hPa,wind_direction_850hPa,'
            f'wind_speed_700hPa,wind_direction_700hPa,'
            f'wind_speed_500hPa,wind_direction_500hPa,'
            f'geopotential_height_500hPa,temperature_2m'
            f'&models=aifs'
            f'&forecast_days={min(hours // 24 + 1, 15)}'
            f'&cell_selection=nearest'
            f'&timeformat=iso8601'
        )

        try:
            response = req_lib.get(url, timeout=15)
            if response.status_code != 200:
                return None
            data = response.json()
        except Exception as e:
            print(f"AIFS API error: {e}")
            return None

        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        if not times:
            return None

        try:
            base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
        except:
            base_time = datetime.now()

        # 找基准索引
        base_idx = 0
        for i, t in enumerate(times):
            try:
                ht = datetime.fromisoformat(t)
                if ht >= base_time:
                    base_idx = i
                    break
            except:
                continue

        predictions = []
        current_lat = lat
        current_lng = lng
        last_pressure = last_point.get('pressure', 1000)

        # 提取各层次风场数据
        wind_500 = hourly.get('wind_speed_500hPa', [])
        dir_500 = hourly.get('wind_direction_500hPa', [])
        wind_700 = hourly.get('wind_speed_700hPa', [])
        dir_700 = hourly.get('wind_direction_700hPa', [])
        wind_850 = hourly.get('wind_speed_850hPa', [])
        dir_850 = hourly.get('wind_direction_850hPa', [])
        pressures = hourly.get('pressure_msl', [])

        dt = 6
        for step in range(1, hours // dt + 1):
            target_idx = base_idx + step * dt
            if target_idx >= len(times):
                break

            # 多层次加权引导气流（业务预报标准方法）
            # 500hPa权重30% + 700hPa权重40% + 850hPa权重30%
            steering_u = 0
            steering_v = 0
            valid_layers = 0

            # 500hPa引导
            if target_idx < len(wind_500) and wind_500[target_idx] is not None:
                ws = wind_500[target_idx]
                wd = dir_500[target_idx] if target_idx < len(dir_500) and dir_500[target_idx] else 0
                u = ws * math.sin(math.radians(wd)) / 3.6
                v = -ws * math.cos(math.radians(wd)) / 3.6
                steering_u += u * 0.30
                steering_v += v * 0.30
                valid_layers += 1

            # 700hPa引导
            if target_idx < len(wind_700) and wind_700[target_idx] is not None:
                ws = wind_700[target_idx]
                wd = dir_700[target_idx] if target_idx < len(dir_700) and dir_700[target_idx] else 0
                u = ws * math.sin(math.radians(wd)) / 3.6
                v = -ws * math.cos(math.radians(wd)) / 3.6
                steering_u += u * 0.40
                steering_v += v * 0.40
                valid_layers += 1

            # 850hPa引导
            if target_idx < len(wind_850) and wind_850[target_idx] is not None:
                ws = wind_850[target_idx]
                wd = dir_850[target_idx] if target_idx < len(dir_850) and dir_850[target_idx] else 0
                u = ws * math.sin(math.radians(wd)) / 3.6
                v = -ws * math.cos(math.radians(wd)) / 3.6
                steering_u += u * 0.30
                steering_v += v * 0.30
                valid_layers += 1

            if valid_layers < 2:
                break

            # 台风移速约为引导气流的70-85%
            dp = max(1010 - last_pressure, 0)
            steering_ratio = min(0.85, 0.55 + dp / 400)

            move_lat_km = steering_v * steering_ratio * dt * 3600 / 1000
            move_lng_km = steering_u * steering_ratio * dt * 3600 / 1000

            pred_lat = current_lat + move_lat_km / 111.0
            pred_lng = current_lng + move_lng_km / (111.0 * math.cos(math.radians(current_lat)))

            pred_lat += 0.06
            pred_lng -= 0.04

            if pred_lat > 25:
                recurvature = min((pred_lat - 25) / 15, 1.0) * 0.25
                pred_lng += recurvature

            pred_pressure = pressures[target_idx] if target_idx < len(pressures) and pressures[target_idx] else last_pressure
            pred_wind = TyphoonPredictor._pressure_to_wind(pred_pressure)

            current_lat = pred_lat
            current_lng = pred_lng

            h = step * dt
            decay = math.exp(-h / (hours * 2.5))
            pred_time = (base_time + timedelta(hours=step * dt)).isoformat()

            predictions.append({
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': TyphoonPredictor._intensity_category(pred_pressure, pred_wind),
                'confidence': round(0.94 * decay, 2),
                'method_desc': 'ECMWF AIFS AI预报(500/700/850hPa多层次引导)',
            })

        return predictions if predictions else None

    # ============================================================
    # 方法4c: CMA GRAPES 中国气象局预报
    # ============================================================
    @staticmethod
    def _cma_prediction(last_point, hours):
        """通过Open-Meteo API获取CMA GRAPES预报。
        CMA GRAPES是中国气象局自主研发的全球数值预报模式(15km分辨率)，
        对西北太平洋台风有最直接的预报经验优势。"""
        lat = last_point.get('lat', 0)
        lng = last_point.get('lng', 0)

        if lat == 0 and lng == 0:
            return None

        url = (
            f'https://api.open-meteo.com/v1/forecast?'
            f'latitude={lat}&longitude={lng}'
            f'&hourly=pressure_msl,wind_speed_10m,wind_direction_10m,'
            f'wind_speed_700hPa,wind_direction_700hPa,'
            f'wind_speed_500hPa,wind_direction_500hPa,'
            f'geopotential_height_500hPa'
            f'&models=cma_grapes_global'
            f'&forecast_days={min(hours // 24 + 1, 10)}'
            f'&cell_selection=nearest'
            f'&timeformat=iso8601'
        )

        try:
            response = req_lib.get(url, timeout=15)
            if response.status_code != 200:
                return None
            data = response.json()
        except Exception as e:
            print(f"CMA GRAPES API error: {e}")
            return None

        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        if not times:
            return None

        try:
            base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
        except:
            base_time = datetime.now()

        base_idx = 0
        for i, t in enumerate(times):
            try:
                ht = datetime.fromisoformat(t)
                if ht >= base_time:
                    base_idx = i
                    break
            except:
                continue

        predictions = []
        current_lat = lat
        current_lng = lng
        last_pressure = last_point.get('pressure', 1000)

        wind_500 = hourly.get('wind_speed_500hPa', [])
        dir_500 = hourly.get('wind_direction_500hPa', [])
        wind_700 = hourly.get('wind_speed_700hPa', [])
        dir_700 = hourly.get('wind_direction_700hPa', [])
        pressures = hourly.get('pressure_msl', [])

        dt = 3  # CMA GRAPES提供3小时间隔数据
        for step in range(1, hours // dt + 1):
            target_idx = base_idx + step * dt
            if target_idx >= len(times):
                break

            steering_u = 0
            steering_v = 0
            valid = 0

            if target_idx < len(wind_500) and wind_500[target_idx] is not None:
                ws = wind_500[target_idx]
                wd = dir_500[target_idx] if target_idx < len(dir_500) and dir_500[target_idx] else 0
                u = ws * math.sin(math.radians(wd)) / 3.6
                v = -ws * math.cos(math.radians(wd)) / 3.6
                steering_u += u * 0.5
                steering_v += v * 0.5
                valid += 1

            if target_idx < len(wind_700) and wind_700[target_idx] is not None:
                ws = wind_700[target_idx]
                wd = dir_700[target_idx] if target_idx < len(dir_700) and dir_700[target_idx] else 0
                u = ws * math.sin(math.radians(wd)) / 3.6
                v = -ws * math.cos(math.radians(wd)) / 3.6
                steering_u += u * 0.5
                steering_v += v * 0.5
                valid += 1

            if valid < 1:
                break

            dp = max(1010 - last_pressure, 0)
            steering_ratio = min(0.85, 0.55 + dp / 400)

            move_lat_km = steering_v * steering_ratio * dt * 3600 / 1000
            move_lng_km = steering_u * steering_ratio * dt * 3600 / 1000

            pred_lat = current_lat + move_lat_km / 111.0
            pred_lng = current_lng + move_lng_km / (111.0 * math.cos(math.radians(current_lat)))

            pred_lat += 0.04 * (dt / 6)
            pred_lng -= 0.03 * (dt / 6)

            if pred_lat > 25:
                recurvature = min((pred_lat - 25) / 15, 1.0) * 0.20 * (dt / 6)
                pred_lng += recurvature

            pred_pressure = pressures[target_idx] if target_idx < len(pressures) and pressures[target_idx] else last_pressure
            pred_wind = TyphoonPredictor._pressure_to_wind(pred_pressure)

            current_lat = pred_lat
            current_lng = pred_lng

            h = step * dt
            decay = math.exp(-h / (hours * 2.5))
            pred_time = (base_time + timedelta(hours=step * dt)).isoformat()

            predictions.append({
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': TyphoonPredictor._intensity_category(pred_pressure, pred_wind),
                'confidence': round(0.90 * decay, 2),
                'method_desc': 'CMA GRAPES中国气象局预报(500+700hPa)',
            })

        return predictions if predictions else None

    @staticmethod
    def _pressure_to_wind(pressure):
        """从中心气压估算最大风速（Atkinson-Holliday经验公式）"""
        if pressure <= 0:
            return 0
        # Vmax = 3.4 * (1010 - Pc)^0.5 (m/s)  西北太平洋经验公式
        dp = max(1010 - pressure, 0)
        return round(3.4 * math.sqrt(dp), 1)

    # ============================================================
    # 方法5: 历史相似路径类比法
    # ============================================================
    @staticmethod
    def _analog_prediction(typhoon_data, hours):
        """历史相似法：从1945年以来的数据中找到与当前台风特征相似的台风，
        用其后续路径作为类比预测。这是气象学中成熟的Analog Method。"""
        points = typhoon_data.get('points', [])
        if len(points) < 3:
            return None

        last = points[-1]
        current_lat = last['lat']
        current_lng = last['lng']
        current_pressure = last.get('pressure', 0)

        # 确定当前台风特征
        try:
            base_time = datetime.fromisoformat(last['time'].replace('Z', '+00:00'))
            month = base_time.month
        except:
            month = 8

        # 计算最近的移动方向和速度
        n = min(4, len(points))
        recent = points[-n:]
        move_dlat = move_dlng = 0
        for i in range(1, len(recent)):
            move_dlat += recent[i]['lat'] - recent[i-1]['lat']
            move_dlng += recent[i]['lng'] - recent[i-1]['lng']
        move_dlat /= max(n-1, 1)
        move_dlng /= max(n-1, 1)
        move_dir_deg = math.degrees(math.atan2(move_dlng, move_dlat)) if (move_dlat != 0 or move_dlng != 0) else 0

        # 搜索历史相似台风
        analogs = TyphoonPredictor._find_analog_typhoons(
            current_lat, current_lng, current_pressure, month, move_dir_deg
        )

        if not analogs:
            return None

        # 取Top-K相似台风的后续路径
        top_k = min(5, len(analogs))
        analog_tracks = []

        for analog in analogs[:top_k]:
            # 找到相似台风中与当前位置最接近的点
            a_points = analog.get('points', [])
            best_idx = 0
            best_dist = float('inf')
            for i, p in enumerate(a_points):
                dist = math.sqrt((p['lat'] - current_lat)**2 + (p['lng'] - current_lng)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # 从该点开始取后续路径
            future_points = a_points[best_idx+1:]
            similarity = analog.get('similarity', 0.5)

            # 截取预测时长对应的数据点
            dt_hours = 3  # ISC数据约3小时间隔
            needed_points = hours // dt_hours
            future_points = future_points[:needed_points]

            if future_points:
                analog_tracks.append({
                    'id': analog.get('id', ''),
                    'name': analog.get('name_cn', '') or analog.get('name_en', ''),
                    'similarity': similarity,
                    'points': future_points,
                })

        if not analog_tracks:
            return None

        # 加权平均所有相似台风的后续路径
        total_weight = sum(t['similarity'] for t in analog_tracks)
        if total_weight == 0:
            total_weight = 1

        # 对齐时间步
        max_len = max(len(t['points']) for t in analog_tracks)
        predictions = []

        for i in range(max_len):
            weighted_lat = 0
            weighted_lng = 0
            weighted_pressure = 0
            weighted_wind = 0
            weight_sum = 0

            for track in analog_tracks:
                if i < len(track['points']):
                    p = track['points'][i]
                    w = track['similarity']
                    weighted_lat += p['lat'] * w
                    weighted_lng += p['lng'] * w
                    weighted_pressure += p.get('pressure', 1000) * w
                    weighted_wind += p.get('wind_speed', 0) * w
                    weight_sum += w

            if weight_sum == 0:
                continue

            pred_lat = weighted_lat / weight_sum
            pred_lng = weighted_lng / weight_sum
            pred_pressure = weighted_pressure / weight_sum
            pred_wind = weighted_wind / weight_sum

            h = (i + 1) * dt_hours
            decay = math.exp(-h / (hours * 2))

            try:
                base_time = datetime.fromisoformat(last['time'].replace('Z', '+00:00'))
                pred_time = (base_time + timedelta(hours=h)).isoformat()
            except:
                pred_time = f"+{h}h"

            predictions.append({
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': TyphoonPredictor._intensity_category(pred_pressure, pred_wind),
                'confidence': round(0.78 * decay, 2),
                'method_desc': f'历史类比({top_k}个相似台风加权)',
                'analog_ids': [t['id'] for t in analog_tracks],
                'analog_names': [t['name'] or t['id'] for t in analog_tracks],
            })

        return predictions if predictions else None

    @staticmethod
    def _find_analog_typhoons(target_lat, target_lng, target_pressure, target_month, target_move_dir):
        """从历史数据中搜索相似台风"""
        analogs = []

        # 搜索近几年的数据（最近3年优先，再搜索更早的）
        current_year = datetime.now().year
        search_years = list(range(current_year - 3, current_year + 1)) + list(range(max(1945, current_year - 20), current_year - 3))

        for year in search_years:
            for month in range(1, 13):
                ym = f"{year}{str(month).zfill(2)}"
                raw_data = fetch_isc_typhoon_data(ym)

                for t in raw_data:
                    t_points = t.get('points', [])
                    if len(t_points) < 6:
                        continue

                    for i, p in enumerate(t_points):
                        # 计算相似度
                        lat_diff = abs(p['lat'] - target_lat)
                        lng_diff = abs(p['lng'] - target_lng)
                        pressure_diff = abs(p.get('pressure', 0) - target_pressure) if target_pressure and p.get('pressure') else 200

                        # 季节相似度（同月±1）
                        try:
                            p_time = datetime.fromisoformat(p['time'].replace('Z', '+00:00'))
                            p_month = p_time.month
                        except:
                            p_month = month
                        month_diff = abs(p_month - target_month)

                        # 移动方向相似度
                        if i > 0:
                            p_move_dir = math.degrees(math.atan2(
                                p['lng'] - t_points[i-1]['lng'],
                                p['lat'] - t_points[i-1]['lat']
                            ))
                            dir_diff = abs(p_move_dir - target_move_dir)
                        else:
                            dir_diff = 180

                        # 综合相似度评分（越小越相似）
                        # 空间距离权重最高
                        spatial_dist = math.sqrt(lat_diff**2 + lng_diff**2)
                        if spatial_dist > 10:  # 超过10度就跳过
                            continue

                        score = (
                            spatial_dist * 3.0 +  # 位置距离（最重要）
                            pressure_diff * 0.01 +  # 强度相似
                            month_diff * 2.0 +  # 季节相似
                            min(dir_diff, 180 - dir_diff) * 0.05  # 移动方向相似
                        )

                        # 转为相似度（0-1）
                        similarity = math.exp(-score / 5)

                        if similarity > 0.3:  # 只保留足够相似的
                            normalized = normalize_isc_data([t])[0]
                            # 标记从哪个点开始
                            normalized['similarity'] = similarity
                            normalized['analog_start_idx'] = i
                            analogs.append(normalized)
                            break  # 每个台风只取一个最佳匹配点

        # 按相似度排序
        analogs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return analogs[:20]  # 返回最相似的20个

    # ============================================================
    # 方法6: LSTM深度学习预测
    # ============================================================
    @staticmethod
    def _lstm_prediction(points, hours):
        """使用训练好的LSTM模型进行递推预测
        - 输入: 8维特征 (lat,lng,pressure,wind,move_speed,move_dir_sin/cos,month)
        - 滑动窗口: 8步 ≈ 24小时
        - 递推: 每步预测1步(6h)，逐步递推"""
        if not is_lstm_ready():
            return None
        if len(points) < WINDOW_SIZE:
            return None

        result = lstm_predict(points, hours)
        return result

    # ============================================================
    # 方法7: Pangu-Weather盘古大模型（可选高级方法）
    # ============================================================
    @staticmethod
    def _pangu_prediction(typhoon_data, hours):
        """使用Pangu-Weather ONNX模型进行本地推理预测。
        需要先下载模型权重到 models/pangu/ 目录。
        这是华为盘古天气大模型(Nature 2022论文)，在台风追踪上表现优于ECMWF HRES。"""
        if not is_pangu_ready():
            return None
        result = pangu_predict(typhoon_data, hours)
        return result

    # ============================================================
    # 方法8: Kalman滤波多方法融合（含机构预报锚定）
    # ============================================================
    @staticmethod
    def _kalman_ensemble(predictions, hours, last_point):
        """Kalman滤波器融合所有预测方法的结果
        核心策略：
        1. 机构预报共识作为主线锚点（权重自适应）
        2. AI/NWP方法仅在共识方向上做有限修正
        3. 异常值剔除：偏离共识过远的AI方法权重骤降
        4. 无机构预报时退化为AI方法加权平均"""

        # 分离机构预报和AI方法
        all_methods = list(predictions.keys())
        agency_methods = [k for k in all_methods if k.startswith('forecast_')]
        ai_methods = [k for k in all_methods if not k.startswith('forecast_') and k != 'ensemble']

        if not ai_methods and not agency_methods:
            return None

        # 自适应机构权重：机构越多，每家权重适当降低但总权重提高
        n_agencies = len(agency_methods)
        if n_agencies >= 5:
            agency_weight_each = 0.14   # 5-6家: 总0.70~0.84
        elif n_agencies >= 3:
            agency_weight_each = 0.18   # 3-4家: 总0.54~0.72
        elif n_agencies >= 1:
            agency_weight_each = 0.25   # 1-2家: 总0.25~0.50
        else:
            agency_weight_each = 0.0    # 无机构预报

        # AI方法权重（仅在机构预报总权重之外分配）
        ai_weights = {
            'ecmwf_bufr': 0.12,   # ECMWF BUFR官方轨迹(全球最准NWP直接输出)
            'ecmwf_bufr_ens': 0.08, # ECMWF BUFR ENS集合均值(不确定性更低)
            'pangu': 0.08,
            'ecmwf': 0.06,      # ECMWF IFS涡旋追踪(Open-Meteo MSLP网格法)
            'aifs': 0.04,        # ECMWF AIFS AI
            'lstm': 0.04,        # LSTM深度学习
            'gfs': 0.02,         # GFS涡旋追踪
            'cma': 0.015,        # CMA GRAPES涡旋追踪
            'gfs_graphcast': 0.01,
            'analog': 0.01,
            'physics': 0.005,
            'trend': 0.005,
        }

        # 异常值剔除阈值：偏离机构共识超过此距离(度)的AI方法权重骤降
        # 3°≈330km，如果AI方法比机构共识慢330km以上就惩罚
        OUTLIER_THRESHOLD_DEG = 3.0   # ~330km（降低阈值，更早惩罚慢速AI方法）
        OUTLIER_PENALTY = 0.05         # 异常方法权重保留比例（从0.1降到0.05，惩罚更狠）

        dt = 6
        results = []

        try:
            base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
        except:
            base_time = datetime.now()

        # 预处理：将机构预报插值到6h步长
        agency_interpolated = {}
        for am in agency_methods:
            raw_pts = predictions.get(am, [])
            if not raw_pts:
                continue
            interpolated = TyphoonPredictor._interpolate_forecast(raw_pts, base_time, hours, dt, last_point)
            if interpolated:
                agency_interpolated[am] = interpolated

        for step in range(1, hours // dt + 1):
            h = step * dt
            time_decay = math.exp(-h / (hours * 2.5))

            # 收集当前时刻各方法的预测
            method_preds = {}

            # AI方法（6h步长，直接取idx）
            for method in ai_methods:
                preds = predictions.get(method, [])
                idx = step - 1
                if idx < len(preds):
                    method_preds[method] = preds[idx]

            # 机构预报（已插值到6h步长）
            for am, interp_pts in agency_interpolated.items():
                idx = step - 1
                if idx < len(interp_pts):
                    method_preds[am] = interp_pts[idx]

            if not method_preds:
                break

            # ★ 计算机构共识（所有可用机构预报的平均位置）
            agency_pts = [method_preds[m] for m in method_preds if m.startswith('forecast_')]
            if agency_pts:
                consensus_lat = sum(p['lat'] for p in agency_pts) / len(agency_pts)
                consensus_lng = sum(p['lng'] for p in agency_pts) / len(agency_pts)
            else:
                consensus_lat = consensus_lng = None

            # 加权平均（含异常值剔除）
            total_weight = 0
            weighted_lat = 0
            weighted_lng = 0
            weighted_pressure = 0
            weighted_wind = 0
            outlier_methods = []

            for method, pred in method_preds.items():
                if method.startswith('forecast_'):
                    # 机构预报：高权重
                    w = agency_weight_each
                else:
                    base_w = ai_weights.get(method, 0.005)
                    # 趋势在短期更准
                    if method == 'trend':
                        w = base_w * (1 + 2.0 * time_decay)
                    elif method in ['lstm', 'analog']:
                        w = base_w * (1 + 0.5 * time_decay)
                    else:
                        w = base_w

                    # ★ 异常值检测：偏离机构共识过远的AI方法权重骤降
                    if consensus_lat is not None:
                        dist_to_consensus = math.sqrt(
                            (pred['lat'] - consensus_lat) ** 2 +
                            (pred['lng'] - consensus_lng) ** 2
                        )
                        if dist_to_consensus > OUTLIER_THRESHOLD_DEG:
                            w *= OUTLIER_PENALTY
                            outlier_methods.append(method)

                weighted_lat += pred['lat'] * w
                weighted_lng += pred['lng'] * w
                weighted_pressure += pred.get('pressure', 1000) * w
                weighted_wind += pred.get('wind_speed', 0) * w
                total_weight += w

            if total_weight == 0:
                break

            pred_lat = weighted_lat / total_weight
            pred_lng = weighted_lng / total_weight
            pred_pressure = weighted_pressure / total_weight
            pred_wind = weighted_wind / total_weight

            # 计算散度（仅基于机构预报，更稳定）
            if len(agency_pts) >= 2:
                lats = [p['lat'] for p in agency_pts]
                lngs = [p['lng'] for p in agency_pts]
                lat_std = math.sqrt(sum((l - consensus_lat) ** 2 for l in lats) / len(lats))
                lng_std = math.sqrt(sum((l - consensus_lng) ** 2 for l in lngs) / len(lngs))
                spread = math.sqrt(lat_std ** 2 + lng_std ** 2)
            elif len(method_preds) >= 2:
                lats = [p['lat'] for p in method_preds.values()]
                lngs = [p['lng'] for p in method_preds.values()]
                lat_std = math.sqrt(sum((l - pred_lat) ** 2 for l in lats) / len(lats))
                lng_std = math.sqrt(sum((l - pred_lng) ** 2 for l in lngs) / len(lngs))
                spread = math.sqrt(lat_std ** 2 + lng_std ** 2)
            else:
                spread = 0
                lat_std = lng_std = 0

            # 置信度：机构一致性高→高置信度
            agency_count = len(agency_pts)
            agency_bonus = min(agency_count * 0.05, 0.15)
            confidence = round(max(0.1, (0.80 + agency_bonus) * time_decay * math.exp(-spread / 5)), 2)

            pred_time = (base_time + timedelta(hours=h)).isoformat()

            desc_parts = [f'Kalman融合({len(method_preds)}种方法,含{agency_count}家机构)']
            if outlier_methods:
                desc_parts.append(f'剔除异常: {",".join(outlier_methods)}')

            result = {
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': TyphoonPredictor._intensity_category(pred_pressure, pred_wind),
                'confidence': confidence,
                'method_desc': '; '.join(desc_parts),
                'spread_lat': round(lat_std, 2),
                'spread_lng': round(lng_std, 2),
                'methods_used': list(method_preds.keys()),
            }

            results.append(result)

        return results if results else None

    @staticmethod
    def _interpolate_forecast(forecast_points, base_time, total_hours, dt, base_point=None):
        """将机构预报点（通常12h/24h间隔）线性插值到dt步长
        base_point: 最后观测点，用于在第一个预报点之前做插值"""
        if not forecast_points:
            return []

        # 解析机构预报的时间和位置
        parsed = []
        for p in forecast_points:
            try:
                t = datetime.fromisoformat(p['time'].replace('Z', '+00:00'))
                parsed.append({
                    'time': t,
                    'lat': float(p['lat'] or 0),
                    'lng': float(p['lng'] or 0),
                    'pressure': float(p.get('pressure') or 1000),
                    'wind_speed': float(p.get('wind_speed') or 0),
                })
            except:
                continue

        if not parsed:
            return []

        parsed.sort(key=lambda x: x['time'])

        # ★ 在预报点序列前面插入当前观测点，使插值从0h开始正确过渡
        if base_point is not None:
            base_pt = {
                'time': base_time,
                'lat': float(base_point.get('lat', 0) or 0),
                'lng': float(base_point.get('lng', 0) or 0),
                'pressure': float(base_point.get('pressure', 1000) or 1000),
                'wind_speed': float(base_point.get('wind_speed', 0) or 0),
            }
            # 如果第一个预报点晚于base_time，在前面插入观测点
            if parsed[0]['time'] > base_time:
                parsed.insert(0, base_pt)

        # 生成dt步长的插值点
        result = []
        for h in range(dt, total_hours + 1, dt):
            target_time = base_time + timedelta(hours=h)

            # 如果在机构预报范围内，线性插值
            if target_time <= parsed[-1]['time']:
                # 找到包围target_time的两个点
                for i in range(len(parsed) - 1):
                    if parsed[i]['time'] <= target_time <= parsed[i+1]['time']:
                        t0 = parsed[i]['time']
                        t1 = parsed[i+1]['time']
                        if t1 == t0:
                            ratio = 0
                        else:
                            ratio = (target_time - t0).total_seconds() / (t1 - t0).total_seconds()

                        result.append({
                            'lat': parsed[i]['lat'] + (parsed[i+1]['lat'] - parsed[i]['lat']) * ratio,
                            'lng': parsed[i]['lng'] + (parsed[i+1]['lng'] - parsed[i]['lng']) * ratio,
                            'pressure': parsed[i]['pressure'] + (parsed[i+1]['pressure'] - parsed[i]['pressure']) * ratio,
                            'wind_speed': parsed[i]['wind_speed'] + (parsed[i+1]['wind_speed'] - parsed[i]['wind_speed']) * ratio,
                        })
                        break
                else:
                    # target_time在第一个点之前
                    result.append({
                        'lat': parsed[0]['lat'],
                        'lng': parsed[0]['lng'],
                        'pressure': parsed[0]['pressure'],
                        'wind_speed': parsed[0]['wind_speed'],
                    })
            else:
                # 超出机构预报范围：用最后两个点的外推
                if len(parsed) >= 2:
                    t0 = parsed[-2]['time']
                    t1 = parsed[-1]['time']
                    if t1 > t0:
                        ratio = (target_time - t1).total_seconds() / (t1 - t0).total_seconds()
                        result.append({
                            'lat': parsed[-1]['lat'] + (parsed[-1]['lat'] - parsed[-2]['lat']) * ratio,
                            'lng': parsed[-1]['lng'] + (parsed[-1]['lng'] - parsed[-2]['lng']) * ratio,
                            'pressure': parsed[-1]['pressure'],
                            'wind_speed': parsed[-1]['wind_speed'],
                        })
                    else:
                        result.append({
                            'lat': parsed[-1]['lat'],
                            'lng': parsed[-1]['lng'],
                            'pressure': parsed[-1]['pressure'],
                            'wind_speed': parsed[-1]['wind_speed'],
                        })
                else:
                    result.append({
                        'lat': parsed[-1]['lat'],
                        'lng': parsed[-1]['lng'],
                        'pressure': parsed[-1]['pressure'],
                        'wind_speed': parsed[-1]['wind_speed'],
                    })

        return result

    @staticmethod
    def _trend_extrapolation(points, hours, agency_calibration=None):
        """趋势外推法 - 基于最近几个数据点的移动趋势
        修正：增加纬度加速因子 + 按时间标准化加速度 + 机构速度校准
        agency_calibration: 当机构预报可用时，提供共识速度校准，避免系统性偏慢"""
        # 取最近 N 个点计算趋势
        n = min(8, len(points))
        recent = points[-n:]

        # 计算平均移动向量（度/小时）
        total_lat_rate = 0
        total_lng_rate = 0
        valid_steps = 0

        for i in range(1, len(recent)):
            dlat = recent[i]['lat'] - recent[i-1]['lat']
            dlng = recent[i]['lng'] - recent[i-1]['lng']

            # 时间差（小时）
            try:
                t1 = datetime.fromisoformat(recent[i]['time'].replace('Z', '+00:00'))
                t2 = datetime.fromisoformat(recent[i-1]['time'].replace('Z', '+00:00'))
                dt_hours = (t1 - t2).total_seconds() / 3600
            except:
                dt_hours = 3  # 默认3小时间隔

            if dt_hours > 0:
                total_lat_rate += dlat / dt_hours
                total_lng_rate += dlng / dt_hours
                valid_steps += 1

        avg_lat_rate = total_lat_rate / max(valid_steps, 1) if valid_steps > 0 else 0  # 度/小时
        avg_lng_rate = total_lng_rate / max(valid_steps, 1) if valid_steps > 0 else 0

        # ★★★ 机构速度校准 ★★★
        # 当机构预报可用时，混合机构共识速度与观测速度
        # 混合比例: 机构70% + 观测30%（机构NWP模型能预测未来加速，观测只反映过去）
        # 如果观测速度比机构快（台风已加速），则尊重观测数据（混合比例反转）
        if agency_calibration:
            a_lat = agency_calibration['lat_rate']
            a_lng = agency_calibration['lng_rate']

            # 判断哪个更快：如果观测速度已经比机构共识快（台风已加速），尊重观测
            obs_speed = math.sqrt(avg_lat_rate**2 + avg_lng_rate**2)
            agency_speed = math.sqrt(a_lat**2 + a_lng**2)

            if obs_speed >= agency_speed:
                # 观测已经更快（台风已加速到机构预期水平），尊重观测
                blend_ratio = 0.4  # 机构40%，观测60%
            else:
                # 观测比机构慢（台风还没加速），用机构速度主导
                blend_ratio = 0.7  # 机构70%，观测30%

            avg_lat_rate = blend_ratio * a_lat + (1 - blend_ratio) * avg_lat_rate
            avg_lng_rate = blend_ratio * a_lng + (1 - blend_ratio) * avg_lng_rate

        # 计算加速度（按时间标准化）—— 前半段 vs 后半段的速度差异
        if len(recent) >= 4:
            half = len(recent) // 2
            first_half = recent[:half]
            second_half = recent[half:]

            # 前半段平均速度（度/小时）
            rate1_lat = rate1_lng = 0
            steps1 = 0
            for i in range(1, len(first_half)):
                dlat = first_half[i]['lat'] - first_half[i-1]['lat']
                dlng = first_half[i]['lng'] - first_half[i-1]['lng']
                try:
                    dt_h = (datetime.fromisoformat(first_half[i]['time'].replace('Z', '+00:00')) -
                            datetime.fromisoformat(first_half[i-1]['time'].replace('Z', '+00:00'))).total_seconds() / 3600
                except:
                    dt_h = 3
                if dt_h > 0:
                    rate1_lat += dlat / dt_h
                    rate1_lng += dlng / dt_h
                    steps1 += 1

            # 后半段平均速度（度/小时）
            rate2_lat = rate2_lng = 0
            steps2 = 0
            for i in range(1, len(second_half)):
                dlat = second_half[i]['lat'] - second_half[i-1]['lat']
                dlng = second_half[i]['lng'] - second_half[i-1]['lng']
                try:
                    dt_h = (datetime.fromisoformat(second_half[i]['time'].replace('Z', '+00:00')) -
                            datetime.fromisoformat(second_half[i-1]['time'].replace('Z', '+00:00'))).total_seconds() / 3600
                except:
                    dt_h = 3
                if dt_h > 0:
                    rate2_lat += dlat / dt_h
                    rate2_lng += dlng / dt_h
                    steps2 += 1

            # 速度变化（度/小时²）：后半段速度 - 前半段速度 / 时间跨度
            avg_rate1_lat = rate1_lat / max(steps1, 1)
            avg_rate1_lng = rate1_lng / max(steps1, 1)
            avg_rate2_lat = rate2_lat / max(steps2, 1)
            avg_rate2_lng = rate2_lng / max(steps2, 1)

            # 加速度 = 速度变化率（度/小时²），保守估计：只用一小部分
            accel_lat = (avg_rate2_lat - avg_rate1_lat) * 0.15  # 度/小时², 15%权重
            accel_lng = (avg_rate2_lng - avg_rate1_lng) * 0.15
        else:
            accel_lat = 0
            accel_lng = 0

        # 生成预测点 —— 增量式计算
        # ★★★ 核心设计：
        #   有机构校准：直接使用校准后的速度，不再叠加纬度加速
        #     （机构NWP模型已预测了未来加速，不需要我们再猜测）
        #   无机构校准：观测速度+极小增量加速（避免正反馈爆炸）
        #     加速因子只补偿"从当前纬度起再增加的加速"，不从头起算
        last_point = points[-1]
        predictions = []
        dt = 6  # 每6小时一个预测点

        current_lat = last_point['lat']
        current_lng = last_point['lng']
        current_pressure = last_point.get('pressure', 1000)
        current_wind = last_point.get('wind_speed', 0)

        # 计算压力/风速趋势（每6h步的变化量）
        recent_pressures = [p.get('pressure', 0) for p in recent if p.get('pressure')]
        recent_winds = [p.get('wind_speed', 0) for p in recent if p.get('wind_speed')]
        p_rate_per_6h = 0
        w_rate_per_6h = 0
        if len(recent_pressures) >= 2:
            p_rate_per_6h = (recent_pressures[-1] - recent_pressures[0]) / max(len(recent_pressures)-1, 1) * 2
        if len(recent_winds) >= 2:
            w_rate_per_6h = (recent_winds[-1] - recent_winds[0]) / max(len(recent_winds)-1, 1) * 2

        try:
            base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
        except:
            base_time = datetime.now()

        # 记录起始纬度（用于计算增量加速）
        start_lat = current_lat

        for step in range(1, hours // dt + 1):
            h = step * dt

            # 置信度衰减因子：仅影响confidence，不影响位置
            decay = math.exp(-h / (hours * 1.5))

            # ★ 纬度加速策略（取决于是否有机构校准）
            if agency_calibration:
                # 有机构校准：NWP模型已预测未来加速，不再叠加额外加速
                # 只在极北纬度(>35°N)加微小转向加速
                effective_lat_rate = avg_lat_rate
                if current_lat > 35:
                    effective_lat_rate += 0.03  # 极北区域转向时略加速
                effective_lng_rate = avg_lng_rate
            else:
                # 无机构校准：保守的增量加速
                # 只补偿"从当前纬度起额外增加的加速"
                # NW Pacific实测：每10°纬度北上速度约增加25-30%
                # 但只用15%/10°避免正反馈（观测速度已包含部分加速）
                lat_progress = max(0, current_lat - start_lat)
                small_boost = 1.0 + 0.015 * lat_progress  # 每北移10°增加15%
                MIN_RATE = 0.13  # 度/小时 ≈ 15km/h
                effective_lat_rate = max(avg_lat_rate * small_boost, MIN_RATE)
                effective_lng_rate = avg_lng_rate

            # ★ 转向修正：基于当前纬度
            if current_lat > 25:
                recurvature = min((current_lat - 25) / 15, 1.0)  # 25°N→40°N逐渐转向
                # 西移逐渐减弱，东移分量增加
                effective_lng_rate = effective_lng_rate * (1.0 - recurvature * 0.6) + recurvature * 0.15

            # 增量更新位置
            current_lat += effective_lat_rate * dt
            current_lng += effective_lng_rate * dt
            current_pressure += p_rate_per_6h
            current_wind += w_rate_per_6h

            pred_time = (base_time + timedelta(hours=h)).isoformat()

            predictions.append({
                'time': pred_time,
                'lat': round(current_lat, 1),
                'lng': round(current_lng, 1),
                'pressure': round(current_pressure),
                'wind_speed': round(max(0, current_wind), 1),
                'category': TyphoonPredictor._intensity_category(current_pressure, current_wind),
                'confidence': round(0.85 * decay, 2),
            })

        return predictions

    @staticmethod
    def _physics_model(points, hours, agency_calibration=None):
        """物理模型预测 - Beta漂移 + 引导气流 + 简化科里奥利力
        agency_calibration: 机构共识速度校准，避免预测偏慢"""
        last = points[-1]
        lat = last['lat']
        lng = last['lng']

        # 确定季节参数
        try:
            base_time = datetime.fromisoformat(last['time'].replace('Z', '+00:00'))
            month = base_time.month
        except:
            month = 8

        if month >= 6 and month <= 9:
            season = 'summer'
        elif month >= 10 and month <= 11:
            season = 'autumn'
        else:
            season = 'winter'

        params = TyphoonPredictor.STEERING_FLOW_PARAMS[season]

        # 计算当前移动向量 —— 按时间标准化为度/小时
        n = min(6, len(points))
        recent = points[-n:]
        move_lat_rate = move_lng_rate = 0
        valid_steps = 0
        for i in range(1, len(recent)):
            dlat = recent[i]['lat'] - recent[i-1]['lat']
            dlng = recent[i]['lng'] - recent[i-1]['lng']
            try:
                t1 = datetime.fromisoformat(recent[i]['time'].replace('Z', '+00:00'))
                t2 = datetime.fromisoformat(recent[i-1]['time'].replace('Z', '+00:00'))
                dt_h = (t1 - t2).total_seconds() / 3600
            except:
                dt_h = 3  # 默认3小时
            if dt_h > 0:
                move_lat_rate += dlat / dt_h  # 度/小时
                move_lng_rate += dlng / dt_h
                valid_steps += 1

        if valid_steps > 0:
            move_lat_rate /= valid_steps
            move_lng_rate /= valid_steps
        else:
            move_lat_rate = 0
            move_lng_rate = 0

        # ★★★ 机构速度校准 ★★★
        if agency_calibration:
            a_lat = agency_calibration['lat_rate']
            a_lng = agency_calibration['lng_rate']
            obs_speed = math.sqrt(move_lat_rate**2 + move_lng_rate**2)
            agency_speed = math.sqrt(a_lat**2 + a_lng**2)
            if obs_speed >= agency_speed:
                blend_ratio = 0.4
            else:
                blend_ratio = 0.7
            move_lat_rate = blend_ratio * a_lat + (1 - blend_ratio) * move_lat_rate
            move_lng_rate = blend_ratio * a_lng + (1 - blend_ratio) * move_lng_rate

        # 科里奥利参数 (简化)
        omega = 7.292e-5  # 地球自转角速度
        f = 2 * omega * math.sin(math.radians(lat))

        # Beta漂移效应: 台风向北偏移约 1-2 度/天，向西偏移约 0.5-1.5 度/天
        beta_n = params['beta_drift_n'] / 24  # 度/小时
        beta_w = params['beta_drift_w'] / 24

        # 引导气流 + Beta漂移
        dt = 6
        predictions = []
        current_lat = lat
        current_lng = lng
        current_pressure = last.get('pressure', 1000)
        current_wind = last.get('wind_speed', 0)
        start_lat = current_lat  # 记录起始纬度

        for h in range(dt, hours + 1, dt):
            # 置信度衰减，不影响位置计算
            decay = math.exp(-h / (hours * 2))

            # ★ 纬度加速策略（与趋势法相同：有机构校准时不叠加额外加速）
            if agency_calibration:
                effective_lat_rate = move_lat_rate
                if current_lat > 35:
                    effective_lat_rate += 0.03
                effective_lng_rate = move_lng_rate
            else:
                # 保守增量加速（避免正反馈爆炸）
                lat_progress = max(0, current_lat - start_lat)
                small_boost = 1.0 + 0.015 * lat_progress
                MIN_RATE = 0.13
                effective_lat_rate = max(move_lat_rate * small_boost, MIN_RATE)
                effective_lng_rate = move_lng_rate

            # ★ 转向修正：高纬时西移减速并转为东移
            if current_lat > 25:
                recurvature = min((current_lat - 25) / 15, 1.0)  # 25°N→40°N逐渐转向
                effective_lng_rate = effective_lng_rate * (1.0 - recurvature * 0.6) + recurvature * 0.15

            # 预测位置: 引导气流(度/小时) × dt(小时) + beta漂移 × dt
            pred_lat = current_lat + (effective_lat_rate + beta_n) * dt
            pred_lng = current_lng + (effective_lng_rate - beta_w) * dt

            # 强度预测: 基于海温简化模型
            # 海上台风增强，登陆后减弱
            is_over_ocean = pred_lng > 105 and pred_lat < 35
            if is_over_ocean:
                pressure_change = -2 * dt  # 继续降低气压
                wind_change = 3 * dt
            else:
                pressure_change = 8 * dt  # 气压升高（减弱）
                wind_change = -5 * dt

            current_pressure = current_pressure + pressure_change
            current_wind = max(0, current_wind + wind_change)
            current_lat = pred_lat
            current_lng = pred_lng

            try:
                base_time = datetime.fromisoformat(last['time'].replace('Z', '+00:00'))
                pred_time = (base_time + timedelta(hours=h)).isoformat()
            except:
                pred_time = f"+{h}h"

            predictions.append({
                'time': pred_time,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(current_pressure),
                'wind_speed': round(current_wind, 1),
                'category': TyphoonPredictor._intensity_category(current_pressure, current_wind),
                'confidence': round(0.70 * decay, 2),
            })

        return predictions

    @staticmethod
    def _ensemble_merge(trend_preds, physics_preds, hours):
        """综合预报 - 趋势外推 + 物理模型加权融合"""
        if len(trend_preds) != len(physics_preds):
            min_len = min(len(trend_preds), len(physics_preds))
            trend_preds = trend_preds[:min_len]
            physics_preds = physics_preds[:min_len]

        ensemble = []
        for i, (t, p) in enumerate(zip(trend_preds, physics_preds)):
            # 权重随时间变化: 近期趋势权重高，远期物理权重高
            time_ratio = i / max(len(trend_preds), 1)
            trend_weight = 0.6 * (1 - time_ratio) + 0.3
            physics_weight = 1 - trend_weight

            merged = {
                'time': t['time'],
                'lat': round(t['lat'] * trend_weight + p['lat'] * physics_weight, 1),
                'lng': round(t['lng'] * trend_weight + p['lng'] * physics_weight, 1),
                'pressure': round(t['pressure'] * trend_weight + p['pressure'] * physics_weight),
                'wind_speed': round(t['wind_speed'] * trend_weight + p['wind_speed'] * physics_weight, 1),
                'category': TyphoonPredictor._intensity_category(
                    t['pressure'] * trend_weight + p['pressure'] * physics_weight,
                    t['wind_speed'] * trend_weight + p['wind_speed'] * physics_weight,
                ),
                'confidence': round(min(t['confidence'], p['confidence']) * 0.95, 2),
                'trend_lat': t['lat'],
                'trend_lng': t['lng'],
                'physics_lat': p['lat'],
                'physics_lng': p['lng'],
            }
            ensemble.append(merged)

        return ensemble

    @staticmethod
    def _intensity_category(pressure, wind_speed):
        """根据气压和风速判断台风等级"""
        if pressure <= 935 or wind_speed >= 51:
            return '超强台风(Super TY)'
        elif pressure <= 955 or wind_speed >= 42:
            return '强台风(STY)'
        elif pressure <= 970 or wind_speed >= 35:
            return '台风(TY)'
        elif pressure <= 985 or wind_speed >= 25:
            return '强热带风暴(STS)'
        elif pressure <= 1000 or wind_speed >= 18:
            return '热带风暴(TS)'
        else:
            return '热带低压(TD)'

    @staticmethod
    def _calculate_confidence(points, hours):
        """计算预测置信度"""
        n = len(points)
        if n < 3:
            return 0.3
        if n < 6:
            return 0.5
        if n < 12:
            return 0.7

        # 数据越密集，置信度越高
        base_confidence = min(0.9, 0.5 + n * 0.03)

        # 预测时间越远，置信度越低（7天仍有60-70%参考价值）
        time_decay = math.exp(-hours / 200)

        return round(base_confidence * time_decay, 2)


# ============================================================
# API 路由
# ============================================================

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/typhoons/current')
def get_current_typhoons():
    """获取当前活跃台风"""
    data = fetch_isc_current_typhoons()
    normalized = normalize_isc_data(data)
    return jsonify({
        'count': len(normalized),
        'typhoons': normalized,
        'update_time': datetime.now().isoformat()
    })


@app.route('/api/typhoons/year/<int:year>')
def get_year_typhoons(year):
    """获取指定年份所有台风"""
    data = fetch_isc_year_list(year)
    normalized = normalize_isc_data(data)
    return jsonify({
        'year': year,
        'count': len(normalized),
        'typhoons': normalized
    })


@app.route('/api/typhoons/detail/<tfbh>')
def get_typhoon_detail(tfbh):
    """获取单个台风详情"""
    year = int(tfbh[:4])
    month = int(tfbh[4:6])
    year_month = f"{year}{str(month).zfill(2)}"
    all_data = fetch_isc_typhoon_data(year_month)

    target = None
    for t in all_data:
        if t.get('tfbh') == tfbh or t.get('ident') == tfbh:
            target = t
            break

    if target:
        normalized = normalize_isc_data([target])[0]
        return jsonify(normalized)

    # 尝试从 NII 获取
    nii_data = fetch_nii_typhoon_geojson(tfbh)
    if nii_data:
        normalized = normalize_nii_data(nii_data, tfbh)
        if normalized:
            return jsonify(normalized)

    return jsonify({'error': f'台风 {tfbh} 数据未找到'}), 404


@app.route('/api/typhoons/predict/<tfbh>')
def predict_typhoon(tfbh):
    """预测台风路径"""
    hours = request.args.get('hours', 72, type=int)
    method = request.args.get('method', 'ensemble')

    # 获取台风数据 - 搜索全年（tfbh是台风编号，不是年月）
    year = int(tfbh[:4])
    target = None
    for month in range(1, 13):
        year_month = f"{year}{str(month).zfill(2)}"
        all_data = fetch_isc_typhoon_data(year_month)
        for t in all_data:
            if t.get('tfbh') == tfbh or t.get('ident') == tfbh:
                target = t
                break
        if target:
            break

    if target:
        normalized = normalize_isc_data([target])[0]
    else:
        nii_data = fetch_nii_typhoon_geojson(tfbh)
        if nii_data:
            normalized = normalize_nii_data(nii_data, tfbh)
        else:
            return jsonify({'error': f'台风 {tfbh} 数据未找到'}), 404

    result = TyphoonPredictor.predict_path(normalized, hours=hours, method=method)
    return jsonify(result)


@app.route('/api/typhoons/search')
def search_typhoons():
    """搜索台风"""
    keyword = request.args.get('keyword', '')
    year = request.args.get('year', datetime.now().year, type=int)

    data = fetch_isc_year_list(year)
    normalized = normalize_isc_data(data)

    if keyword:
        results = [t for t in normalized if
                   keyword.lower() in t.get('name_cn', '').lower() or
                   keyword.lower() in t.get('name_en', '').lower() or
                   keyword in t.get('id', '')]
    else:
        results = normalized

    return jsonify({
        'keyword': keyword,
        'year': year,
        'count': len(results),
        'typhoons': results
    })


@app.route('/api/typhoons/years-list')
def get_available_years():
    """获取可用年份列表"""
    current_year = datetime.now().year
    years = list(range(1945, current_year + 1))
    return jsonify({'years': years})


@app.route('/api/data/status')
def data_status():
    """查询本地缓存数据状态"""
    current_year = datetime.now().year

    # 检查本地缓存文件
    cached_years = {}
    for year in range(1945, current_year + 1):
        month_count = 0
        for month in range(1, 13):
            ym = f"{year}{str(month).zfill(2)}"
            if os.path.exists(os.path.join(ISC_DIR, f'{ym}.json')):
                month_count += 1
        if month_count > 0:
            cached_years[year] = month_count

    # 当前年份是否完整缓存
    current_cached = cached_years.get(current_year, 0)
    prev_cached = cached_years.get(current_year - 1, 0)

    return jsonify({
        'current_year': current_year,
        'cached_years': cached_years,
        'current_year_cached_months': current_cached,
        'prev_year_cached_months': prev_cached,
        'total_cached_files': len([f for f in os.listdir(ISC_DIR) if f.endswith('.json')]),
        'lstm_ready': is_lstm_ready(),
        'pangu_ready': is_pangu_ready(),
        'pangu_detail': _get_pangu_detail(),
        'ecmwf_bufr_available': is_bufr_available(),
    })


def _get_pangu_detail():
    """获取Pangu模型的详细状态信息"""
    from pangu_predictor import PANGU_24H_MODEL, PANGU_6H_MODEL
    detail = {'ready': False, 'models': {}}
    for name, path in [('24h', PANGU_24H_MODEL), ('6h', PANGU_6H_MODEL)]:
        exists = os.path.exists(path)
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 1) if exists else 0
        detail['models'][name] = {
            'exists': exists,
            'size_mb': size_mb,
            'ready': exists and size_mb > 100,
            'path': path,
        }
    detail['ready'] = all(m['ready'] for m in detail['models'].values())
    return detail


# ---- Pangu-Weather 模型下载 API ----

_pangu_download_thread = None
_pangu_download_lock = threading.Lock()

@app.route('/api/pangu/download', methods=['POST'])
def trigger_pangu_download():
    """手动触发Pangu-Weather模型下载（后台线程）"""
    global _pangu_download_thread

    # 检查是否已有下载在进行
    with _pangu_download_lock:
        if _pangu_download_thread and _pangu_download_thread.is_alive():
            return jsonify({'status': 'already_downloading', 'message': '下载正在进行中...'})

        from pangu_downloader import download_pangu_models, _write_status
        _write_status('downloading', '用户手动触发下载...', 0)

        def _bg_download():
            try:
                download_pangu_models()
            except Exception as e:
                from pangu_downloader import _write_status
                _write_status('failed', f'下载异常: {str(e)[:200]}', 0)

        _pangu_download_thread = threading.Thread(target=_bg_download, daemon=True)
        _pangu_download_thread.start()

    return jsonify({
        'status': 'started',
        'message': 'Pangu模型下载已启动，请稍后查看状态...',
        'estimated_time': '约5-15分钟（取决于网速）'
    })


@app.route('/api/pangu/download-status')
def pangu_download_status():
    """查询Pangu模型下载状态"""
    from pangu_downloader import get_download_status
    status = get_download_status()
    status['pangu_ready'] = is_pangu_ready()
    return jsonify(status)


@app.route('/api/pangu/manual-download-info')
def pangu_manual_download_info():
    """返回Pangu模型手动下载说明（百度网盘地址等）"""
    from pangu_downloader import get_manual_download_info
    return jsonify(get_manual_download_info())


@app.route('/api/data/cache', methods=['POST'])
def cache_data():
    """批量缓存指定年份范围的ISC数据到本地"""
    params = request.json or {}
    start_year = params.get('start_year', datetime.now().year - 5)
    end_year = params.get('end_year', datetime.now().year)

    results = {
        'success': [],
        'failed': [],
        'already_cached': [],
    }

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            ym = f"{year}{str(month).zfill(2)}"
            local_file = os.path.join(ISC_DIR, f'{ym}.json')

            if os.path.exists(local_file):
                results['already_cached'].append(ym)
                continue

            url = f'https://data.istrongcloud.com/v2/data/complex/{ym}.json'
            try:
                response = req_lib.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    with open(local_file, 'w') as f:
                        json.dump(data, f)
                    results['success'].append(ym)
                else:
                    results['failed'].append({'ym': ym, 'status': response.status_code})
            except Exception as e:
                results['failed'].append({'ym': ym, 'error': str(e)})

    return jsonify({
        'message': f'缓存完成: {len(results["success"])}新下载, {len(results["already_cached"])}已有, {len(results["failed"])}失败',
        'results': results,
    })


@app.route('/api/data/cache-current-year')
def cache_current_year():
    """快速缓存当前年份和前一年数据（用于首次访问）"""
    current_year = datetime.now().year
    results = {'success': 0, 'failed': 0, 'already_cached': 0}

    for year in [current_year - 1, current_year]:
        for month in range(1, 13):
            ym = f"{year}{str(month).zfill(2)}"
            local_file = os.path.join(ISC_DIR, f'{ym}.json')

            if os.path.exists(local_file):
                results['already_cached'] += 1
                continue

            url = f'https://data.istrongcloud.com/v2/data/complex/{ym}.json'
            try:
                response = req_lib.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    with open(local_file, 'w') as f:
                        json.dump(data, f)
                    results['success'] += 1
                else:
                    results['failed'] += 1
            except:
                results['failed'] += 1

    return jsonify({
        'current_year': current_year,
        'results': results,
        'message': f'当前年{current_year}和前一年数据: {results["success"]}新缓存, {results["already_cached"]}已有',
    })


@app.route('/api/stats/intensity-distribution/<int:year>')
def intensity_distribution(year):
    """获取指定年份台风强度分布统计"""
    data = fetch_isc_year_list(year)
    normalized = normalize_isc_data(data)

    distribution = {}
    for t in normalized:
        # 取台风最强等级
        max_power = max([p.get('power', 0) for p in t['points']], default=0)
        cat = TyphoonPredictor._intensity_category(
            min([p.get('pressure', 9999) for p in t['points']], default=9999),
            max([p.get('wind_speed', 0) for p in t['points']], default=0)
        )
        if cat not in distribution:
            distribution[cat] = 0
        distribution[cat] += 1

    return jsonify({
        'year': year,
        'total': len(normalized),
        'distribution': distribution
    })


@app.route('/api/lstm/status')
def lstm_status():
    """查询LSTM模型状态"""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'lstm_best.pt')
    history_path = os.path.join(os.path.dirname(__file__), 'models', 'training_history.json')

    status = {
        'model_exists': os.path.exists(model_path),
        'ready': is_lstm_ready(),
    }

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        status['training_history'] = {
            'epochs_trained': history.get('epochs_trained', 0),
            'best_val_loss': history.get('best_val_loss', 0),
        }

    return jsonify(status)


@app.route('/api/lstm/train', methods=['POST'])
def lstm_train():
    """训练LSTM模型（需要先有ISC历史数据）"""
    params = request.json or {}
    years = params.get('years', list(range(2015, 2025)))
    epochs = params.get('epochs', 80)
    batch_size = params.get('batch_size', 32)
    lr = params.get('lr', 0.001)

    trainer = LSTMTrainer()

    # 加载训练数据
    X, Y, typhoon_count = trainer.load_training_data(years)
    if X is None or typhoon_count == 0:
        return jsonify({
            'error': '训练数据不足，请先下载ISC历史数据（访问/api/typhoons/year/年份 来触发缓存）',
            'suggestion': '建议先访问多个年份的数据: 2015-2024'
        }), 400

    # 训练
    history = trainer.train(X, Y, epochs=epochs, batch_size=batch_size, lr=lr)

    # 计算训练质量指标
    best_loss = history.get('best_val_loss', 0)
    error_km = math.sqrt(best_loss) * 50 * 111  # 粗略估算

    return jsonify({
        'success': True,
        'typhoon_count': typhoon_count,
        'sequence_count': len(X),
        'epochs_trained': history.get('epochs_trained', 0),
        'best_val_loss': best_loss,
        'estimated_error_km': round(error_km, 1),
        'model_path': os.path.join(trainer.model_dir, 'lstm_best.pt'),
        'message': f'LSTM模型训练完成！基于{typhoon_count}个台风训练，估计位置误差≈{round(error_km)}km'
    })


@app.route('/api/predictions/cached/<tfid>/<int:hours>')
def get_cached_prediction(tfid, hours):
    """
    从缓存中读取自动计算的预测结果（前端秒级查询）
    
    优先架构: 前端先查缓存 → 缓存新鲜则秒级返回 → 
    缓存过时/缺失则触发按需计算 + 降级实时计算
    """
    from scheduler import get_cached_prediction, compute_on_demand
    method = request.args.get('method', None)
    result = get_cached_prediction(tfid, hours, method=method)
    if result:
        return jsonify(result)
    
    # 无缓存 → 触发后台按需计算 + 前端降级到实时计算
    compute_on_demand(tfid, hours)
    return jsonify({
        'cached': False,
        'message': '无缓存预测，已触发后台计算',
        'fallback_hint': '前端应回退到实时预测API',
    })


@app.route('/api/predictions/trigger/<tfid>/<int:hours>')
def trigger_on_demand_prediction(tfid, hours):
    """
    触发后台按需预测计算（立即返回，计算在后台线程中完成）
    前端可在展示实时计算结果后调用此API，为下次访问缓存结果
    """
    from scheduler import compute_on_demand
    method = request.args.get('method', 'all')
    compute_on_demand(tfid, hours, method=method)
    return jsonify({
        'triggered': True,
        'tfid': tfid,
        'hours': hours,
        'message': '后台计算已启动，约30秒后可查询缓存',
    })


@app.route('/api/scheduler/status')
def scheduler_status():
    """查询调度引擎v2完整状态"""
    from scheduler import get_scheduler_status
    return jsonify(get_scheduler_status())


@app.route('/api/predictions/active-cache')
def get_active_cached_predictions():
    """
    ★ 批量获取所有活跃台风的缓存预测结果
    前端启动时调用此接口，一次性获取所有已缓存的数据
    不需要用户选择台风，后台已经计算好了
    """
    from scheduler import get_active_cache_coverage, get_cached_prediction, PREDICTION_HOURS
    hours = request.args.get('hours', 168, type=int)

    coverage = get_active_cache_coverage()
    results = []

    for item in coverage:
        tfid = item['tfid']
        # 读取缓存（前端最常用的168h）
        cached = get_cached_prediction(tfid, hours)
        if cached and cached.get('predictions'):
            results.append({
                'tfid': tfid,
                'name_cn': item.get('name_cn', ''),
                'name_en': item.get('name_en', ''),
                'hours': hours,
                'predictions': cached.get('predictions', {}),
                'cache_age_minutes': cached.get('cache_age_minutes', 0),
                'cache_fresh': cached.get('cache_fresh', False),
                'computed_at': cached.get('computed_at', ''),
                'available_methods': list(cached.get('predictions', {}).keys()),
            })

    return jsonify({
        'success': True,
        'active_count': len(coverage),
        'cached_count': len(results),
        'predictions': results,
        'message': f'活跃台风{len(coverage)}个, 已缓存{len(results)}个预测',
    })


@app.route('/api/predictions/active-cache-all')
def get_active_cached_predictions_all_hours():
    """
    ★ 批量获取所有活跃台风的所有时长缓存预测
    前端可以一次性获取24h/48h/72h/120h/168h/240h的所有缓存
    """
    from scheduler import get_active_cache_coverage, get_cached_prediction, PREDICTION_HOURS

    coverage = get_active_cache_coverage()
    results = {}

    for item in coverage:
        tfid = item['tfid']
        results[tfid] = {
            'name_cn': item.get('name_cn', ''),
            'name_en': item.get('name_en', ''),
            'hours_data': {},
        }
        for h in PREDICTION_HOURS:
            cached = get_cached_prediction(tfid, h)
            if cached and cached.get('predictions'):
                results[tfid]['hours_data'][h] = {
                    'predictions': cached.get('predictions', {}),
                    'cache_age_minutes': cached.get('cache_age_minutes', 0),
                    'cache_fresh': cached.get('cache_fresh', False),
                    'computed_at': cached.get('computed_at', ''),
                    'available_methods': list(cached.get('predictions', {}).keys()),
                }

    return jsonify({
        'success': True,
        'active_count': len(coverage),
        'cached_typhoons': list(results.keys()),
        'data': results,
    })


@app.route('/api/scheduler/init-status')
def scheduler_init_status():
    """查询启动初始化是否完成（前端轮询此接口判断缓存是否可用）"""
    from scheduler import _scheduler_state
    return jsonify({
        'init_done': _scheduler_state.get('startup_init_done', False),
        'active_count': _scheduler_state.get('active_typhoon_count', 0),
        'cached_count': _scheduler_state.get('cached_predictions_count', 0),
        'on_demand_queue': _scheduler_state.get('on_demand_queue_size', 0),
        'last_prediction': _scheduler_state.get('last_prediction', ''),
        'last_method': _scheduler_state.get('last_method', ''),
    })


@app.route('/api/prediction-methods')
def get_prediction_methods():
    """获取所有可用的预测方法及其说明"""
    methods = [
        {
            'id': 'trend',
            'name': '趋势外推',
            'accuracy': '低',
            'description': '基于最近6个数据点的移动趋势和加速度，简单快速但远期误差大',
            'color': '#3b82f6',
            'best_for': '6-12小时极短期',
        },
        {
            'id': 'physics',
            'name': '物理模型',
            'accuracy': '中',
            'description': 'Beta漂移+引导气流+科里奥利力简化模型，含季节参数',
            'color': '#a855f7',
            'best_for': '12-48小时短期',
        },
        {
            'id': 'gfs',
            'name': 'GFS涡旋追踪',
            'accuracy': '高',
            'description': 'NOAA GFS模型MSLP气压场网格追踪台风低压中心，业务预报标准方法',
            'color': '#f59e0b',
            'best_for': '24-120小时中期',
        },
        {
            'id': 'ecmwf',
            'name': 'ECMWF IFS',
            'accuracy': '极高',
            'description': '欧洲中期天气预报中心确定性预报MSLP涡旋追踪，全球最准NWP之一',
            'color': '#22c55e',
            'best_for': '24-120小时中期（最权威NWP）',
        },
        {
            'id': 'gfs_graphcast',
            'name': 'GraphCast AI',
            'accuracy': '较高',
            'description': 'DeepMind开发AI天气模型，超越传统NWP',
            'color': '#ef4444',
            'best_for': '24-120小时（当前台风）',
        },
        {
            'id': 'aifs',
            'name': 'ECMWF AIFS AI',
            'accuracy': '高',
            'description': 'ECMWF的AI预报系统MSLP涡旋追踪',
            'color': '#14b8a6',
            'best_for': '24-120小时中期',
        },
        {
            'id': 'cma',
            'name': 'CMA GRAPES',
            'accuracy': '高',
            'description': '中国气象局全球数值预报MSLP涡旋追踪，对西北太平洋台风有本地优势',
            'color': '#f97316',
            'best_for': '24-72小时短期（西太平洋台风最佳）',
        },
        {
            'id': 'analog',
            'name': '历史类比法',
            'accuracy': '中',
            'description': '从1945年以来数据中找相似台风，加权预测，经典气象学方法',
            'color': '#84cc16',
            'best_for': '路径稳定的台风',
        },
        {
            'id': 'lstm',
            'name': 'LSTM深度学习',
            'accuracy': '较高',
            'description': '8特征LSTM网络(位置+气压+风速+移动方向+季节)，递推预测，需要先训练模型',
            'color': '#ec4899',
            'best_for': '6-72小时全时段',
            'requires_training': True,
        },
        {
            'id': 'pangu',
            'name': '盘古大模型(Pangu-Weather)',
            'accuracy': '极高(需设置)',
            'description': '华为盘古天气大模型(Nature 2022论文)，在台风追踪上优于ECMWF HRES，需下载ONNX权重+ERA5数据',
            'color': '#8b5cf6',
            'best_for': '6-120小时全时段(最先进AI天气模型)',
            'requires_setup': True,
        },
        {
            'id': 'ecmwf_bufr',
            'name': 'ECMWF BUFR官方轨迹',
            'accuracy': '极高',
            'description': '直接获取ECMWF官方台风轨迹BUFR预报数据(24h误差~50km)，含HRES确定性预报+ENS集合均值，无需自建NWP',
            'color': '#00d4aa',
            'best_for': '24-240小时全时段(全球最权威NWP官方输出)',
            'requires_dependency': True,  # 需要ecmwf-opendata+eccodes
        },
        {
            'id': 'ensemble',
            'name': '综合融合(含机构)',
            'accuracy': '最高',
            'description': 'Kalman融合所有方法+6家气象机构预报(中/日/美/欧/韩/港)加权锚定，机构预报占72%权重',
            'color': '#06b6d4',
            'best_for': '全时段综合最佳（推荐）',
        },
    ]

    return jsonify({
        'methods': methods,
        'lstm_ready': is_lstm_ready(),
        'recommendation': '建议使用 ensemble 或 lstm 方法获得最准确的预测。'
                          'LSTM需要先训练：POST /api/lstm/train'
    })


@app.route('/api/coastline')
def get_coastline():
    """返回NW Pacific海岸线GeoJSON（用于前端地图展示）"""
    return jsonify(get_coastline_geojson())


@app.route('/api/ecmwf/bufr/status')
def ecmwf_bufr_status():
    """检查 ECMWF BUFR 数据获取功能状态"""
    return jsonify({
        'bufr_available': is_bufr_available(),
        'dependencies': {
            'ecmwf_opendata': _check_import('ecmwf.opendata'),
            'pdbufr': _check_import('pdbufr'),
            'eccodes': _check_import('eccodes'),
        },
        'description': 'ECMWF BUFR需要ecmwf-opendata+eccodes/pdbufr依赖，Docker需安装libeccodes-dev',
    })


@app.route('/api/ecmwf/bufr/active-storms')
def ecmwf_bufr_active_storms():
    """获取 ECMWF 当前追踪的活跃热带气旋"""
    storms = get_ecmwf_active_storms()
    return jsonify({
        'success': True,
        'count': len(storms),
        'storms': storms,
        'source': 'ECMWF Open Data BUFR',
    })


def _check_import(module_path):
    """检查Python模块是否可导入"""
    try:
        parts = module_path.split('.')
        mod = __import__(parts[0])
        for part in parts[1:]:
            mod = getattr(mod, part)
        return True
    except (ImportError, AttributeError):
        return False


@app.route('/api/landfall/<tfbh>')
def predict_landfall(tfbh):
    """独立登陆点预测端点，返回详细的登陆信息"""
    hours = request.args.get('hours', 120, type=int)
    method = request.args.get('method', 'ensemble')

    # 查找台风数据
    year = int(tfbh[:4])
    target = None
    for month in range(1, 13):
        year_month = f"{year}{str(month).zfill(2)}"
        all_data = fetch_isc_typhoon_data(year_month)
        for t in all_data:
            if t.get('tfbh') == tfbh or t.get('ident') == tfbh:
                target = t
                break
        if target:
            break

    if not target:
        nii_data = fetch_nii_typhoon_geojson(tfbh)
        if nii_data:
            normalized = normalize_nii_data(nii_data, tfbh)
        else:
            return jsonify({'error': f'台风 {tfbh} 数据未找到'}), 404
    else:
        normalized = normalize_isc_data([target])[0]

    result = TyphoonPredictor.predict_path(normalized, hours=hours, method=method)
    landfalls = result.get('landfalls', {})

    return jsonify({
        'typhoon_id': tfbh,
        'name': normalized.get('name_cn', '') or normalized.get('name_en', ''),
        'hours': hours,
        'method': method,
        'base_time': result.get('base_time', ''),
        'base_position': f"{result.get('base_lat', 0)}°N, {result.get('base_lng', 0)}°E",
        'landfalls': landfalls,
        'summary': _format_landfall_summary(landfalls, normalized),
        'prediction_count': sum(1 for k in result.get('predictions', {})
                                if not k.startswith('forecast_')),
    })


def _format_landfall_summary(landfalls, typhoon_data):
    """格式化登陆摘要文本"""
    if not landfalls:
        return {
            'text': '预测路径未显示登陆（台风可能在海上减弱消散或转向远离陆地）',
            'will_landfall': False
        }

    # 优先使用 ensemble 方法的结果
    lf = landfalls.get('ensemble') or next(iter(landfalls.values()))
    method_name = 'ensemble' if 'ensemble' in landfalls else list(landfalls.keys())[0]
    method_label = {
        'ensemble': 'Kalman融合', 'trend': '趋势外推', 'physics': '物理模型',
        'gfs': 'GFS数值预报', 'lstm': 'LSTM深度学习', 'ecmwf': 'ECMWF IFS'
    }.get(method_name, method_name)

    name = typhoon_data.get('name_cn', '') or typhoon_data.get('name_en', '') or '该台风'

    return {
        'will_landfall': True,
        'text': f'{name}预计将于{lf["time"][:16] if lf.get("time") else "待定"}'
                f'在{lf["coast_name"]}沿海登陆'
                f'（{lf["lat"]}°N, {lf["lng"]}°E）'
                f'，登陆时中心气压约{lf.get("pressure", "?")}hPa，'
                f'最大风速约{lf.get("wind_speed", "?")}m/s。'
                f'距当前约{lf["hours_from_base"]}小时。'
                f'（预测方法: {method_label}）',
        'landfall_point': {
            'lat': lf['lat'],
            'lng': lf['lng'],
            'time': lf.get('time', ''),
            'coast_name': lf['coast_name'],
            'pressure': lf.get('pressure', 0),
            'wind_speed': lf.get('wind_speed', 0),
            'hours_from_base': lf.get('hours_from_base', 0),
        },
        'all_methods': {k: {
            'lat': v['lat'], 'lng': v['lng'],
            'coast_name': v['coast_name'],
            'hours_from_base': v.get('hours_from_base', 0),
        } for k, v in landfalls.items()},
    }


# ============================================================
# 启动
# ============================================================

ensure_dirs()

if __name__ == '__main__':
    # 启动后台自动调度引擎
    from scheduler import setup_flask_scheduler
    scheduler = setup_flask_scheduler(app)

    print("=" * 60)
    print("  台风路径预测系统 - 后端服务")
    print("  数据源: ISC (istrongcloud) + NII (日本信息研究所)")
    print("  自动调度: 数据获取1h/预测30min/训练3:00AM")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8088, debug=False)
