"""
台风路径预测系统 - 后端服务
数据源: ISC (istrongcloud.com) + NII (日本信息研究所)
"""

import json
import math
import os
import re
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests as req_lib

# LSTM深度学习预测模块
from lstm_predictor import lstm_predict, is_lstm_ready, LSTMTrainer, WINDOW_SIZE
from pangu_predictor import pangu_predict, is_pangu_ready

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

        # 1. 趋势外推
        if method in ['trend', 'ensemble', 'all']:
            predictions['trend'] = TyphoonPredictor._trend_extrapolation(points, hours)

        # 2. 物理模型
        if method in ['physics', 'ensemble', 'all']:
            predictions['physics'] = TyphoonPredictor._physics_model(points, hours)

        # 3. GFS涡旋追踪（MSLP气压场最低中心）- ensemble只调用GFS+ECMWF减少API压力
        if method in ['gfs', 'ensemble', 'all']:
            gfs_pred = TyphoonPredictor._nwp_vortex_track(last_point, hours, model='gfs', model_label='GFS')
            if gfs_pred:
                predictions['gfs'] = gfs_pred

        # 3b. ECMWF IFS涡旋追踪（欧洲中心确定性预报，全球最准NWP之一）
        if method in ['ecmwf', 'ensemble', 'all']:
            ecmwf_pred = TyphoonPredictor._nwp_vortex_track(last_point, hours, model='ecmwf_ifs04', model_label='ECMWF IFS')
            if ecmwf_pred:
                predictions['ecmwf'] = ecmwf_pred

        # 4. GraphCast AI预报（仅单独调用时使用）
        if method in ['gfs_graphcast', 'all']:
            gc_pred = TyphoonPredictor._gfs_graphcast_prediction(last_point, hours)
            if gc_pred:
                predictions['gfs_graphcast'] = gc_pred

        # 4b. ECMWF AIFS AI预报系统（仅单独调用时使用）
        if method in ['aifs', 'all']:
            aifs_pred = TyphoonPredictor._nwp_vortex_track(last_point, hours, model='aifs', model_label='AIFS')
            if aifs_pred:
                predictions['aifs'] = aifs_pred

        # 4c. CMA GRAPES涡旋追踪（仅单独调用时使用）
        if method in ['cma', 'all']:
            cma_pred = TyphoonPredictor._nwp_vortex_track(last_point, hours, model='cma_grapes_global', model_label='CMA GRAPES')
            if cma_pred:
                predictions['cma'] = cma_pred

        # 5. 历史相似路径类比法（仅单独调用或all时使用，ensemble跳过以避免慢速历史搜索）
        if method in ['analog', 'all']:
            analog_pred = TyphoonPredictor._analog_prediction(typhoon_data, hours)
            if analog_pred:
                predictions['analog'] = analog_pred

        # 6. LSTM深度学习预测
        if method in ['lstm', 'ensemble', 'all']:
            lstm_pred = TyphoonPredictor._lstm_prediction(points, hours)
            if lstm_pred:
                predictions['lstm'] = lstm_pred

        # 7. Pangu-Weather盘古大模型（可选，需下载ONNX权重+ERA5数据）
        if method in ['pangu', 'all']:
            pangu_pred = TyphoonPredictor._pangu_prediction(typhoon_data, hours)
            if pangu_pred:
                predictions['pangu'] = pangu_pred

        # ★ 关键：在ensemble融合之前，先纳入机构预报
        # 否则_kalman_ensemble看不到任何forecast_*数据，机构锚定完全失效
        if typhoon_data.get('forecasts'):
            for agency, fc_data in typhoon_data['forecasts'].items():
                predictions[f'forecast_{agency}'] = fc_data['points']

        # 综合融合 (Kalman滤波加权，含机构预报锚定)
        if method == 'ensemble' and len(predictions) >= 2:
            ensemble_pred = TyphoonPredictor._kalman_ensemble(predictions, hours, last_point)
            if ensemble_pred:
                predictions['ensemble'] = ensemble_pred

        if method == 'all' and len(predictions) >= 2:
            ensemble_pred = TyphoonPredictor._kalman_ensemble(predictions, hours, last_point)
            if ensemble_pred:
                predictions['ensemble'] = ensemble_pred

        # ============================================================
        # 登陆点检测
        # ============================================================
        landfalls = {}
        for method_name, pred_points in predictions.items():
            if method_name.startswith('forecast_'):
                continue
            lf = detect_landfall_from_segments(pred_points, margin_deg=0.3)
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
            response = req_lib.get(url, timeout=20)
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
            'pangu': 0.08,
            'ecmwf': 0.06,      # ECMWF IFS涡旋追踪
            'aifs': 0.04,        # ECMWF AIFS AI
            'lstm': 0.04,        # LSTM深度学习
            'gfs': 0.02,         # GFS涡旋追踪
            'cma': 0.015,        # CMA GRAPES涡旋追踪
            'gfs_graphcast': 0.01,
            'analog': 0.01,
            'physics': 0.005,
            'trend': 0.005,
        }

        # 异常值剔除阈值：偏离机构共识超过此距离(度)的AI方法权重降为1/10
        OUTLIER_THRESHOLD_DEG = 5.0   # ~550km
        OUTLIER_PENALTY = 0.1          # 异常方法权重保留比例

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
    def _trend_extrapolation(points, hours):
        """趋势外推法 - 基于最近几个数据点的移动趋势"""
        # 取最近 N 个点计算趋势
        n = min(6, len(points))
        recent = points[-n:]

        # 计算平均移动向量
        total_lat_delta = 0
        total_lng_delta = 0
        total_time_delta = 0

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
                total_lat_delta += dlat / dt_hours
                total_lng_delta += dlng / dt_hours
                total_time_delta += dt_hours

        avg_lat_rate = total_lat_delta / (n - 1) if n > 1 else 0  # 度/小时
        avg_lng_rate = total_lng_delta / (n - 1) if n > 1 else 0

        # 计算加速度（趋势变化）
        if len(recent) >= 4:
            half = len(recent) // 2
            first_half = recent[:half]
            second_half = recent[half:]

            rate1_lat = rate1_lng = 0
            rate2_lat = rate2_lng = 0
            for i in range(1, len(first_half)):
                rate1_lat += first_half[i]['lat'] - first_half[i-1]['lat']
                rate1_lng += first_half[i]['lng'] - first_half[i-1]['lng']
            for i in range(1, len(second_half)):
                rate2_lat += second_half[i]['lat'] - second_half[i-1]['lat']
                rate2_lng += second_half[i]['lng'] - second_half[i-1]['lng']

            accel_lat = (rate2_lat - rate1_lat) / max(len(first_half)-1, 1)
            accel_lng = (rate2_lng - rate1_lng) / max(len(first_half)-1, 1)
        else:
            accel_lat = 0
            accel_lng = 0

        # 生成预测点
        last_point = points[-1]
        predictions = []
        dt = 6  # 每6小时一个预测点

        for h in range(dt, hours + 1, dt):
            # 置信度衰减因子：仅影响confidence，不影响位置
            # 台风不会因为预测久了就减速停下！
            decay = math.exp(-h / (hours * 1.5))

            # 线性外推：速度 * 时间，不衰减
            pred_lat = last_point['lat'] + avg_lat_rate * h + accel_lat * h * 0.5
            pred_lng = last_point['lng'] + avg_lng_rate * h + accel_lng * h * 0.5

            # 压力和风速趋势预测
            recent_pressures = [p.get('pressure', 0) for p in recent if p.get('pressure')]
            recent_winds = [p.get('wind_speed', 0) for p in recent if p.get('wind_speed')]

            pred_pressure = last_point.get('pressure', 0)
            pred_wind = last_point.get('wind_speed', 0)
            if len(recent_pressures) >= 2:
                p_rate = (recent_pressures[-1] - recent_pressures[0]) / max(len(recent_pressures)-1, 1)
                pred_pressure = last_point.get('pressure', 0) + p_rate * h
            if len(recent_winds) >= 2:
                w_rate = (recent_winds[-1] - recent_winds[0]) / max(len(recent_winds)-1, 1)
                pred_wind = last_point.get('wind_speed', 0) + w_rate * h

            try:
                base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
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
                'confidence': round(0.85 * decay, 2),
            })

        return predictions

    @staticmethod
    def _physics_model(points, hours):
        """物理模型预测 - Beta漂移 + 引导气流 + 简化科里奥利力"""
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

        # 计算当前移动向量
        n = min(4, len(points))
        recent = points[-n:]
        move_lat = move_lng = 0
        for i in range(1, len(recent)):
            move_lat += recent[i]['lat'] - recent[i-1]['lat']
            move_lng += recent[i]['lng'] - recent[i-1]['lng']
        move_lat /= max(n-1, 1)
        move_lng /= max(n-1, 1)

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

        for h in range(dt, hours + 1, dt):
            # 置信度衰减，不影响位置计算
            decay = math.exp(-h / (hours * 2))

            # 预测位置: 引导气流 + beta漂移（线性外推，不衰减）
            pred_lat = current_lat + (move_lat * 2 + beta_n) * dt
            pred_lng = current_lng + (move_lng * 2 - beta_w) * dt

            # 当台风接近陆地或高纬度时，偏向东北转向
            if pred_lat > 25:
                recurvature_factor = min((pred_lat - 25) / 15, 1.0) * 0.5
                pred_lng += recurvature_factor * dt * 0.3  # 向东偏转

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

        # 预测时间越远，置信度越低
        time_decay = math.exp(-hours / 120)

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
    })


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
    """从缓存中读取自动计算的预测结果（前端快速查询）"""
    from scheduler import get_cached_prediction
    result = get_cached_prediction(tfid, hours)
    if result:
        return jsonify(result)
    return jsonify({'cached': False, 'message': '无缓存预测，请手动触发预测'})


@app.route('/api/scheduler/status')
def scheduler_status():
    """查询调度引擎状态"""
    from scheduler import PREDICTION_CACHE_DIR, HASH_DIR
    from lstm_predictor import is_lstm_ready

    # 统计缓存预测文件
    pred_files = [f for f in os.listdir(PREDICTION_CACHE_DIR) if f.endswith('.json')] if os.path.exists(PREDICTION_CACHE_DIR) else []

    # 统计数据文件
    data_files = [f for f in os.listdir(ISC_DIR) if f.endswith('.json')] if os.path.exists(ISC_DIR) else []

    # 统计哈希文件(有哈希 = 已检测过变化)
    hash_files = [f for f in os.listdir(HASH_DIR) if f.endswith('.hash')] if os.path.exists(HASH_DIR) else []

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

    return jsonify({
        'scheduler_running': True,
        'data_files_count': len(data_files),
        'hash_files_count': len(hash_files),
        'cached_predictions_count': len(pred_files),
        'cached_prediction_ids': pred_files[:10],
        'latest_prediction_time': latest_pred_time,
        'lstm_model_ready': is_lstm_ready(),
        'auto_fetch_interval': '1小时',
        'auto_prediction_interval': '30分钟',
        'auto_training_schedule': '每天3:00AM',
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
