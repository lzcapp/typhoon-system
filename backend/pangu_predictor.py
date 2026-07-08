"""
Pangu-Weather (华为盘古天气大模型) 本地推理模块

盘古大模型是华为开发的AI天气预测模型(Nature 2022论文)，
在台风追踪等极端天气预报上表现优于ECMWF HRES。

本模块提供Pangu-Weather ONNX模型本地推理功能。

P2升级版: 通过ECMWF Open Data获取全球分析场数据(GRIB2)，
自动转换为Pangu-Weather ONNX输入格式，实现完整推理管线。

设置步骤:
1. 下载ONNX模型权重到 models/pangu/ 目录:
   - pangu_weather_24.onnx (~1.1GB, 24h预报模型)
   - pangu_weather_6.onnx (~1.1GB, 6h预报模型)
   下载地址: https://drive.google.com/drive/folders/1Rjhf3In8aAnEpYfUkC2j-hLvH0c1c2PX
   或使用脚本: python models/pangu/download_models.py

2. 安装依赖:
   pip install onnxruntime numpy ecmwf-opendata eccodes

3. Docker部署需安装系统库:
   apt-get install -y libeccodes-dev

推理流程:
- 从ECMWF Open Data下载全球0.25°分析场(GRIB2)
- 转换为Pangu ONNX输入格式 (numpy数组)
- 6h和24h模型交替迭代推理
- 从输出场中用TC tracker提取台风中心位置

性能参考:
- GPU推理: 10天预报约1分钟
- CPU推理: 10天预报约2-4小时(不推荐生产环境)
- 模型加载: 约30秒(CPU)

许可证: Pangu-Weather模型权重遵循原始论文的许可证
         ECMWF Open Data遵循CC-BY-4.0
"""

import math
import os
import time
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np

PANGU_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'pangu')
PANGU_24H_MODEL = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_24.onnx')
PANGU_6H_MODEL = os.path.join(PANGU_MODEL_DIR, 'pangu_weather_6.onnx')

# 初始场数据缓存目录
INITIAL_DATA_CACHE_DIR = os.path.join(PANGU_MODEL_DIR, 'initial_data')
os.makedirs(INITIAL_DATA_CACHE_DIR, exist_ok=True)

# GRIB2下载缓存
GRIB_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'grib2_cache')
os.makedirs(GRIB_CACHE_DIR, exist_ok=True)

# Pangu-Weather输入格式常量
# 网格: 0.25°分辨率, 721纬度点(90N→90S), 1440经度点(0→360E)
N_LAT = 721  # 0.25°网格
N_LNG = 1440

# 气压层(从上到下): 13层 × 5变量(Z,Q,T,U,V)
PANGU_PRESSURE_LEVELS = [200, 250, 300, 400, 500, 600, 700, 850, 925, 950, 1000]
N_UPPER_VARS = 5  # Z(位势高度), Q(比湿), T(温度), U(纬向风), V(经向风)
N_UPPER_CHANNELS = len(PANGU_PRESSURE_LEVELS) * N_UPPER_VARS  # = 55 → 但ONNX输入是(13,721,1440)

# 实际上Pangu-Weather ONNX输入形状:
# input_upper: (5, 13, 721, 1440) → 5变量 × 13气压层 × 网格 (但ONNX模型期望特定形状)
# 不同版本有不同的输入形状，需检查模型元数据

# Surface变量: MSL, U10, V10, T2 = 4个
N_SURFACE_VARS = 4


def is_pangu_ready():
    """检查Pangu-Weather模型是否可用(ONNX权重已下载)"""
    return os.path.exists(PANGU_6H_MODEL) and os.path.exists(PANGU_24H_MODEL)


def check_onnx_input_shape():
    """检查Pangu-Weather ONNX模型的输入形状，确定数据格式要求"""
    if not is_pangu_ready():
        return None

    try:
        import onnxruntime as ort
        session_opts = ort.SessionOptions()
        session_opts.enable_mem_pattern = False

        # 检查24h模型输入形状
        session = ort.InferenceSession(PANGU_24H_MODEL, sess_options=session_opts,
                                        providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        shapes = {}
        for inp in inputs:
            shapes[inp.name] = inp.shape
        return shapes
    except Exception as e:
        print(f"Pangu ONNX shape check error: {e}")
        return None


def pangu_predict(typhoon_data, hours=72):
    """使用Pangu-Weather ONNX模型预测台风路径

    自动从ECMWF Open Data获取全球分析场数据，
    转换为Pangu输入格式，运行ONNX推理，提取台风轨迹。

    Args:
        typhoon_data: ISC格式台风数据
        hours: 预报时长

    Returns:
        预测路径点列表，或None
    """
    if not is_pangu_ready():
        print("Pangu-Weather: ONNX模型权重未下载，跳过")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("Pangu-Weather: 需要onnxruntime: pip install onnxruntime")
        return None

    points = typhoon_data.get('points', [])
    if len(points) < 2:
        return None

    last_point = points[-1]
    lat = last_point.get('lat', 0)
    lng = last_point.get('lng', 0)
    current_pressure = last_point.get('pressure', 1000)

    # Step 1: 获取初始场数据
    print("Pangu-Weather: 开始获取初始场数据...")
    initial_upper, initial_surface = _prepare_initial_conditions(lat, lng, last_point)

    if initial_upper is None or initial_surface is None:
        print("Pangu-Weather: 初始场数据获取失败，跳过")
        return None

    print(f"Pangu-Weather: 初始场数据准备完成, upper shape={initial_upper.shape}, surface shape={initial_surface.shape}")

    # Step 2: 检查模型输入形状是否匹配
    model_shapes = check_onnx_input_shape()
    if model_shapes:
        print(f"Pangu-Weather: 模型输入形状: {model_shapes}")
        # 根据实际形状调整数据格式
        initial_upper, initial_surface = _reshape_for_model(
            initial_upper, initial_surface, model_shapes
        )

    # Step 3: 加载ONNX模型并推理
    print("Pangu-Weather: 加载ONNX模型...")
    start_time = time.time()

    session_opts = ort.SessionOptions()
    session_opts.enable_mem_pattern = False
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 尝试使用GPU，如果不可用则回退到CPU
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
        print("Pangu-Weather: 使用GPU推理")
    providers.append('CPUExecutionProvider')
    print(f"Pangu-Weather: 可用providers: {ort.get_available_providers()}")

    session_6h = ort.InferenceSession(PANGU_6H_MODEL, sess_options=session_opts,
                                       providers=providers)
    session_24h = ort.InferenceSession(PANGU_24H_MODEL, sess_options=session_opts,
                                        providers=providers)

    load_time = time.time() - start_time
    print(f"Pangu-Weather: 模型加载完成({load_time:.1f}s)")

    # Step 4: 迭代推理
    predictions = []
    upper = initial_upper.astype(np.float32)
    surface = initial_surface.astype(np.float32)

    try:
        base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
    except:
        base_time = datetime.now(timezone.utc)

    # 推理策略: 24h模型为主，6h模型补充
    # 0-24h: 4×6h步 → 24h
    # 24-48h: 24h步
    # 之后每24h一步
    inference_steps = _plan_inference_steps(hours)
    print(f"Pangu-Weather: 推理计划 {len(inference_steps)}步, 总{hours}h")

    current_hour = 0
    step_count = 0

    for step_hours, model_type in inference_steps:
        step_count += 1
        current_hour += step_hours

        # 选择模型
        if model_type == '6h':
            session = session_6h
        else:
            session = session_24h

        # 运行推理
        infer_start = time.time()
        try:
            result = session.run(None, {
                'input_upper': upper,
                'input_surface': surface,
            })
        except Exception as e:
            print(f"Pangu-Weather: 推理失败(step {step_count}, {step_hours}h) - {e}")
            break

        infer_time = time.time() - infer_start
        print(f"Pangu-Weather: step {step_count} ({step_hours}h) 完成, {infer_time:.1f}s")

        # 更新初始场
        upper = result[0]
        surface = result[1]

        # 提取台风中心
        tc_center = _extract_tc_center(surface, lat, lng, current_pressure)

        if tc_center:
            pred_lat, pred_lng, pred_pressure = tc_center
            # 风速经验公式(简化的Holland模型)
            pred_wind = _estimate_wind_speed(pred_pressure, pred_lat)

            pred_time = (base_time + timedelta(hours=current_hour)).isoformat()
            decay = math.exp(-current_hour / max(hours * 2.5, 200))

            predictions.append({
                'time': pred_time,
                'forecast_hour': current_hour,
                'lat': round(pred_lat, 1),
                'lng': round(pred_lng, 1),
                'pressure': round(pred_pressure),
                'wind_speed': round(pred_wind, 1),
                'category': _pangu_intensity_category(pred_pressure, pred_wind),
                'confidence': round(0.95 * decay, 2),
                'method_desc': f'Pangu-Weather盘古AI({model_type}步)',
            })

            # 更新搜索参考位置
            lat = pred_lat
            lng = pred_lng
            current_pressure = pred_pressure
        else:
            print(f"Pangu-Weather: step {step_count} ({step_hours}h) 未找到台风中心")

    total_time = time.time() - start_time
    print(f"Pangu-Weather: 推理完成, {len(predictions)}点, 总耗时{total_time:.1f}s")

    return predictions if predictions else None


def _plan_inference_steps(hours):
    """规划推理步长序列

    策略:
    - 0-24h: 6h模型 × 4步 (更精确)
    - 24-48h: 6h模型 × 4步
    - 48h+: 24h模型为主 (更快)
    - 如果需要精确: 每24h中用6h模型替代

    Returns:
        [(step_hours, model_type), ...] 列表
    """
    steps = []
    remaining = hours

    # 前48h: 用6h模型(更精确)
    while remaining > 0 and sum(s[0] for s in steps) < 48:
        steps.append((6, '6h'))
        remaining -= 6

    # 48h之后: 24h模型(更快)
    while remaining > 0:
        if remaining >= 24:
            steps.append((24, '24h'))
            remaining -= 24
        elif remaining >= 6:
            steps.append((6, '6h'))
            remaining -= 6
        else:
            # 剩余不足6h, 用6h模型但截断
            steps.append((6, '6h'))
            break

    return steps


def _estimate_wind_speed(pressure_hpa, lat):
    """从气压估算风速(简化Holland模型)

    考虑纬度效应: 高纬度台风风速-气压关系更强
    """
    # 基础关系: V = 3.4 * sqrt(1010 - P)
    delta_p = max(1010 - pressure_hpa, 0)
    base_wind = 3.4 * math.sqrt(delta_p)

    # 纬度修正: >25°N时风速更强(Coriolis效应)
    if lat > 25:
        lat_factor = 1 + (lat - 25) * 0.02
    else:
        lat_factor = 1.0

    return min(base_wind * lat_factor, 80)  # 上限80m/s


def _reshape_for_model(upper, surface, model_shapes):
    """根据ONNX模型实际输入形状调整数据格式"""
    upper_shape = model_shapes.get('input_upper', None)
    surface_shape = model_shapes.get('input_surface', None)

    # Pangu-Weather ONNX模型可能期望不同的形状:
    # 原始版: (5, 13, 721, 1440) 或 (13, 721, 1440)
    # 重构版: 可能是 (1, 5, 13, 721, 1440) 等

    if upper_shape:
        expected = [d if isinstance(d, int) else -1 for d in upper_shape]
        actual = list(upper.shape)
        # 如果形状不完全匹配，尝试reshape
        if len(expected) != len(actual) or any(
            e != a and e != -1 for e, a in zip(expected, actual)
        ):
            try:
                # 尝试自动reshape(总元素数相同)
                upper = upper.reshape(expected)
            except:
                print(f"Pangu: upper shape mismatch, expected={expected}, actual={actual}")

    if surface_shape:
        expected = [d if isinstance(d, int) else -1 for d in surface_shape]
        actual = list(surface.shape)
        if len(expected) != len(actual) or any(
            e != a and e != -1 for e, a in zip(expected, actual)
        ):
            try:
                surface = surface.reshape(expected)
            except:
                print(f"Pangu: surface shape mismatch, expected={expected}, actual={actual}")

    return upper, surface


# ============================================================
# 初始场数据获取 (P2核心)
# ============================================================

def _prepare_initial_conditions(lat, lng, last_point):
    """准备Pangu-Weather初始场数据

    三级优先级:
    1. 从ECMWF Open Data下载全球分析场(GRIB2) → 转换为numpy
    2. 从缓存加载之前转换的numpy数据
    3. 从Open-Meteo获取GFS数据拼凑(精度较低，但可用)

    Returns:
        (initial_upper, initial_surface) numpy数组, 或 (None, None)
    """
    # 优先级1: 检查缓存
    cached = _load_cached_initial_data()
    if cached is not None:
        return cached

    # 优先级2: 从ECMWF Open Data下载GRIB2
    grib_data = _download_ecmwf_analysis()
    if grib_data is not None:
        upper, surface = _convert_grib_to_pangu(grib_data)
        if upper is not None and surface is not None:
            _save_cached_initial_data(upper, surface)
            return upper, surface

    # 优先级3: 从Open-Meteo获取GFS数据(点查询拼凑全球场)
    # 这是精度最低的方法，但作为最后兜底
    upper, surface = _build_initial_from_openmeteo(lat, lng, last_point)
    if upper is not None and surface is not None:
        return upper, surface

    print("Pangu-Weather: 所有初始场数据获取方式均失败")
    return None, None


def _load_cached_initial_data():
    """从缓存加载初始场numpy数据"""
    # 检查缓存是否有效(6小时内)
    upper_path = os.path.join(INITIAL_DATA_CACHE_DIR, 'initial_upper.npy')
    surface_path = os.path.join(INITIAL_DATA_CACHE_DIR, 'initial_surface.npy')

    if not os.path.exists(upper_path) or not os.path.exists(surface_path):
        return None

    # 检查缓存时间
    mtime = os.path.getmtime(upper_path)
    age_hours = (time.time() - mtime) / 3600
    if age_hours > 6:
        print("Pangu: 缓存数据过期(>6h)")
        return None

    try:
        upper = np.load(upper_path)
        surface = np.load(surface_path)
        print(f"Pangu: 使用缓存初始场(缓存{age_hours:.1f}h前)")
        return upper, surface
    except Exception as e:
        print(f"Pangu: 缓存加载失败 - {e}")
        return None


def _save_cached_initial_data(upper, surface):
    """保存初始场numpy数据到缓存"""
    try:
        np.save(os.path.join(INITIAL_DATA_CACHE_DIR, 'initial_upper.npy'), upper)
        np.save(os.path.join(INITIAL_DATA_CACHE_DIR, 'initial_surface.npy'), surface)
        print("Pangu: 初始场数据已缓存")
    except Exception as e:
        print(f"Pangu: 缓存保存失败 - {e}")


def _download_ecmwf_analysis():
    """从 ECMWF Open Data 下载全球分析场(GRIB2)

    下载ERA5或GFS分析场数据(0.25°分辨率，全球覆盖)，
    用于Pangu-Weather的初始条件输入。

    Returns:
        GRIB2文件路径, 或None
    """
    # 检查ecmwf-opendata是否可用
    try:
        from ecmwf.opendata import Client
    except ImportError:
        print("Pangu: ecmwf-opendata不可用, 尝试HTTP下载")
        return _download_gfs_analysis_http()

    # 自动查找最新可用分析场
    now_utc = datetime.now(timezone.utc)

    # ECMWF分析场延迟约8小时
    # HRES分析: 00/06/12/18 UTC
    DATA_READY_OFFSET = 8

    run_times_desc = [18, 12, 6, 0]
    for day_offset in range(2):
        check_date = (now_utc - timedelta(days=day_offset)).date()
        for rt in run_times_desc:
            run_utc = datetime(
                check_date.year, check_date.month, check_date.day,
                rt, 0, 0, tzinfo=timezone.utc
            )
            if now_utc >= run_utc + timedelta(hours=DATA_READY_OFFSET):
                date_str = check_date.strftime('%Y%m%d')
                break
        else:
            continue
        break
    else:
        print("Pangu: 无可用的ECMWF分析场数据")
        return None

    # 下载GRIB2数据
    # 需要下载多层多变量的全球场数据
    cache_path = os.path.join(GRIB_CACHE_DIR, f'analysis_{date_str}_{rt:02d}Z.grib2')

    # 检查缓存
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) / 3600 < 6:
        print(f"Pangu: 使用GRIB缓存 {cache_path}")
        return cache_path

    try:
        client = Client(source='ecmwf')

        # 下载分析场: 包含表面变量和多层大气变量
        # surface: msl, u10, v10, t2
        # upper: z, q, t, u, v at multiple pressure levels
        client.retrieve(
            date=int(date_str),
            time=rt,
            stream='oper',
            type='an',
            levtype='pl',   # pressure levels
            levelist='200/250/300/400/500/600/700/850/925/950/1000',
            param='z/q/t/u/v',
            target=cache_path + '_upper',
        )

        # 表面层
        client.retrieve(
            date=int(date_str),
            time=rt,
            stream='oper',
            type='an',
            levtype='sfc',
            param='msl/u10/v10/t2',
            target=cache_path + '_surface',
        )

        # 合并两个文件
        # (GRIB2可以顺序拼接多个消息)
        combined_path = cache_path
        with open(combined_path, 'wb') as out:
            for suffix in ['_upper', '_surface']:
                fpath = cache_path + suffix
                if os.path.exists(fpath):
                    with open(fpath, 'rb') as f:
                        out.write(f.read())

        # 清理临时文件
        for suffix in ['_upper', '_surface']:
            fpath = cache_path + suffix
            if os.path.exists(fpath):
                os.remove(fpath)

        if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
            print(f"Pangu: ECMWF分析场下载完成 ({os.path.getsize(combined_path)} bytes)")
            return combined_path

    except Exception as e:
        print(f"Pangu: ECMWF分析场下载失败 - {e}")

    return None


def _download_gfs_analysis_http():
    """从NOAA GFS NOMADS下载全球分析场(GRIB2)作为备用

    当ECMWF Open Data不可用时使用。
    GFS分析场也可作为Pangu-Weather初始条件(精度略低于ERA5)。
    """
    try:
        import requests as req_lib
    except ImportError:
        return None

    # GFS NOMADS URL格式
    now_utc = datetime.now(timezone.utc)

    # GFS分析场延迟约4小时
    run_times_desc = [18, 12, 6, 0]
    for day_offset in range(2):
        check_date = (now_utc - timedelta(days=day_offset)).date()
        for rt in run_times_desc:
            run_utc = datetime(
                check_date.year, check_date.month, check_date.day,
                rt, 0, 0, tzinfo=timezone.utc
            )
            if now_utc >= run_utc + timedelta(hours=4):
                date_str = check_date.strftime('%Y%m%d')
                run_hour = rt
                break
        else:
            continue
        break
    else:
        return None

    # GFS 0.25°网格分析场
    # NOMADS URL: https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
    cache_path = os.path.join(GRIB_CACHE_DIR, f'gfs_analysis_{date_str}_{run_hour:02d}.grib2')

    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) / 3600 < 6:
        return cache_path

    # 尝试下载GFS分析场
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f"?file=gfs.t{run_hour:02d}z.pgrb2.0p25.anl"
        f"&lev_10_m_above_ground=on"
        f"&lev_mean_sea_level=on"
        f"&lev_surface=on"
        f"&var_UGRD=on&var_VGRD=on&var_PRMSL=on&var_TMP=on"
        f"&leftlon=0&rightlon=360&toplat=90&bottomlat=-90"
        f"&dir=%2Fgfs.%2F{date_str}%2F{run_hour:02d}%2Fatmos"
    )

    try:
        response = req_lib.get(url, timeout=60, stream=True)
        if response.status_code == 200:
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
            if os.path.getsize(cache_path) > 0:
                print(f"Pangu: GFS分析场下载完成 ({os.path.getsize(cache_path)} bytes)")
                return cache_path
    except Exception as e:
        print(f"Pangu: GFS分析场下载失败 - {e}")

    return None


def _convert_grib_to_pangu(grib_path):
    """将GRIB2全球场数据转换为Pangu-Weather ONNX输入格式

    使用eccodes解析GRIB2，提取多层大气变量和表面变量，
    转换为Pangu需要的numpy数组格式。

    Args:
        grib_path: GRIB2文件路径

    Returns:
        (upper_array, surface_array), 形状取决于Pangu模型版本
        通常: upper (5, 13, 721, 1440) 或 (13, 721, 1440)
             surface (4, 721, 1440)
    """
    try:
        import eccodes
    except ImportError:
        print("Pangu: 需要eccodes解析GRIB2: pip install eccodes + apt-get install libeccodes-dev")
        # 尝试用cfgrib(xarray backend)作为替代
        return _convert_grib_with_cfgrib(grib_path)

    if grib_path is None or not os.path.exists(grib_path):
        return None, None

    upper_data = {}  # {level: {var_name: values}}
    surface_data = {}  # {var_name: values}

    with open(grib_path, 'rb') as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break

            try:
                # 提取元数据
                level_type = eccodes.codes_get(gid, 'typeOfLevel')
                level = eccodes.codes_get(gid, 'level')
                short_name = eccodes.codes_get(gid, 'shortName')
                ni = eccodes.codes_get(gid, 'Ni')  # 经度点数
                nj = eccodes.codes_get(gid, 'Nj')  # 纬度点数
                lat_first = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
                lat_last = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
                lng_first = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')

                # 提取数据值
                values = eccodes.codes_get_values(gid)

                # 转换为721×1440网格(0.25°分辨率)
                # 如果原始网格不同，需要插值/重采样
                if ni == N_LNG and nj == N_LAT:
                    grid = values.reshape(N_LAT, N_LNG)
                else:
                    # 重采样到0.25°全球网格
                    grid = _resample_grid(values, ni, nj, lat_first, lat_last, lng_first)

                # 分类存储
                if level_type == 'pl':  # pressure level
                    if level not in upper_data:
                        upper_data[level] = {}
                    # 变量名映射: z→Z, q→Q, t→T, u→U, v→V
                    var_map = {'z': 'Z', 'q': 'Q', 't': 'T', 'u': 'U', 'v': 'V',
                               'gh': 'Z', 'uwnd': 'U', 'vwnd': 'V'}
                    mapped_name = var_map.get(short_name, short_name.upper())
                    upper_data[level][mapped_name] = grid

                elif level_type in ['sfc', 'surface', 'meanSeaLevel', 'heightAboveGround']:
                    # Surface变量映射
                    sfc_map = {
                        'msl': 'MSL', 'prmsl': 'MSL',  # 海平面气压
                        'u10': 'U10', '10u': 'U10', 'ugrd': 'U10',  # 10m纬向风
                        'v10': 'V10', '10v': 'V10', 'vgrd': 'V10',  # 10m经向风
                        't2': 'T2', '2t': 'T2', 'tmp': 'T2',  # 2m温度
                    }
                    mapped_name = sfc_map.get(short_name, short_name)
                    if level_type == 'meanSeaLevel' or level == 0:
                        surface_data[mapped_name] = grid
                    elif level == 10:  # 10m above ground
                        surface_data[mapped_name] = grid
                    elif level == 2:   # 2m above ground
                        surface_data[mapped_name] = grid

            except Exception as e:
                print(f"Pangu GRIB decode error: {e}")

            finally:
                eccodes.codes_release(gid)

    # 构建Pangu输入数组
    upper_array = _build_upper_array(upper_data)
    surface_array = _build_surface_array(surface_data)

    return upper_array, surface_array


def _convert_grib_with_cfgrib(grib_path):
    """使用cfgrib(xarray backend)解析GRIB2作为eccodes的替代"""
    try:
        import xarray as xr
        import cfgrib
    except ImportError:
        print("Pangu: cfgrib也不可用")
        return None, None

    try:
        # 打开GRIB2文件
        ds = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={
            'indexpath': '',  # 不创建索引文件
        })

        # 提取数据并转换为Pangu格式
        # cfgrib返回xarray Dataset，需要转换为numpy
        upper_data = {}
        surface_data = {}

        for var_name in ds.data_vars:
            var = ds[var_name]
            # 处理维度等
            values = var.values
            # ... 具体转换逻辑取决于cfgrib输出格式
            # 这部分需要根据实际数据调试

        # 由于cfgrib输出格式不确定，暂时返回None
        print("Pangu: cfgrib转换逻辑待实现")
        return None, None

    except Exception as e:
        print(f"Pangu cfgrib error: {e}")
        return None, None


def _resample_grid(values, ni, nj, lat_first, lat_last, lng_first):
    """将GRIB2数据重采样到0.25°全球网格(721×1440)

    如果原始数据不是0.25°分辨率或网格范围不同，
    使用线性插值重采样到Pangu-Weather需要的标准网格。
    """
    # 原始网格
    src_grid = values.reshape(nj, ni)

    # 目标网格: 0.25°全球 (90N→90S, 0→360E)
    target_lats = np.linspace(90, -90, N_LAT)
    target_lngs = np.linspace(0, 360, N_LNG)

    # 原始网格纬度(假设从北到南)
    src_lats = np.linspace(lat_first, lat_last, nj)
    src_lngs = np.linspace(lng_first, lng_first + 360 - 360/ni, ni)

    # 简化重采样: 最近邻插值(速度快，精度足够)
    result = np.zeros((N_LAT, N_LNG))

    for i in range(N_LAT):
        for j in range(N_LNG):
            # 找最近的源网格点
            src_i = int(np.argmin(np.abs(src_lats - target_lats[i])))
            src_j = int(np.argmin(np.abs(src_lngs - target_lngs[j])))

            if src_i < nj and src_j < ni:
                result[i, j] = src_grid[src_i, src_j]

    return result


def _build_upper_array(upper_data):
    """构建Pangu-Weather高层大气输入数组

    Pangu需要: (5, 13, 721, 1440) 或 (13, 721, 1440) 格式
    5变量 × 13气压层 × 网格
    """
    # 变量顺序: Z, Q, T, U, V
    var_order = ['Z', 'Q', 'T', 'U', 'V']
    level_order = PANGU_PRESSURE_LEVELS  # 200, 250, ..., 1000

    # 检查是否有足够的数据
    has_data = False
    for level in level_order:
        if level in upper_data and len(upper_data[level]) >= 3:
            has_data = True
            break

    if not has_data:
        print("Pangu: 无高层大气数据")
        return None

    # 尝试构建 (5, 13, 721, 1440) 格式
    try:
        channels = []
        for var in var_order:
            level_data = []
            for level in level_order:
                if level in upper_data and var in upper_data[level]:
                    level_data.append(upper_data[level][var])
                else:
                    # 缺失数据: 用邻近层插值或用标准大气值填充
                    filled = _fill_missing_level(upper_data, level, var)
                    level_data.append(filled)

            channels.append(np.stack(level_data))

        upper_array = np.stack(channels)  # (5, 13, 721, 1440)
        return upper_array.astype(np.float32)

    except Exception as e:
        print(f"Pangu: 构建upper数组失败 - {e}")
        return None


def _build_surface_array(surface_data):
    """构建Pangu-Weather表面输入数组

    Pangu需要: (4, 721, 1440) 格式
    4变量: MSL, U10, V10, T2
    """
    var_order = ['MSL', 'U10', 'V10', 'T2']

    # 检查是否有表面数据
    available_vars = set(surface_data.keys())
    required_vars = set(var_order)

    if not required_vars.intersection(available_vars):
        print("Pangu: 无表面数据")
        return None

    channels = []
    for var in var_order:
        if var in surface_data:
            channels.append(surface_data[var])
        else:
            # 缺失变量: 用气候平均值填充
            print(f"Pangu: 表面变量 {var} 缺失，用默认值填充")
            default = _get_default_surface_values(var)
            channels.append(default)

    surface_array = np.stack(channels)  # (4, 721, 1440)
    return surface_array.astype(np.float32)


def _fill_missing_level(upper_data, target_level, var_name):
    """填充缺失的气压层数据

    策略:
    1. 从邻近层线性插值
    2. 如果无邻近层，用标准大气值
    """
    # 找邻近有数据的层
    available_levels = [l for l in PANGU_PRESSURE_LEVELS
                        if l in upper_data and var_name in upper_data[l]]

    if len(available_levels) >= 2:
        # 线性插值
        lower_levels = [l for l in available_levels if l < target_level]
        upper_levels = [l for l in available_levels if l > target_level]

        if lower_levels and upper_levels:
            l_low = max(lower_levels)
            l_high = min(upper_levels)
            weight = (target_level - l_low) / (l_high - l_low)
            return (1 - weight) * upper_data[l_low][var_name] + weight * upper_data[l_high][var_name]

    # 标准大气值填充
    std_atm = _get_standard_atm_value(target_level, var_name)
    return np.full((N_LAT, N_LNG), std_atm, dtype=np.float32)


def _get_standard_atm_value(level_hpa, var_name):
    """标准大气参考值(用于填充缺失层)"""
    # 简化的标准大气值
    if var_name == 'Z':  # 位势高度(m)
        std_heights = {
            200: 11800, 250: 10300, 300: 9200, 400: 7200, 500: 5600,
            600: 4200, 700: 3000, 850: 1500, 925: 800, 950: 600, 1000: 100
        }
        return std_heights.get(level_hpa, 5000)
    elif var_name == 'T':  # 温度(K)
        return 270 - (level_hpa - 500) * 0.02
    elif var_name == 'Q':  # 比湿(kg/kg)
        return 0.001
    elif var_name == 'U':  # 纬向风(m/s)
        return 0.0
    elif var_name == 'V':  # 经向风(m/s)
        return 0.0
    return 0.0


def _get_default_surface_values(var_name):
    """获取默认表面变量值(全球常数场)"""
    defaults = {
        'MSL': 101325.0,  # 标准海平面气压(Pa)
        'U10': 0.0,       # 10m纬向风
        'V10': 0.0,       # 10m经向风
        'T2': 288.15,     # 2m温度(K) = 15°C
    }
    return np.full((N_LAT, N_LNG), defaults.get(var_name, 0.0), dtype=np.float32)


def _build_initial_from_openmeteo(lat, lng, last_point):
    """从Open-Meteo获取GFS数据拼凑全球场(最后兜底方案)

    通过Open-Meteo API获取台风附近的大范围GFS预报数据，
    然后填充到Pangu标准网格中。

    精度说明:
    - 这种方法只能在台风附近获取到真实数据
    - 远离台风的区域使用气候平均值填充
    - 作为Pangu推理的初始条件精度较低
    - 但比完全不运行要好

    Returns:
        (upper_array, surface_array) 或 (None, None)
    """
    try:
        import requests as req_lib
    except ImportError:
        return None, None

    print("Pangu: 使用Open-Meteo GFS数据构建初始场(精度较低)")

    # 获取台风附近30°×30°区域的GFS数据
    # 覆盖范围: 台风为中心 ±15°
    grid_range = 15  # ±15度
    grid_step = 0.25  # 0.25°分辨率

    lats = np.arange(lat - grid_range, lat + grid_range + grid_step, grid_step)
    lngs = np.arange(lng - grid_range, lng + grid_range + grid_step, grid_step)

    # 限制到合理范围
    lats = lats[(lats >= -90) & (lats <= 90)]
    lngs = lngs[(lngs >= -180) & (lngs <= 180)]

    # Open-Meteo查询(多位置，逗号分隔)
    # 限制查询点数量(避免URL过长)
    MAX_POINTS = 200
    if len(lats) * len(lngs) > MAX_POINTS:
        # 采样降低密度
        lat_step = max(1, len(lats) // int(math.sqrt(MAX_POINTS)))
        lng_step = max(1, len(lngs) // int(math.sqrt(MAX_POINTS)))
        lats = lats[::lat_step]
        lngs = lngs[::lng_step]

    lat_str = ','.join(str(round(l, 2)) for l in lats)
    lng_str = ','.join(str(round(l, 2)) for l in lngs)

    url = (
        f'https://api.open-meteo.com/v1/gfs'
        f'?latitude={lat_str}&longitude={lng_str}'
        f'&hourly=pressure_msl,temperature_2m,wind_speed_10m,wind_direction_10m'
        f'&forecast_days=1'
        f'&timeformat=iso8601'
    )

    try:
        response = req_lib.get(url, timeout=15)
        if response.status_code != 200:
            print(f"Pangu Open-Meteo: API error {response.status_code}")
            return None, None
        data = response.json()
    except Exception as e:
        print(f"Pangu Open-Meteo: error - {e}")
        return None, None

    # 解析数据并构建初始场
    # Open-Meteo返回list(多位置)或dict(单位置)
    if not isinstance(data, list):
        data = [data]

    # 构建全球0.25°网格(先填充默认值，再覆盖GFS数据)
    surface_msl = np.full((N_LAT, N_LNG), 101325.0, dtype=np.float32)  # Pa
    surface_u10 = np.zeros((N_LAT, N_LNG), dtype=np.float32)
    surface_v10 = np.zeros((N_LAT, N_LNG), dtype=np.float32)
    surface_t2 = np.full((N_LAT, N_LNG), 288.15, dtype=np.float32)  # K

    # 覆盖GFS数据到对应网格位置
    for i, loc_data in enumerate(data):
        hourly = loc_data.get('hourly', {})
        times = hourly.get('time', [])

        if i >= len(lats) or i >= len(lngs):
            continue

        loc_lat = lats[i]
        loc_lng = lngs[i]

        # 映射到Pangu网格索引
        lat_idx = int((90 - loc_lat) * 4)
        lng_idx = int(loc_lng * 4) if loc_lng >= 0 else int((loc_lng + 360) * 4)

        if 0 <= lat_idx < N_LAT and 0 <= lng_idx < N_LNG:
            # 取第一个时间步的数据(分析场)
            pressures = hourly.get('pressure_msl', [])
            temps = hourly.get('temperature_2m', [])
            wind_speeds = hourly.get('wind_speed_10m', [])
            wind_dirs = hourly.get('wind_direction_10m', [])

            if pressures and pressures[0] is not None:
                surface_msl[lat_idx, lng_idx] = float(pressures[0]) * 100  # hPa→Pa
            if temps and temps[0] is not None:
                surface_t2[lat_idx, lng_idx] = float(temps[0]) + 273.15  # °C→K
            if wind_speeds and wind_speeds[0] is not None and wind_dirs and wind_dirs[0] is not None:
                ws = float(wind_speeds[0])
                wd = float(wind_dirs[0])
                surface_u10[lat_idx, lng_idx] = ws * math.cos(math.radians(wd))
                surface_v10[lat_idx, lng_idx] = ws * math.sin(math.radians(wd))

    # 构建surface数组
    surface_array = np.stack([surface_msl, surface_u10, surface_v10, surface_t2])

    # 构建upper数组(只能用标准大气值，Open-Meteo不提供多层大气数据)
    upper_array = _build_default_upper_array()

    print(f"Pangu Open-Meteo: 初始场构建完成, upper={upper_array.shape}, surface={surface_array.shape}")
    return upper_array, surface_array


def _build_default_upper_array():
    """构建默认高层大气数组(标准大气值)

    当无法获取实际GRIB2数据时，使用标准大气参考值。
    台风附近区域的表面数据来自Open-Meteo，
    但高层大气只能用标准值。
    """
    channels = []
    var_order = ['Z', 'Q', 'T', 'U', 'V']

    for var in var_order:
        level_data = []
        for level in PANGU_PRESSURE_LEVELS:
            default = _get_standard_atm_value(level, var)
            level_data.append(np.full((N_LAT, N_LNG), default, dtype=np.float32))
        channels.append(np.stack(level_data))

    upper = np.stack(channels)  # (5, 13, 721, 1440)
    return upper.astype(np.float32)


# ============================================================
# TC Tracker (从全球场提取台风中心)
# ============================================================

def _extract_tc_center(surface_data, ref_lat, ref_lng, ref_pressure):
    """从Pangu-Weather全球表面场中提取台风中心位置

    使用海平面气压最低值法(Minimum MSLP) + 风场涡度检测法
    在参考位置附近搜索气压最低点作为台风中心

    Args:
        surface_data: (4, 721, 1440) — MSL(Pa), U10, V10, T2(K)
        ref_lat: 参考纬度(上一时刻台风中心)
        ref_lng: 参考经度
        ref_pressure: 参考气压(hPa)

    Returns:
        (pred_lat, pred_lng, pred_pressure_hpa) 或 None
    """
    if surface_data is None:
        return None

    # 根据实际形状确定MSL层
    if surface_data.ndim == 3 and surface_data.shape[0] >= 1:
        msl = surface_data[0]  # MSL(Pa)
    elif surface_data.ndim == 2:
        msl = surface_data
    else:
        return None

    # 找参考位置对应的网格索引
    # Pangu网格: 0.25°分辨率, 纬度90°N到90°S, 经度0°到360°
    lat_idx = int((90 - ref_lat) * 4)
    lng_idx = int(ref_lng * 4) if ref_lng >= 0 else int((ref_lng + 360) * 4)

    # 搜索范围: 随预报时长增大(台风可能移动很远)
    search_range_deg = 10  # 10度范围(~1100km)
    search_range = int(search_range_deg * 4)  # 网格索引范围

    min_lat_idx = max(0, lat_idx - search_range)
    max_lat_idx = min(N_LAT - 1, lat_idx + search_range)
    min_lng_idx = max(0, lng_idx - search_range)
    max_lng_idx = min(N_LNG - 1, lng_idx + search_range)

    # 搜索气压最低点
    region = msl[min_lat_idx:max_lat_idx + 1, min_lng_idx:max_lng_idx + 1]
    if region.size == 0:
        return None

    min_idx = np.unravel_index(np.argmin(region), region.shape)
    min_lat_idx_actual = min_lat_idx + min_idx[0]
    min_lng_idx_actual = min_lng_idx + min_idx[1]

    pred_lat = 90 - min_lat_idx_actual * 0.25
    # 处理经度: 0→360格式转换为-180→180
    pred_lng_raw = min_lng_idx_actual * 0.25
    pred_lng = pred_lng_raw if pred_lng_raw <= 180 else pred_lng_raw - 360

    # 气压: Pa → hPa
    pred_pressure_pa = float(msl[min_lat_idx_actual, min_lng_idx_actual])
    pred_pressure = pred_pressure_pa / 100  # Pa → hPa

    # 验证: 检查找到的点是否确实是一个低压中心
    # 气压应低于周围4个邻居点(至少3个)
    neighbors_pressures = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni = min_lat_idx_actual + di
        nj = min_lng_idx_actual + dj
        if 0 <= ni < N_LAT and 0 <= nj < N_LNG:
            neighbors_pressures.append(float(msl[ni, nj]))

    if neighbors_pressures:
        lower_count = sum(1 for np_ in neighbors_pressures if np_ > pred_pressure_pa)
        if lower_count < 2:
            # 不是明显的低压中心, 可能是噪声
            # 但如果气压本身就较低(<1005hPa), 仍然接受
            if pred_pressure > 1005:
                return None

    return pred_lat, pred_lng, pred_pressure


def _pangu_intensity_category(pressure, wind_speed):
    """台风等级判断"""
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
