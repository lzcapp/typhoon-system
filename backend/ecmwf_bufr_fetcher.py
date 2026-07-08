"""
ECMWF Open Data BUFR 台风轨迹数据获取模块

从 ECMWF 免费开放数据中下载热带气旋(TC) BUFR 轨迹预报数据，
解析为标准台风预测格式，集成到预测管线中。

ECMWF BUFR 数据包含:
- 确定性预报(HRES): stream=oper, member=52
- 集合预报(ENS): stream=enfo, member=1-50 + 控制成员51
- 风暴中心纬度/经度
- 海平面气压(Pa)
- 10m最大风速(m/s)
- 风圈半径数据

数据许可证: CC-BY-4.0 (允许商业使用)
更新频率: 每天4次 (00/06/12/18 UTC)
数据可用延迟: 约7-9小时

设置步骤:
1. 安装依赖:
   pip install ecmwf-opendata pdbufr eccodes
   (Docker中: apt-get install -y libeccodes-dev)
2. 无需API Key，直接使用
"""

import os
import tempfile
import math
import time
from datetime import datetime, timedelta, timezone

# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ecmwf_bufr')
os.makedirs(CACHE_DIR, exist_ok=True)

# 缓存有效期: 6小时（ECMWF每6h更新一次）
CACHE_EXPIRY_HOURS = 6

# 最大预报步长(小时)
MAX_STEP = {0: 240, 6: 144, 12: 240, 18: 144}


def _get_cache_key(date_str=None, run_time=0, stream='enfo'):
    """生成缓存文件名"""
    now = datetime.now(timezone.utc)
    if date_str is None:
        date_str = now.strftime('%Y%m%d')
    return f"tc_bufr_{stream}_{date_str}_{run_time:02d}Z.bin"


def _is_cache_valid(cache_path):
    """检查缓存是否有效"""
    if not os.path.exists(cache_path):
        return False
    mtime = os.path.getmtime(cache_path)
    age_hours = (time.time() - mtime) / 3600
    return age_hours < CACHE_EXPIRY_HOURS


def download_tc_bufr(date_str=None, run_time=None, stream='enfo', source='ecmwf'):
    """从 ECMWF Open Data 下载热带气旋 BUFR 数据

    Args:
        date_str: 日期字符串 YYYYMMDD，None=自动找最新可用数据
        run_time: 运行时间(0,6,12,18)，None=自动找最新可用
        stream: 'enfo'(集合预报) 或 'oper'(HRES确定性预报)
        source: 'ecmwf', 'aws', 'azure', 'google'

    Returns:
        BUFR文件路径(缓存), 或None(下载失败/无活跃台风)
    """
    # 检查缓存
    if date_str and run_time is not None:
        cache_key = _get_cache_key(date_str, run_time, stream)
        cache_path = os.path.join(CACHE_DIR, cache_key)
        if _is_cache_valid(cache_path):
            print(f"ECMWF BUFR: 使用缓存 {cache_path}")
            return cache_path

    # 自动查找最新可用数据
    if date_str is None or run_time is None:
        date_str, run_time = _find_latest_available_data()
        if date_str is None:
            print("ECMWF BUFR: 无最新可用数据")
            return None

    cache_key = _get_cache_key(date_str, run_time, stream)
    cache_path = os.path.join(CACHE_DIR, cache_key)

    # 尝试下载
    try:
        from ecmwf.opendata import Client
        client = Client(source=source)
        step = MAX_STEP.get(run_time, 240)

        client.retrieve(
            date=int(date_str),
            time=run_time,
            stream=stream,
            type='tf',
            step=step,
            target=cache_path,
        )

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            print(f"ECMWF BUFR: 下载成功 {cache_path} ({os.path.getsize(cache_path)} bytes)")
            return cache_path
        else:
            print("ECMWF BUFR: 下载文件为空(可能无活跃台风)")
            return None

    except ImportError:
        print("ECMWF BUFR: 需要安装 ecmwf-opendata: pip install ecmwf-opendata")
        return None

    except Exception as e:
        print(f"ECMWF BUFR: 下载失败 - {e}")
        # 尝试备用源(ecmwf-opendata包可能不支持时，直接HTTP下载)
        return _download_from_diss(date_str, run_time)


def _find_latest_available_data():
    """自动查找最新可用的 ECMWF BUFR 数据时间

    ECMWF数据可用延迟约8小时:
    - 00Z → ~08:00 UTC 可用
    - 06Z → ~14:00 UTC 可用
    - 12Z → ~20:00 UTC 可用
    - 18Z → ~02:00 UTC+1 可用
    """
    now_utc = datetime.now(timezone.utc)
    DATA_READY_OFFSET = 8  # 保守估计8小时后可用

    run_times_desc = [18, 12, 6, 0]

    for day_offset in range(3):  # 尝试今天、昨天、前天
        check_date = (now_utc - timedelta(days=day_offset)).date()
        for rt in run_times_desc:
            run_utc = datetime(
                check_date.year, check_date.month, check_date.day,
                rt, 0, 0, tzinfo=timezone.utc
            )
            if now_utc >= run_utc + timedelta(hours=DATA_READY_OFFSET):
                return check_date.strftime('%Y%m%d'), rt

    return None, None


def _download_from_diss(date_str, run_time):
    """从 ECMWF DISS 门户备用下载 TC BUFR 数据

    当 ecmwf-opendata 包不可用或主源失败时，
    从 essential.ecmwf.int 按风暴单独下载 BUFR 文件并合并。
    """
    import re
    try:
        import requests as req_lib
    except ImportError:
        return None

    dt_str = f"{date_str}{run_time:02d}0000"
    listing_url = f"https://essential.ecmwf.int/file/{dt_str}/"

    try:
        r = req_lib.get(listing_url, timeout=30)
        if r.status_code == 404:
            print(f"ECMWF DISS: 无数据 {dt_str} (可能无活跃台风)")
            return None
        r.raise_for_status()
    except Exception as e:
        print(f"ECMWF DISS: 请求失败 - {e}")
        return None

    # 解析文件列表，找BUFR台风轨迹文件
    tc_pattern = re.compile(r'tropical_cyclone_track.*bufr4', re.IGNORECASE)
    urls = []
    for match in re.finditer(r'href="(/file/[^"]+)"', r.text):
        path = match.group(1)
        if tc_pattern.search(path):
            urls.append(f"https://essential.ecmwf.int{path}")

    if not urls:
        print("ECMWF DISS: 无TC轨迹文件")
        return None

    # 下载并合并
    cache_key = _get_cache_key(date_str, run_time, 'diss')
    cache_path = os.path.join(CACHE_DIR, cache_key)

    try:
        with open(cache_path, 'wb') as out:
            for url in urls:
                r = req_lib.get(url, stream=True, timeout=120)
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=65536):
                    out.write(chunk)

        if os.path.getsize(cache_path) > 0:
            print(f"ECMWF DISS: 下载成功 {cache_path} ({os.path.getsize(cache_path)} bytes)")
            return cache_path
    except Exception as e:
        print(f"ECMWF DISS: 下载合并失败 - {e}")

    return None


def parse_tc_bufr(bufr_path, storm_id_filter=None):
    """解析 ECMWF BUFR 台风轨迹数据

    Args:
        bufr_path: BUFR文件路径
        storm_id_filter: 可选，只提取指定风暴ID的数据

    Returns:
        dict: {storm_id: {storm_name, hres_track, ens_tracks}}
        每个track是预测点列表 [{time, lat, lng, pressure, wind_speed, ...}]
    """
    if bufr_path is None or not os.path.exists(bufr_path):
        return {}

    # 尝试使用 pdbufr (高级接口，更简单)
    try:
        return _parse_with_pdbufr(bufr_path, storm_id_filter)
    except ImportError:
        print("ECMWF BUFR: pdbufr不可用，尝试eccodes")
    except Exception as e:
        print(f"ECMWF BUFR: pdbufr解析失败 - {e}")

    # 尝试使用 eccodes (底层接口)
    try:
        return _parse_with_eccodes(bufr_path, storm_id_filter)
    except ImportError:
        print("ECMWF BUFR: eccodes不可用")
    except Exception as e:
        print(f"ECMWF BUFR: eccodes解析失败 - {e}")

    # 最后尝试: 用requests直接从ECMWF获取简单的文本格式数据
    # Open-Meteo已经提供了部分NWP数据，但BUFR是更全面的来源
    print("ECMWF BUFR: 无可用解析库，请安装 pdbufr 或 eccodes")
    return {}


def _parse_with_pdbufr(bufr_path, storm_id_filter=None):
    """使用 pdbufr 解析 BUFR 文件"""
    import pdbufr

    # 提取风暴标识符列表
    df_storms = pdbufr.read_bufr(
        bufr_path,
        columns=("stormIdentifier", "longStormName"),
    )
    storms_meta = df_storms.drop_duplicates(subset=["stormIdentifier"])

    results = {}

    for _, storm_row in storms_meta.iterrows():
        sid = storm_row["stormIdentifier"]
        sname = storm_row.get("longStormName", "")

        # 跳过编号扰动(如 "70U", "93E")
        if len(sid) == 3 and sid[:2].isdigit() and sid[2].isalpha():
            continue

        if storm_id_filter and sid != storm_id_filter:
            continue

        # 提取该风暴的轨迹数据
        df = pdbufr.read_bufr(
            bufr_path,
            columns=(
                "stormIdentifier",
                "longStormName",
                "ensembleMemberNumber",
                "latitude",
                "longitude",
                "pressureReducedToMeanSeaLevel",
                "windSpeedAt10M",
                "timePeriod",
            ),
            filters={"stormIdentifier": sid},
        )

        # 按集合成员分组
        hres_track = []
        ens_tracks = {}

        for _, row in df.iterrows():
            if row["latitude"] is None or row["longitude"] is None:
                continue

            member = int(row["ensembleMemberNumber"]) if row["ensembleMemberNumber"] else 0
            step_h = int(row["timePeriod"]) if row["timePeriod"] else 0
            lat = float(row["latitude"])
            lng = float(row["longitude"])
            pressure_pa = float(row["pressureReducedToMeanSeaLevel"]) if row["pressureReducedToMeanSeaLevel"] else None
            pressure_hpa = pressure_pa / 100 if pressure_pa else None
            wind_ms = float(row["windSpeedAt10M"]) if row["windSpeedAt10M"] else None

            point = {
                'forecast_hour': step_h,
                'lat': round(lat, 1),
                'lng': round(lng, 1),
                'pressure': round(pressure_hpa) if pressure_hpa else None,
                'wind_speed': round(wind_ms) if wind_ms else None,
            }

            # HRES = member 52, 控制成员 = 51, 扰动成员 = 1-50
            if member == 52:
                hres_track.append(point)
            elif member == 51:
                ens_tracks['control'] = ens_tracks.get('control', [])
                ens_tracks['control'].append(point)
            else:
                ens_tracks[f'member_{member}'] = ens_tracks.get(f'member_{member}', [])
                ens_tracks[f'member_{member}'].append(point)

        # 排序
        hres_track.sort(key=lambda p: p['forecast_hour'])
        for k in ens_tracks:
            ens_tracks[k].sort(key=lambda p: p['forecast_hour'])

        results[sid] = {
            'storm_id': sid,
            'storm_name': sname,
            'hres_track': hres_track,
            'ens_tracks': ens_tracks,
            'source': 'ECMWF BUFR',
        }

    return results


def _parse_with_eccodes(bufr_path, storm_id_filter=None):
    """使用 eccodes 底层接口解析 BUFR 文件"""
    import eccodes

    results = {}

    with open(bufr_path, "rb") as f:
        while True:
            bufr = eccodes.codes_bufr_new_from_file(f)
            if bufr is None:
                break

            # 展开所有描述符
            eccodes.codes_set(bufr, "unpack", 1)

            # 提取元数据
            try:
                storm_id = eccodes.codes_get(bufr, "stormIdentifier")
            except Exception:
                eccodes.codes_release(bufr)
                continue

            try:
                storm_name = eccodes.codes_get(bufr, "longStormName")
            except Exception:
                storm_name = ""

            # 跳过编号扰动
            if len(storm_id) == 3 and storm_id[:2].isdigit() and storm_id[2].isalpha():
                eccodes.codes_release(bufr)
                continue

            if storm_id_filter and storm_id != storm_id_filter:
                eccodes.codes_release(bufr)
                continue

            # 提取时间步
            try:
                time_periods = eccodes.codes_get_array(bufr, "timePeriod")
                unique_periods = sorted(set(int(t) for t in time_periods))
            except Exception:
                unique_periods = [0]

            # 提取集合成员
            try:
                members = eccodes.codes_get_array(bufr, "ensembleMemberNumber")
                unique_members = sorted(set(int(m) for m in members))
            except Exception:
                unique_members = [0]

            num_periods = len(unique_periods)

            hres_track = []
            ens_tracks = {}

            for i, step_h in enumerate(unique_periods):
                # 风暴中心: rank = i * 2 + 2
                rank1 = i * 2 + 2

                try:
                    center_lat = eccodes.codes_get(bufr, f"#{rank1}#latitude")
                    center_lng = eccodes.codes_get(bufr, f"#{rank1}#longitude")
                except Exception:
                    center_lat = center_lng = None

                try:
                    pressure = eccodes.codes_get(bufr, f"#{i+1}#pressureReducedToMeanSeaLevel")
                    pressure_hpa = pressure / 100 if pressure else None
                except Exception:
                    pressure_hpa = None

                try:
                    wind_speed = eccodes.codes_get(bufr, f"#{i+1}#windSpeedAt10M")
                except Exception:
                    wind_speed = None

                if center_lat is not None and center_lng is not None:
                    point = {
                        'forecast_hour': step_h,
                        'lat': round(float(center_lat), 1),
                        'lng': round(float(center_lng), 1),
                        'pressure': round(pressure_hpa) if pressure_hpa else None,
                        'wind_speed': round(float(wind_speed)) if wind_speed else None,
                    }

                    # 根据成员号分组(52=HRES, 51=control, 1-50=扰动)
                    # 在底层接口中，数据是压缩的多子集格式
                    # 简化处理：所有数据作为HRES轨道
                    hres_track.append(point)

            # 排序
            hres_track.sort(key=lambda p: p['forecast_hour'])

            eccodes.codes_release(bufr)

            results[storm_id] = {
                'storm_id': storm_id,
                'storm_name': storm_name,
                'hres_track': hres_track,
                'ens_tracks': ens_tracks,
                'source': 'ECMWF BUFR (eccodes)',
            }

    return results


def fetch_ecmwf_tracks_for_typhoon(lat, lng, hours=168, storm_id=None):
    """获取 ECMWF BUFR 台风轨迹预报，格式化为预测管线兼容格式

    这是 P1 的核心接口：从 ECMWF Open Data 获取官方台风轨迹预报，
    直接提供 ECMWF 全球最准NWP的预测路径。

    Args:
        lat: 台风当前纬度
        lng: 台风当前经度
        hours: 需要的预报时长(默认168=7天)
        storm_id: 可选的风暴标识符(如"07W")

    Returns:
        dict: {method_name: prediction_points} 格式，
        与其他预测方法(Trend/Physics/NWP)输出格式一致
    """
    # 下载BUFR数据
    bufr_path = download_tc_bufr()
    if bufr_path is None:
        return {}

    # 解析
    all_tracks = parse_tc_bufr(bufr_path, storm_id_filter=storm_id)
    if not all_tracks:
        return {}

    results = {}

    # 对每个活跃风暴，提取HRES确定性预报和ENS均值
    for sid, track_data in all_tracks.items():
        hres = track_data.get('hres_track', [])
        if not hres:
            continue

        # 匹配最近的台风(通过位置距离)
        if storm_id is None and lat and lng:
            # 找最近的风暴
            first_hres_point = hres[0]
            dist = math.sqrt(
                (first_hres_point['lat'] - lat) ** 2 +
                (first_hres_point['lng'] - lng) ** 2
            )
            if dist > 15:  # 超过15度(~1600km)跳过
                continue

        # 截取到指定预报时长
        hres_filtered = [p for p in hres if p['forecast_hour'] <= hours]

        # 格式化为预测管线标准格式
        formatted = []
        for p in hres_filtered:
            # 计算强度等级
            pressure = p.get('pressure')
            wind = p.get('wind_speed')
            category = _bufr_intensity_category(pressure, wind)

            # 信心衰减
            h = p['forecast_hour']
            decay = math.exp(-h / max(hours * 2.5, 200))

            formatted.append({
                'time': '',  # 预报时间(需要从base_time+step计算，但BUFR只给步长)
                'forecast_hour': h,
                'lat': p['lat'],
                'lng': p['lng'],
                'pressure': pressure,
                'wind_speed': wind,
                'category': category,
                'confidence': round(0.95 * decay, 2),
                'method_desc': f'ECMWF BUFR HRES({sid})',
            })

        if formatted:
            # ENS集合均值(如果有扰动成员数据)
            ens_tracks = track_data.get('ens_tracks', {})
            if ens_tracks:
                ens_mean = _compute_ens_mean(ens_tracks, hours)
                if ens_mean:
                    results['ecmwf_bufr_ens'] = ens_mean

            results['ecmwf_bufr'] = formatted

    return results


def _compute_ens_mean(ens_tracks, hours):
    """计算ENS集合预报均值轨迹"""
    # 收集所有扰动成员的点
    all_member_points = {}  # forecast_hour -> [points from different members]

    for member_key, track in ens_tracks.items():
        if member_key == 'control':
            continue  # 控制成员单独处理
        for p in track:
            fh = p['forecast_hour']
            if fh > hours:
                continue
            if fh not in all_member_points:
                all_member_points[fh] = {'lats': [], 'lngs': [], 'pressures': [], 'winds': []}
            all_member_points[fh]['lats'].append(p['lat'])
            all_member_points[fh]['lngs'].append(p['lng'])
            if p.get('pressure'):
                all_member_points[fh]['pressures'].append(p['pressure'])
            if p.get('wind_speed'):
                all_member_points[fh]['winds'].append(p['wind_speed'])

    # 计算均值
    mean_track = []
    for fh in sorted(all_member_points.keys()):
        pts = all_member_points[fh]
        n = len(pts['lats'])
        if n < 3:
            continue  # 至少3个成员才有意义

        mean_lat = sum(pts['lats']) / n
        mean_lng = sum(pts['lngs']) / n
        mean_pressure = sum(pts['pressures']) / len(pts['pressures']) if pts['pressures'] else None
        mean_wind = sum(pts['winds']) / len(pts['winds']) if pts['winds'] else None

        # 计算散度(信心度)
        lat_std = math.sqrt(sum((x - mean_lat) ** 2 for x in pts['lats']) / n)
        lng_std = math.sqrt(sum((x - mean_lng) ** 2 for x in pts['lngs']) / n)
        spread = math.sqrt(lat_std ** 2 + lng_std ** 2)

        # 信心度: spread越大信心越低
        # 1° spread ≈ 110km，经验上spread>5°(~550km)时信心很低
        base_confidence = max(0.3, 0.95 - spread * 0.1)

        decay = math.exp(-fh / max(hours * 2.5, 200))
        confidence = round(base_confidence * decay, 2)

        category = _bufr_intensity_category(mean_pressure, mean_wind)

        mean_track.append({
            'time': '',
            'forecast_hour': fh,
            'lat': round(mean_lat, 1),
            'lng': round(mean_lng, 1),
            'pressure': round(mean_pressure) if mean_pressure else None,
            'wind_speed': round(mean_wind) if mean_wind else None,
            'category': category,
            'confidence': confidence,
            'method_desc': f'ECMWF BUFR ENS均值({n}成员)',
            'ensemble_spread_deg': round(spread, 2),
        })

    return mean_track


def _bufr_intensity_category(pressure, wind_speed):
    """BUFR数据台风等级判断"""
    if pressure is None and wind_speed is None:
        return '未知'

    # 优先用气压判断
    if pressure:
        if pressure <= 935:
            return '超强台风(Super TY)'
        elif pressure <= 955:
            return '强台风(STY)'
        elif pressure <= 970:
            return '台风(TY)'
        elif pressure <= 985:
            return '强热带风暴(STS)'
        elif pressure <= 1000:
            return '热带风暴(TS)'
        else:
            return '热带低压(TD)'

    # 用风速判断
    if wind_speed:
        if wind_speed >= 51:
            return '超强台风(Super TY)'
        elif wind_speed >= 42:
            return '强台风(STY)'
        elif wind_speed >= 35:
            return '台风(TY)'
        elif wind_speed >= 25:
            return '强热带风暴(STS)'
        elif wind_speed >= 18:
            return '热带风暴(TS)'
        else:
            return '热带低压(TD)'

    return '未知'


def get_ecmwf_active_storms():
    """获取 ECMWF 当前追踪的所有活跃热带气旋列表

    Returns:
        list: [{storm_id, storm_name, latest_lat, latest_lng, latest_pressure}]
    """
    bufr_path = download_tc_bufr()
    if bufr_path is None:
        return []

    all_tracks = parse_tc_bufr(bufr_path)
    storms = []

    for sid, track_data in all_tracks.items():
        hres = track_data.get('hres_track', [])
        if hres:
            first = hres[0]
            storms.append({
                'storm_id': sid,
                'storm_name': track_data.get('storm_name', ''),
                'lat': first['lat'],
                'lng': first['lng'],
                'pressure': first.get('pressure'),
                'wind_speed': first.get('wind_speed'),
            })

    return storms


def is_bufr_available():
    """检查 ECMWF BUFR 数据获取功能是否可用

    不要求BUFR库一定安装——系统可以退化为仅用Open-Meteo
    """
    try:
        from ecmwf.opendata import Client
        return True
    except ImportError:
        pass

    # 检查是否有eccodes或pdbufr
    try:
        import pdbufr
        return True
    except ImportError:
        pass

    try:
        import eccodes
        return True
    except ImportError:
        pass

    return False
