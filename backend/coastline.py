"""
NW Pacific 海岸线数据 & 登陆检测模块

使用简化海岸线段，检测台风预测路径是否与陆地相交。
覆盖区域: 100°E-150°E, 5°N-45°N (西北太平洋台风主要影响区)
"""

import math
from collections import namedtuple

# 海岸线段: (名称, [(lat, lng), ...]) — 从左到右/从南到北排列
# 精确到 0.1-0.5°，专为台风登陆场景优化

COASTLINE_SEGMENTS = [
    # ============================================================
    # 中国东南沿海（广东→福建→浙江→上海→江苏→山东）
    # ============================================================
    ("中国华南", [
        (21.5, 109.5),  # 广西-越南交界
        (21.5, 110.0), (21.4, 110.5), (21.3, 110.8),  # 湛江/雷州半岛东
        (21.0, 111.2), (21.5, 111.5), (21.8, 112.0),   # 阳江
        (22.2, 113.0), (22.3, 113.5), (22.5, 114.0),   # 珠江口西
        (22.3, 114.3), (22.5, 114.5),                    # 香港附近
        (22.7, 114.7), (23.0, 115.0), (23.3, 115.5),    # 粤东
        (23.5, 116.5), (23.8, 117.0), (24.0, 117.5),    # 汕头-漳州
    ]),
    ("中国华东", [
        (24.0, 117.5),  # 接华南
        (24.5, 118.0), (25.0, 118.5), (25.5, 119.0),    # 泉州-福州
        (26.0, 119.5), (26.5, 119.7), (27.0, 120.0),    # 宁德-温州
        (27.5, 120.5), (28.0, 121.0), (28.5, 121.5),    # 台州-宁波
        (29.0, 121.7), (29.5, 121.8), (30.0, 121.7),    # 舟山
        (30.5, 121.5), (31.0, 121.7), (31.5, 121.3),    # 上海-南通
        (32.0, 121.0), (32.5, 120.8), (33.0, 120.5),    # 盐城
    ]),
    ("中国华北", [
        (33.0, 120.5),  # 接华东
        (33.5, 120.2), (34.0, 119.8), (34.5, 119.5),    # 连云港
        (35.0, 119.3), (35.5, 119.5), (36.0, 120.0),    # 日照-青岛
        (36.5, 120.5), (37.0, 121.0), (37.5, 121.5),    # 威海
        (37.8, 122.0), (37.5, 122.5),                     # 荣成
    ]),
    # 辽东半岛（偶有台风影响）
    ("辽东半岛", [
        (38.5, 121.0),  # 大连南
        (39.0, 121.5), (39.5, 122.0),                    # 大连
        (40.0, 122.5), (40.5, 123.0),                    # 丹东附近
    ]),

    # ============================================================
    # 海南岛
    # ============================================================
    ("海南岛东岸", [
        (20.0, 110.0),
        (19.8, 110.2), (19.5, 110.5), (19.2, 110.7),    # 海口东
        (19.0, 110.8), (18.8, 110.5), (18.5, 110.3),    # 文昌
        (18.3, 110.0), (18.2, 109.8), (18.2, 109.5),    # 三亚
    ]),
    ("海南岛北岸", [
        (20.0, 110.0),
        (20.0, 109.5), (20.0, 109.2),                     # 北岸西段
        (19.8, 109.0),                                     # 澄迈
    ]),

    # ============================================================
    # 台湾岛
    # ============================================================
    ("台湾东岸", [
        (25.3, 121.5),  # 台北/基隆
        (24.9, 121.8), (24.5, 121.8), (24.0, 121.7),    # 宜兰-花莲
        (23.5, 121.5), (23.0, 121.3), (22.5, 121.0),    # 台东
        (22.0, 120.8), (21.9, 120.8),                     # 恒春半岛
    ]),
    ("台湾西岸", [
        (25.3, 121.5),  # 台北（连接点）
        (25.0, 121.0), (24.5, 120.5), (24.0, 120.3),    # 新竹-台中
        (23.5, 120.2), (23.0, 120.2), (22.5, 120.3),    # 彰化-嘉义
        (22.0, 120.2),                                     # 台南
    ]),

    # ============================================================
    # 菲律宾 (吕宋岛)
    # ============================================================
    ("菲律宾吕宋东岸", [
        (18.5, 121.5),  # 北端
        (18.0, 121.8), (17.5, 122.0), (17.0, 122.2),
        (16.5, 122.3), (16.0, 122.3), (15.5, 122.2),
        (15.0, 122.0), (14.5, 122.5), (14.0, 122.8),
        (13.5, 123.0), (13.0, 123.5),  # 黎牙实比
    ]),
    ("菲律宾萨马-莱特东岸", [
        (12.5, 124.5),
        (12.0, 125.0), (11.5, 125.0), (11.0, 125.0),
        (10.5, 125.5), (10.0, 125.5),
    ]),
    ("菲律宾棉兰老东岸", [
        (9.5, 126.0),
        (9.0, 126.0), (8.5, 126.3), (8.0, 126.5),
        (7.5, 126.5), (7.0, 126.5), (6.5, 126.2),
        (6.0, 126.0), (5.5, 125.5),
    ]),

    # ============================================================
    # 越南海岸
    # ============================================================
    ("越南", [
        (21.5, 108.0),  # 中越边境
        (21.0, 107.5), (20.5, 107.0), (20.0, 106.8),    # 北部
        (19.5, 106.5), (19.0, 106.3), (18.5, 106.2),    # 荣市
        (18.0, 106.0), (17.5, 106.5), (17.0, 107.0),    # 洞海
        (16.5, 108.0), (16.0, 108.2), (15.5, 108.5),    # 岘港
        (15.0, 108.7), (14.5, 109.0), (14.0, 109.2),
        (13.5, 109.2), (13.0, 109.3), (12.5, 109.2),
        (12.0, 109.2), (11.5, 109.0), (11.0, 108.6),
        (10.5, 108.0), (10.0, 107.5), (9.5, 107.0),
        (9.0, 106.5), (8.5, 106.0),
    ]),

    # ============================================================
    # 朝鲜半岛
    # ============================================================
    ("韩国南岸", [
        (34.5, 126.0),  # 西南端
        (34.5, 126.5), (34.7, 127.0), (35.0, 127.5),
        (35.0, 128.0), (35.0, 128.5), (35.0, 129.0),    # 釜山
        (35.5, 129.3), (36.0, 129.5),  # 浦项
    ]),
    ("韩国西岸", [
        (37.5, 126.5),  # 仁川
        (37.0, 126.3), (36.5, 126.5), (36.0, 126.7),
        (35.5, 126.5), (35.0, 126.5), (34.5, 126.3),
    ]),

    # ============================================================
    # 日本
    # ============================================================
    ("日本九州南岸", [
        (31.0, 130.0),  # 西端
        (31.0, 130.5), (31.2, 131.0), (31.5, 131.5),
        (31.5, 132.0), (31.3, 132.5),
    ]),
    ("日本九州东岸", [
        (33.0, 131.5),
        (32.5, 131.7), (32.0, 131.8), (31.5, 131.8),
        (31.3, 131.5),
    ]),
    ("日本四国南岸", [
        (33.0, 133.0), (32.8, 133.2), (32.5, 133.5),
        (33.0, 134.0), (33.5, 134.5),
    ]),
    ("日本本州南岸", [
        (34.5, 135.0),  # 和歌山
        (34.5, 136.0), (34.5, 137.0), (34.5, 138.0),    # 名古屋南
        (35.0, 138.5), (35.0, 139.0), (35.0, 139.5),    # 东京湾
        (35.0, 140.0), (35.5, 140.5),
    ]),
    ("日本东北东岸", [
        (35.5, 140.5),
        (36.0, 141.0), (36.5, 141.0), (37.0, 141.0),
        (37.5, 141.0), (38.0, 141.0), (38.5, 141.5),
        (39.0, 141.5), (39.5, 142.0), (40.0, 142.0),
    ]),

    # ============================================================
    # 琉球群岛
    # ============================================================
    ("琉球群岛", [
        (24.5, 124.0), (25.0, 124.5), (25.5, 125.0),
        (26.0, 125.5), (26.5, 126.0), (27.0, 126.5),
        (27.5, 127.0), (28.0, 128.0), (28.5, 129.0),
        (29.0, 129.5), (29.5, 130.0), (30.0, 130.5),
    ]),
]


def _segments_intersect(a1, a2, b1, b2):
    """检查两条线段是否相交，返回交点 (lat, lng) 或 None
    
    使用参数方程法: 两条线段 p = a1 + s*(a2-a1), p = b1 + t*(b2-b1)
    相交条件: 0 <= s <= 1, 0 <= t <= 1
    """
    x1, y1 = a1[1], a1[0]  # (lng, lat) -> (x, y) for math
    x2, y2 = a2[1], a2[0]
    x3, y3 = b1[1], b1[0]
    x4, y4 = b2[1], b2[0]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # 平行或重合
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        # 交点坐标 (lng, lat)
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (iy, ix)
    
    return None


def _point_to_segment_distance(point, seg_a, seg_b):
    """计算点到线段的最短距离（度）"""
    lat, lng = point
    ax, ay = seg_a[1], seg_a[0]
    bx, by = seg_b[1], seg_b[0]
    
    # 线段长度平方
    dx = bx - ax
    dy = by - ay
    len2 = dx * dx + dy * dy
    
    if len2 < 1e-10:
        return math.sqrt((lng - ax)**2 + (lat - ay)**2)
    
    # 投影参数
    t = max(0, min(1, ((lng - ax) * dx + (lat - ay) * dy) / len2))
    
    # 投影点
    px = ax + t * dx
    py = ay + t * dy
    
    return math.sqrt((lng - px)**2 + (lat - py)**2)


def is_over_land(lat, lng, margin_deg=0.3):
    """判断一个点是否在陆地上（简单版：距离任一海岸线不足margin_deg）
    
    这不是真正的point-in-polygon，但对于台风登陆检测足够了——
    台风中心距离海岸线0.3°（约33km）内即视为"接近登陆"。
    
    返回: (is_land: bool, nearest_coast_name: str, distance_deg: float)
    """
    min_dist = float('inf')
    nearest_name = ''
    
    for name, points in COASTLINE_SEGMENTS:
        for i in range(len(points) - 1):
            dist = _point_to_segment_distance((lat, lng), points[i], points[i + 1])
            if dist < min_dist:
                min_dist = dist
                nearest_name = name
    
    return (min_dist < margin_deg, nearest_name, min_dist)


def detect_landfall(prediction_points, margin_deg=0.3):
    """检测预测路径中的登陆点
    
    遍历预测路径点，找到第一个"靠近海岸线"的点作为登陆点。
    
    参数:
        prediction_points: [(lat, lng, time_str, pressure, wind_speed), ...] 
                           或 [{'lat':.., 'lng':.., 'time':..}, ...]
        margin_deg: 距离海岸线多少度内视为"登陆"（默认0.3°≈33km）
    
    返回: {
        'found': bool,
        'index': int,           # 预测点索引
        'lat': float,
        'lng': float,
        'time': str,
        'pressure': float,
        'wind_speed': float,
        'coast_name': str,      # 登陆海岸名称
        'distance_km': float,   # 距海岸线距离(km)
        'hours_from_base': int, # 距当前时间的小时数
    } 或 None (未检测到登陆)
    """
    if not prediction_points or len(prediction_points) < 2:
        return None
    
    for i, pt in enumerate(prediction_points):
        if isinstance(pt, dict):
            lat = pt.get('lat', 0)
            lng = pt.get('lng', 0)
        else:
            lat, lng = pt[0], pt[1]
        
        is_land, coast_name, dist = is_over_land(lat, lng, margin_deg)
        
        if is_land:
            # 提取完整信息
            if isinstance(pt, dict):
                return {
                    'found': True,
                    'index': i,
                    'lat': lat,
                    'lng': lng,
                    'time': pt.get('time', ''),
                    'pressure': pt.get('pressure', 0),
                    'wind_speed': pt.get('wind_speed', 0),
                    'coast_name': coast_name,
                    'distance_km': round(dist * 111, 1),
                    'hours_from_base': (i + 1) * 6,  # 假设6h步长
                }
            else:
                return {
                    'found': True,
                    'index': i,
                    'lat': lat,
                    'lng': lng,
                    'time': pt[2] if len(pt) > 2 else '',
                    'pressure': pt[3] if len(pt) > 3 else 0,
                    'wind_speed': pt[4] if len(pt) > 4 else 0,
                    'coast_name': coast_name,
                    'distance_km': round(dist * 111, 1),
                    'hours_from_base': (i + 1) * 6,
                }
    
    return None


def detect_landfall_from_segments(prediction_points, margin_deg=0.3):
    """更精确的登陆检测：使用线段相交法
    
    检测预测路径线段是否与海岸线段相交。
    相交点 = 准确的登陆位置（而非仅靠近海岸的点）。
    
    返回格式同 detect_landfall()
    """
    if not prediction_points or len(prediction_points) < 2:
        return None
    
    for i in range(len(prediction_points) - 1):
        p1 = prediction_points[i]
        p2 = prediction_points[i + 1]
        
        if isinstance(p1, dict):
            a_lat, a_lng = p1['lat'], p1['lng']
            b_lat, b_lng = p2['lat'], p2['lng']
        else:
            a_lat, a_lng = p1[0], p1[1]
            b_lat, b_lng = p2[0], p2[1]
        
        a = (a_lat, a_lng)
        b = (b_lat, b_lng)
        
        best_dist = float('inf')
        best_intersection = None
        best_coast_name = ''
        
        for name, coast_pts in COASTLINE_SEGMENTS:
            for j in range(len(coast_pts) - 1):
                c = coast_pts[j]
                d = coast_pts[j + 1]
                
                intersection = _segments_intersect(a, b, c, d)
                if intersection:
                    # 检查是否是从海上→陆地（而非陆→海）
                    # 简化：A点距海岸线距离 > margin_deg（在海上），B点距海岸线距离 < margin_deg
                    a_dist = _point_to_segment_distance(a, c, d)
                    if a_dist > margin_deg * 0.8:
                        dist = math.sqrt((intersection[0] - a_lat)**2 + (intersection[1] - a_lng)**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_intersection = intersection
                            best_coast_name = name
        
        if best_intersection:
            # 用下一个预测点的时间信息
            if isinstance(p2, dict):
                time_str = p2.get('time', '')
                pressure = p2.get('pressure', 0)
                wind_speed = p2.get('wind_speed', 0)
            else:
                time_str = p2[2] if len(p2) > 2 else ''
                pressure = p2[3] if len(p2) > 3 else 0
                wind_speed = p2[4] if len(p2) > 4 else 0
            
            return {
                'found': True,
                'index': i + 1,
                'lat': round(best_intersection[0], 1),
                'lng': round(best_intersection[1], 1),
                'time': time_str,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'coast_name': best_coast_name,
                'distance_km': round(best_dist * 111, 1),
                'hours_from_base': (i + 1) * 6,
            }
    
    # 线段相交未找到，回退到点距离法
    return detect_landfall(prediction_points, margin_deg)


def get_coastline_geojson():
    """将海岸线数据转换为 GeoJSON 格式（用于前端地图展示）"""
    features = []
    for name, points in COASTLINE_SEGMENTS:
        coords = [[lng, lat] for lat, lng in points]  # GeoJSON: [lng, lat]
        features.append({
            "type": "Feature",
            "properties": {"name": name},
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        })
    return {"type": "FeatureCollection", "features": features}
