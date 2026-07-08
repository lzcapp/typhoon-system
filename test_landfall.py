"""
独立测试脚本：测试登陆点检测逻辑
直接调用coastline.py的检测函数，使用真实台风数据验证
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from coastline import detect_landfall_from_segments, detect_landfall, is_over_land, COASTLINE_SEGMENTS

# ============================================================
# 测试1: 模拟一条预测路径（从海上移动到中国沿海）
# ============================================================
print("=" * 60)
print("测试1: 模拟台风从菲律宾海移向中国华东沿海")
print("=" * 60)

# 模拟预测路径点（从菲律宾东部海面向西北移动到浙江/上海一带）
mock_predictions = [
    {"lat": 12.0, "lng": 130.0, "time": "2024-01-01T00:00:00", "pressure": 980, "wind_speed": 30},  # 菲律宾东
    {"lat": 15.0, "lng": 128.0, "time": "2024-01-01T06:00:00", "pressure": 970, "wind_speed": 35},
    {"lat": 18.0, "lng": 126.0, "time": "2024-01-01T12:00:00", "pressure": 960, "wind_speed": 40},
    {"lat": 21.0, "lng": 124.0, "time": "2024-01-01T18:00:00", "pressure": 950, "wind_speed": 45},
    {"lat": 24.0, "lng": 122.0, "time": "2024-01-02T00:00:00", "pressure": 955, "wind_speed": 42},  # 台湾附近
    {"lat": 27.0, "lng": 121.0, "time": "2024-01-02T06:00:00", "pressure": 965, "wind_speed": 38},  # 浙江沿海
    {"lat": 30.0, "lng": 120.0, "time": "2024-01-02T12:00:00", "pressure": 975, "wind_speed": 32},  # 杭州湾
    {"lat": 32.0, "lng": 119.5, "time": "2024-01-02T18:00:00", "pressure": 985, "wind_speed": 25},  # 内陆
]

# 检查每个点是否在陆地上
print("\n各点距海岸线距离:")
for i, p in enumerate(mock_predictions):
    is_land, coast_name, dist = is_over_land(p["lat"], p["lng"], margin_deg=0.3)
    print(f"  Step {i}: ({p['lat']},{p['lng']}) -> is_land={is_land}, coast={coast_name}, dist={dist:.3f}°")

# 检测登陆点
print("\n检测登陆点 (detect_landfall):")
lf = detect_landfall(mock_predictions, margin_deg=0.3)
if lf:
    print(f"  ✅ 找到登陆点!")
    print(f"     位置: ({lf['lat']}, {lf['lng']})")
    print(f"     海岸: {lf['coast_name']}")
    print(f"     时间: {lf['time']}")
    print(f"     距起点小时: {lf['hours_from_base']}")
else:
    print(f"  ❌ 未找到登陆点")

print("\n检测登陆点 (detect_landfall_from_segments):")
lf2 = detect_landfall_from_segments(mock_predictions, margin_deg=0.3)
if lf2:
    print(f"  ✅ 找到登陆点!")
    print(f"     位置: ({lf2['lat']}, {lf2['lng']})")
    print(f"     海岸: {lf2['coast_name']}")
    print(f"     时间: {lf2['time']}")
    print(f"     距起点小时: {lf2['hours_from_base']}")
else:
    print(f"  ❌ 未找到登陆点")

# ============================================================
# 测试2: 利用真实台风数据（202106 烟花）模拟预测路径
# ============================================================
print("\n" + "=" * 60)
print("测试2: 使用真实台风202106（烟花）的最后位置点来趋势外推")
print("=" * 60)

import json
import math

isc_dir = os.path.join(os.path.dirname(__file__), 'data', 'isc')

# 找到202106的数据
target = None
for fn in sorted(os.listdir(isc_dir)):
    if not fn.endswith('.json') or not fn.startswith('2021'):
        continue
    with open(os.path.join(isc_dir, fn)) as f:
        data = json.load(f)
    for t in data:
        if t.get('tfbh') == '202106' or t.get('ident') == '202106':
            target = t
            break
    if target:
        break

if target:
    points = target['points']
    print(f"台风名称: {target.get('name','')}({target.get('ename','')})")
    print(f"数据点总数: {len(points)}")
    print(f"最后位置: ({points[-1]['lat']}, {points[-1]['lng']})")
    print(f"最后气压: {points[-1].get('pressure', '?')} hPa")

    # 计算前几个点之间的移动趋势
    if len(points) >= 4:
        recent = points[-4:]
        avg_dlat = sum(recent[i+1]['lat'] - recent[i]['lat'] for i in range(len(recent)-1)) / (len(recent)-1)
        avg_dlng = sum(recent[i+1]['lng'] - recent[i]['lng'] for i in range(len(recent)-1)) / (len(recent)-1)
        print(f"平均每步移动: dlat={avg_dlat:.2f}°, dlng={avg_dlng:.2f}°")

        # 生成12步预测（72小时）
        last = points[-1]
        last_lat = last['lat']
        last_lng = last['lng']
        trend_preds = []
        for i in range(1, 13):
            pred_lat = last_lat + avg_dlat * i
            pred_lng = last_lng + avg_dlng * i
            trend_preds.append({
                "lat": round(pred_lat, 1),
                "lng": round(pred_lng, 1),
                "time": f"+{i*6}h",
                "pressure": last.get('pressure', 1000),
                "wind_speed": last.get('speed', 0),
            })

        print("\n趋势外推预测路径:")
        for i, p in enumerate(trend_preds):
            is_land, coast_name, dist = is_over_land(p["lat"], p["lng"], margin_deg=0.3)
            flag = " ⬅️ 陆地" if is_land else ""
            print(f"  +{(i+1)*6}h: ({p['lat']},{p['lng']}) dist={dist:.3f}° coast={coast_name}{flag}")

        print("\n登陆检测 (detect_landfall_from_segments):")
        lf_real = detect_landfall_from_segments(trend_preds, margin_deg=0.3)
        if lf_real:
            print(f"  ✅ 检测到登陆!")
            print(f"     位置: ({lf_real['lat']}, {lf_real['lng']})")
            print(f"     海岸: {lf_real['coast_name']}")
            print(f"     时间: {lf_real['time']}")
            print(f"     气压: {lf_real.get('pressure', '?')} hPa")
            print(f"     风速: {lf_real.get('wind_speed', '?')} m/s")
            print(f"     距起点: {lf_real['hours_from_base']}小时")
        else:
            print(f"  ❌ 未检测到登陆")
            # 检查为什么
            print("\n  详细诊断:")
            for i, p in enumerate(trend_preds):
                is_land, coast_name, dist = is_over_land(p["lat"], p["lng"], margin_deg=0.3)
                print(f"    +{(i+1)*6}h ({p['lat']},{p['lng']}): is_land={is_land}, dist={dist:.4f}°")

# ============================================================
# 测试3: 边界条件测试 - 只有1个点
# ============================================================
print("\n" + "=" * 60)
print("测试3: 边界条件测试")
print("=" * 60)

# 空列表
result = detect_landfall_from_segments([], margin_deg=0.3)
print(f"  空列表: {result}")

# 单个点
result = detect_landfall_from_segments([{"lat": 20, "lng": 120}], margin_deg=0.3)
print(f"  单个点: {result}")

# 两个点都在海上
points_sea = [
    {"lat": 15, "lng": 130, "time": "2024-01-01T00:00:00", "pressure": 980, "wind_speed": 30},
    {"lat": 18, "lng": 128, "time": "2024-01-01T06:00:00", "pressure": 970, "wind_speed": 35},
]
result = detect_landfall_from_segments(points_sea, margin_deg=0.3)
print(f"  两点海上: {result}")

# ============================================================
# 测试4: 验证 coastline segment 的结构
# ============================================================
print("\n" + "=" * 60)
print("测试4: 海岸线段数据完整性")
print("=" * 60)

print(f"海岸线段总数: {len(COASTLINE_SEGMENTS)}")
for name, points in COASTLINE_SEGMENTS:
    print(f"  {name}: {len(points)}个点, 范围({min(p[0] for p in points):.1f}°N-{max(p[0] for p in points):.1f}°N, {min(p[1] for p in points):.1f}°E-{max(p[1] for p in points):.1f}°E)")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
