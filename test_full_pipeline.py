"""
完整链路测试：从原始数据 -> predict_path -> landfalls检测 -> 返回格式
直接模拟 app.py 中的完整调用链
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from coastline import detect_landfall_from_segments, detect_landfall, is_over_land

# 直接模拟 predict_path 函数中的趋势外推
def trend_extrapolation(points, hours):
    """模拟趋势外推方法（与 app.py 中 TyphoonPredictor._trend_extrapolation 等价）"""
    from datetime import datetime, timedelta
    
    n = min(6, len(points))
    recent = points[-n:]
    
    total_lat_delta = 0
    total_lng_delta = 0
    
    for i in range(1, len(recent)):
        total_lat_delta += recent[i]['lat'] - recent[i-1]['lat']
        total_lng_delta += recent[i]['lng'] - recent[i-1]['lng']
    
    avg_lat_rate = total_lat_delta / (n - 1) if n > 1 else 0
    avg_lng_rate = total_lng_delta / (n - 1) if n > 1 else 0
    
    last_point = points[-1]
    predictions = []
    dt = 6
    
    for h in range(dt, hours + 1, dt):
        pred_lat = last_point['lat'] + avg_lat_rate * h
        pred_lng = last_point['lng'] + avg_lng_rate * h
        
        try:
            base_time = datetime.fromisoformat(last_point['time'].replace('Z', '+00:00'))
            pred_time = (base_time + timedelta(hours=h)).isoformat()
        except:
            pred_time = f"+{h}h"
        
        predictions.append({
            'time': pred_time,
            'lat': round(pred_lat, 1),
            'lng': round(pred_lng, 1),
            'pressure': round(last_point.get('pressure', 0) or 1000),
            'wind_speed': round(last_point.get('speed', 0) or 0, 1),
        })
    
    return predictions


# ============================================================
# 加载真实台风数据
# ============================================================
isc_dir = os.path.join(os.path.dirname(__file__), 'data', 'isc')

# 找所有最后位置在海上的台风（这些才是真正需要预测登陆的）
sea_typhoons = []

for fn in sorted(os.listdir(isc_dir)):
    if not fn.endswith('.json'):
        continue
    with open(os.path.join(isc_dir, fn)) as f:
        data = json.load(f)
    for t in data:
        pts = t.get('points', [])
        if len(pts) < 6:
            continue
        lp = pts[-1]
        lat, lng = lp.get('lat', 0), lp.get('lng', 0)
        
        # 判断是否在海上（简单启发式）
        # 在西北太平洋海域：5°N-50°N, 100°E-180°E 但不是明显在陆地上的坐标
        # 粗略排除明显在陆地的点
        is_at_sea = True
        # 中国内陆：22-50N, 100-120E
        if 22 < lat < 45 and 100 < lng < 118:
            is_at_sea = False
        # 如果经度>125，基本在海上或近海
        # 如果lng > 130，肯定在海上
        
        if is_at_sea:
            land_count = len(t.get('land', []))
            sea_typhoons.append({
                'id': t.get('tfbh', ''),
                'name': f"{t.get('name','')}({t.get('ename','')})",
                'points': pts,
                'land_count': land_count,
                'last_lat': lat,
                'last_lng': lng,
            })

print(f"找到 {len(sea_typhoons)} 个最后位置在海上的台风")

# 按数据点数量排序
sea_typhoons.sort(key=lambda x: len(x['points']), reverse=True)

# ============================================================
# 对每个台风运行趋势外推 + landfall检测
# ============================================================
print("\n" + "=" * 80)
print("完整链路测试：预测 -> landfall 检测")
print("=" * 80)

found_landfall_count = 0

for st in sea_typhoons[:20]:
    tid = st['id']
    name = st['name']
    pts = st['points']
    last = pts[-1]
    
    # 运行趋势外推
    try:
        predictions = trend_extrapolation(pts, hours=120)
    except Exception as e:
        print(f"  {tid} {name}: 趋势外推失败: {e}")
        continue
    
    # 检测登陆
    lf = detect_landfall_from_segments(predictions, margin_deg=0.3)
    
    if lf:
        found_landfall_count += 1
        print(f"  ✅ {tid} {name}")
        print(f"     起点: ({last['lat']},{last['lng']}) | 预测步数: {len(predictions)}")
        print(f"     登陆点: ({lf['lat']},{lf['lng']}) | 海岸: {lf['coast_name']} | {lf['hours_from_base']}h后")
        print(f"     气压: {lf.get('pressure','?')}hPa | 风速: {lf.get('wind_speed','?')}m/s")
        
        # 验证返回格式是否与前端期望一致
        required_keys = ['found', 'lat', 'lng', 'time', 'coast_name', 'pressure', 'wind_speed', 'hours_from_base']
        missing = [k for k in required_keys if k not in lf]
        if missing:
            print(f"     ⚠️ 缺少字段: {missing}")
    else:
        # 只在部分案例上打印未找到信息
        pass

print(f"\n总结: {found_landfall_count}/{min(20, len(sea_typhoons))} 个台风检测到登陆点")

# ============================================================
# 详细分析一个具体案例
# ============================================================
print("\n" + "=" * 80)
print("详细案例: 找一个能检测到登陆的台风，完整展示数据流")
print("=" * 80)

# 找一个有登陆检测的台风详细分析
for st in sea_typhoons[:30]:
    tid = st['id']
    pts = st['points']
    
    try:
        predictions = trend_extrapolation(pts, hours=120)
    except:
        continue
    
    lf = detect_landfall_from_segments(predictions, margin_deg=0.3)
    if not lf:
        continue
    
    print(f"\n台风: {st['id']} {st['name']}")
    print(f"数据点: {len(pts)}, 最后观测: ({st['last_lat']},{st['last_lng']})")
    
    # 显示最后5个观测点
    print("\n最后5个观测点:")
    for p in pts[-5:]:
        print(f"  {p.get('time','')[:19]} -> ({p['lat']},{p['lng']}) 气压:{p.get('pressure','?')}hPa")
    
    print("\n预测的12个步长（120小时）:")
    for i, p in enumerate(predictions):
        is_land, coast_name, dist = is_over_land(p['lat'], p['lng'], margin_deg=0.3)
        flag = " 🔴 LAND" if is_land else ""
        if i < 6:
            print(f"  +{(i+1)*6}h: ({p['lat']},{p['lng']}) dist_to_coast={dist:.3f}°{flag}")
    
    print(f"\n✅ 检测到登陆:")
    print(f"   位置: ({lf['lat']},{lf['lng']})")
    print(f"   海岸: {lf['coast_name']}")
    print(f"   时间: {lf.get('time','')[:19]}")
    print(f"   预计{lf['hours_from_base']}小时后")
    print(f"   登陆强度: {lf.get('pressure','?')}hPa, {lf.get('wind_speed','?')}m/s")
    
    # 显示完整返回JSON（模拟API响应中的landfalls字段）
    print("\n模拟的 API 响应中 landfalls 字段:")
    simulated_response = {
        'landfalls': {
            'trend': lf
        },
        'typhoon_id': tid,
        'method': 'trend',
        'base_lat': st['last_lat'],
        'base_lng': st['last_lng'],
    }
    print(json.dumps(simulated_response, ensure_ascii=False, indent=2))
    break

# ============================================================
# 检查前端渲染逻辑匹配
# ============================================================
print("\n" + "=" * 80)
print("前端渲染逻辑检查")
print("=" * 80)

# 模拟前端 loadAndRenderPrediction 的 landfall 处理逻辑
print("""
前端代码检查清单:

1. API调用: fetch('/api/typhoons/predict/' + tfbh) ✅ (line 1387)
2. 检查 landfalls 非空: if (data.landfalls && Object.keys(data.landfalls).length > 0) ✅ (line 1496)
3. 提取 ensemble 优先: const lfEnsemble = data.landfalls['ensemble'] ✅ (line 1498)
4. 渲染登陆摘要面板: html = '预测登陆点...' + html ✅ (lines 1502-1521)
5. 地图标记: L.marker([lf.lat, lf.lng], ...) ✅ (line 1545)
6. 脉冲动画CSS: @keyframes landfallPulse ✅ (line 442-444)

登陆点无法显示的可能原因:
""")

# 逐一分析
print("A) 后端未返回 landfalls 数据")
print("   -> 如果台风预测路径不与任何海岸线相交, landfalls 为空对象 {}")
print("   -> 这发生在: (1)台风已在陆地 (2)路径完全在海上 (3)海岸线段覆盖不够密")
print()
print("B) prediction 中只有 forecast_* 方法")
print("   -> predict_path 跳过 forecast_ 开头的 key (line 371-372)")
print("   -> 如果只有机构预报, landfalls 会是空的")
print()
print("C) detect_landfall_from_segments 返回 None")
print("   -> 边界情况: predictions < 2 个点, 或线段不相交且点都不近岸")
print()
print("D) 前端 network error 或 JSON 解析失败")
print("   -> 如果服务器超时或返回非JSON, loadAndRenderPrediction 不会执行")
print()
print("E) showPrediction 为 false")
print("   -> 但用户点击预测按钮时始终调用 loadAndRenderPrediction (line 1335)")
