import numpy as np
import pandas as pd


# ================================================================
# --- 构建多项式模型函数 ---
# ================================================================
def build_poly_models(model, temps, time_max=2000, poly_order=3):
    """
    为每个温度拟合多项式 (time -> TVB-N)
    返回字典 {temp: poly_model}
    """
    poly_models = {}
    time_grid = np.arange(0, time_max+1)

    for temp in temps:
        # 构造输入特征
        features = np.array([[t, np.log1p(t), temp, temp**2, t*temp] for t in time_grid])
        amines = model.predict(features)

        # 多项式拟合
        coeffs = np.polyfit(time_grid, amines, poly_order)
        poly_models[temp] = np.poly1d(coeffs)

    return poly_models


# ================================================================
# --- TVB-N 动态预测核心函数 (基于多项式导数) ---
# ================================================================
def calculate_dynamic_tvbn_value(last_abnormal_value, last_monitor_time, rec_time,
                                 current_temp, predictor, poly_models=None,
                                 poly_order=3, max_rate=0.3, time_max=2000):
    """
    TVB-N 动态含量计算函数 - 基于多项式导数的改进方法

    参数
    ----
    last_abnormal_value : float
        上一次的 TVB-N 含量
    last_monitor_time : datetime
        上一次监测的时间
    rec_time : datetime
        当前记录的时间
    current_temp : float
        当前温度
    predictor : 对象
        训练好的模型包装器，需有 predictor.model
    poly_models : dict, 可选
        已构建的 {温度: 多项式模型} 字典，若为 None 则自动构建
    poly_order : int
        多项式阶数 (默认 3)
    max_rate : float
        最大允许的增长速率 (mg/100g/h)
    time_max : int
        多项式拟合的最大时间范围
    """
    # 1. 计算时间步长 (单位：小时)
    if pd.isna(last_monitor_time) or pd.isna(rec_time):
        return last_abnormal_value
    time_delta_hours = (rec_time - last_monitor_time).total_seconds() / 3600.0
    if time_delta_hours <= 0:
        return last_abnormal_value

    # 2. 如果没有传 poly_models，就临时构建
    if poly_models is None:
        # 这里可以根据常见温度区间设定，比如 -20 到 10 °C
        temps = np.linspace(-20, 10, 31)
        poly_models = predictor.get_poly_models(temps, time_max, poly_order)


    # 3. 找到最接近的温度多项式
    closest_temp = min(poly_models.keys(), key=lambda x: abs(x-current_temp))
    poly_model = poly_models[closest_temp]

    # 4. 用多项式的导数近似速率，并限制最大值
    # 这里的 t 可以简单取 "累计小时数"，
    # 由于实际没有全局累计时间，可直接取上一个点的预测位置近似
    t_eff = min(time_max, int(time_delta_hours))
    rate = min(max_rate, max(0, poly_model.deriv()(t_eff)))

    # 5. 计算含量累积
    amine_increase = rate * time_delta_hours
    predicted_value_now = last_abnormal_value + amine_increase

    return max(predicted_value_now, last_abnormal_value)
