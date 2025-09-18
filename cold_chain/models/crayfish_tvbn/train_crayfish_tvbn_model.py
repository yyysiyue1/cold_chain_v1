# -*- coding: utf-8 -*-
"""
train_crayfish_tvbn_model.py  （无噪声/无温度扰动版，单位 mg/100g）

目的：
1) 加载清洗后的 TVB-N 实验数据（单位：mg/100g，含 t=0）。
2) 仅用样条插值平滑扩充数据（不加噪声、不做温度扰动）。
3) 训练 XGBoost 回归模型预测 TVB-N（mg/100g）。
4) 将模型保存为 tvbn_predictor_model_smooth.pkl。
5) 在原始采样点上评估与可视化。

依赖：numpy, scipy, scikit-learn, xgboost, matplotlib, joblib
"""

import numpy as np
import joblib
from scipy.interpolate import make_interp_spline
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

print("--- TVB-N Model Training (SMOOTH, no noise/perturbation) ---")

# ------------------------------------------------------------
# Step 1: 原始实验数据（单位 mg/100g）
# ------------------------------------------------------------
print("\n[Step 1] Loading original experimental data...")

temp_values = [3.5, 0, -3, -18]
time_values_days = [
    np.array([1, 3, 6, 10, 15, 21]),  # 3.5°C
    np.array([1, 3, 6, 10, 15, 21]),  # 0°C
    np.array([1, 3, 6, 10, 15, 21]),  # -3°C
    np.array([14, 28, 42, 56, 70, 84])  # -18°C
]
tvbn_values = [
    np.array([9.5, 11, 29, 42.5, 51, 80]),  # 3.5°C
    np.array([9, 10, 15.5, 25, 34, 49]),    # 0°C
    np.array([8, 8.5, 10, 11, 13, 19]),     # -3°C
    np.array([8.6, 11.1, 13.2, 13.5, 13.6, 14.0])  # -18°C（末值回到14.0以与清洗版一致）
]

initial_tvbn = 8.0  # t=0 起始值 mg/100g

time_values_hours = []
amine_values = []
for i, T in enumerate(temp_values):
    hours = time_values_days[i] * 24
    time_with_zero = np.insert(hours, 0, 0)
    tvbn_with_zero = np.insert(tvbn_values[i], 0, initial_tvbn)
    time_values_hours.append(time_with_zero)
    amine_values.append(tvbn_with_zero)

print("✅ Data prepared (hours + t=0).")

# ------------------------------------------------------------
# Step 2: 数据增强（仅样条插值，不加噪声/无温度扰动）
# ------------------------------------------------------------
print("\n[Step 2] Building smooth training set (interpolation only)...")

X_train_list, y_train_list = [], []
np.random.seed(42)

for i, T in enumerate(temp_values):
    t_raw = time_values_hours[i]
    y_raw = amine_values[i]

    # 样条插值，得到平滑曲线
    spline = make_interp_spline(t_raw, y_raw, k=3)
    # 细时间网格（可按需调整密度）
    interp_time = np.linspace(0, t_raw.max(), 1000)
    y_interp = spline(interp_time)

    # 不做噪声，不做温度扰动 —— 使用固定温度 T
    for t, y in zip(interp_time, y_interp):
        features = [t, np.log1p(t), T, T**2, t * T]
        X_train_list.append(features)
        y_train_list.append(y)

X_train = np.array(X_train_list, dtype=float)
y_train = np.array(y_train_list, dtype=float)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(f"✅ Training samples: {len(X_train)}")

# ------------------------------------------------------------
# Step 3: 模型训练与保存
# ------------------------------------------------------------
print("\n[Step 3] Training XGBoost...")

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,         # 平滑数据可适当更深
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model trained.")

MODEL_FILENAME = "crayfish_tvbn_predictor_model.pkl"
joblib.dump(model, MODEL_FILENAME)
print(f"✅ Model saved to '{MODEL_FILENAME}'")

# ------------------------------------------------------------
# Step 4: 在原始采样点上评估
# ------------------------------------------------------------
print("\n[Step 4] Evaluating on original measured time points...")

all_y_true, all_y_pred = [], []
plt.figure(figsize=(14, 8))
plt.title("模型验证：原始数据点上的预测 vs. 真实值（平滑训练版）", fontsize=16)

for i, T in enumerate(temp_values):
    t_pts = time_values_hours[i]
    y_true = amine_values[i]
    X_eval = np.array([[t, np.log1p(t), T, T**2, t*T] for t in t_pts])
    y_pred = model.predict(X_eval)

    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

    plt.plot(t_pts, y_true, 'o-', linewidth=2.5, markersize=8, label=f'真实值 @ {T}°C')
    plt.plot(t_pts, y_pred, 's--', alpha=0.9, label=f'预测值 @ {T}°C')

# 评估指标
r2  = r2_score(all_y_true, all_y_pred)
mse = mean_squared_error(all_y_true, all_y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_y_true, all_y_pred)

print("\n--- Evaluation on Original Points ---")
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print("-------------------------------------")

plt.xlabel("时间 (h)")
plt.ylabel("TVB-N (mg/100g)")
plt.grid(True, linestyle=':')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

print("\n--- Done ---")
