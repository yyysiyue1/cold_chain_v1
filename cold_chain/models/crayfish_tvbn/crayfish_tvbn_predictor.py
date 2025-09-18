# -*- coding: utf-8 -*-
"""
crayfish_tvbn_predictor.py
封装 TVBN 模型的加载和预测逻辑（小龙虾）。
- 模型路径：默认加载与本文件同目录下的 pkl；也可传绝对/相对路径。
- 相对导入：优先包内相对导入，兼容脚本式运行。
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional

import numpy as np
import joblib

# 兼容：若文件与本模块同目录
try:
    from .advanced_prediction_models import build_poly_models
except Exception:  # 当作为脚本运行时，可能没有包上下文
    from advanced_prediction_models import build_poly_models  # type: ignore


DEFAULT_MODEL_FILENAME = "crayfish_tvbn_predictor_model.pkl"


class Crayfish_TVBNPredictor:
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        初始化预测器并加载模型。
        :param model_path: 模型文件路径。None 或未传 -> 使用本文件同目录的默认 pkl。
                           传入相对路径时，将相对于本文件目录解析，而不是 CWD。
        """
        base_dir = Path(__file__).resolve().parent  # .../models/crayfish_tvbn
        p = Path(model_path) if model_path else Path(DEFAULT_MODEL_FILENAME)
        # 如果不是绝对路径，则相对于当前文件目录
        abs_path = p if p.is_absolute() else (base_dir / p)

        if not abs_path.is_file():
            raise FileNotFoundError(f"❌ 模型文件未找到: {abs_path}")

        try:
            self.model = joblib.load(str(abs_path))
        except Exception as e:
            raise RuntimeError(f"❌ 加载模型失败: {abs_path}\n原因: {e}") from e

        self._poly_models_cache: Dict[Tuple[Tuple[float, ...], int, int], Any] = {}
        print(f"✅ 模型已成功加载: {abs_path}")

    # --------------------------
    # 特征工程
    # --------------------------
    @staticmethod
    def make_features(time_h: float, temp_c: float) -> np.ndarray:
        """
        根据时间(h)与温度(°C)构建特征。
        """
        t = float(time_h)
        c = float(temp_c)
        feats = np.array(
            [[
                t,
                np.log1p(t),
                c,
                c ** 2,
                t * c
            ]],
            dtype=float
        )
        return feats

    # --------------------------
    # 推理接口
    # --------------------------
    def predict(self, time_h: float, temp_c: float) -> float:
        """
        单点预测 TVBN 含量 (mg/100g)。
        """
        features = self.make_features(time_h, temp_c)
        y = self.model.predict(features)
        return float(np.asarray(y).ravel()[0])

    def predict_batch(self, times_h: Iterable[float], temps_c: Iterable[float]) -> np.ndarray:
        """
        批量预测：times 与 temps 长度需一致。
        返回 shape (N,) 的 numpy 数组。
        """
        times = list(times_h)
        temps = list(temps_c)
        if len(times) != len(temps):
            raise ValueError("times_h 与 temps_c 长度必须一致")
        X = np.vstack([self.make_features(t, c) for t, c in zip(times, temps, strict=False)])
        y = self.model.predict(X)
        return np.asarray(y, dtype=float).ravel()

    # --------------------------
    # 多项式近似（用于绘曲线/搜索）
    # --------------------------
    def get_poly_models(self, temps: Iterable[float], time_max: int = 2000, poly_order: int = 3):
        """
        获取或构建多项式拟合曲线，避免重复计算。
        """
        temps_tuple = tuple(sorted(float(x) for x in temps))
        key = (temps_tuple, int(time_max), int(poly_order))
        if key not in self._poly_models_cache:
            print(f"⚡ 缓存未命中，重建多项式：temps={temps_tuple}, time_max={time_max}, order={poly_order}")
            self._poly_models_cache[key] = build_poly_models(self.model, temps_tuple, time_max, poly_order)
        else:
            print(f"⚡ 缓存命中：temps={temps_tuple}, time_max={time_max}, order={poly_order}")
        return self._poly_models_cache[key]

    # --------------------------
    # 友好显示
    # --------------------------
    def __repr__(self) -> str:
        return f"TVBNPredictor(model=<{self.model.__class__.__name__}>)"
