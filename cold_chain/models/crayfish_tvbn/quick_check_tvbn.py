# quick_check_tvbn.py

import time
import pandas as pd
from sqlalchemy import text

# 使用相对路径导入您项目中的模块，以确保能正确找到

from db import database_setup
from app import prediction_logic
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_test_for_single_tracode(tracode_to_test):
    """
    针对单个追溯码运行完整的预测流程。
    """
    print(f"--- 开始为追溯码 [{tracode_to_test}] 进行快速测试 ---")
    start_time = time.time()

    # --- 1. 连接数据库 ---
    print("\n[Step 1] 正在连接数据库...")
    engine = database_setup.connect_to_db()
    if not engine:
        print("❌ 数据库连接失败，测试中止。")
        return
    print("✅ 数据库连接成功。")

    # --- 2. 获取该追溯码对应的监控数据 (monitor_df) ---
    print(f"\n[Step 2] 正在获取追溯码 '{tracode_to_test}' 的所有监控数据...")
    try:
        monitor_sql = text("""
            SELECT * FROM monitoring_informationtable 
            WHERE TraCode = :tracode 
            ORDER BY RecTime ASC
        """)
        monitor_df = pd.read_sql(monitor_sql, engine, params={"tracode": tracode_to_test})

        if monitor_df.empty:
            print(f"❌ 未找到追溯码 '{tracode_to_test}' 的任何监控数据，测试中止。")
            return
        print(f"✅ 成功获取 {len(monitor_df)} 条监控记录。")
    except Exception as e:
        print(f"❌ 查询监控数据时出错: {e}")
        return

    # --- 3. 获取该追溯码对应的食品基础信息 (food_df) ---
    # 我们这里使用一个优化的查询，而不是调用会查询所有食品的 get_food_info
    print(f"\n[Step 3] 正在获取追溯码 '{tracode_to_test}' 的食品基础信息...")
    try:
        food_sql = text("""
            SELECT
                ccfo.TraCode, fsti.*, ccfo.ProDate, ccfo.FoodName
            FROM cold_chain_food_origin ccfo
            JOIN food_knowledge fsti ON ccfo.FoodClassificationCode = fsti.FoodClassificationCode
            WHERE ccfo.TraCode = :tracode
            LIMIT 1
        """)
        food_df = pd.read_sql(food_sql, engine, params={"tracode": tracode_to_test})

        if food_df.empty:
            print(f"❌ 未找到追溯码 '{tracode_to_test}' 对应的食品基础信息，测试中止。")
            return
        print("✅ 成功获取食品基础信息。")
    except Exception as e:
        print(f"❌ 查询食品基础信息时出错: {e}")
        return

    # --- 4. 调用核心处理函数 ---
    print("\n[Step 4] 开始执行 handle_prediction_results 核心预测流程...")
    # 调用您已经完成的、包含所有逻辑的主函数
    prediction_logic.handle_prediction_results(monitor_df, food_df, engine)

    end_time = time.time()
    print(f"\n--- 快速测试完成 ---")
    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒。")


# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == '__main__':
    # --- 在这里输入您想测试的追溯码 ---
    TARGET_TRACODE = ('T2025041715351500000022'
                      '')

    # 运行测试
    run_test_for_single_tracode(TARGET_TRACODE)