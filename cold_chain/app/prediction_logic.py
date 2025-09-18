import pandas as pd # Although not used in this specific file, might be used with the engine later

from datetime import datetime  # Not directly used here
import numpy as np # Not directly used here
from typing import Optional, Tuple, Dict, Callable, Union
import math
from sqlalchemy import  text # 确保导入 text

from db.database_setup import prediction_table_name
from models.crayfish_tvbn.crayfish_tvbn_predictor import Crayfish_TVBNPredictor
from models.crayfish_tvbn.advanced_prediction_models import calculate_dynamic_tvbn_value
# ==============================================================================
# --- 预测逻辑函数  ---
# ==============================================================================

# 从食品知识表中获得食品信息（保质期、温度上下限、食品分类代码）
# ******** 开始:预测逻辑函数 ********
def get_food_info(engine):
    """
    从数据库查询冷链食品信息,包括温度/湿度上下限、保质期、生产日期等。
    返回: DataFrame
    """
    query = """
    SELECT
        mit.TraCode,
        fsti.StorTempUpper,
        fsti.StorTempLower,
        fsti.StorHumidUpper,
        fsti.StorHumidLower,
        fsti.ShelfLife,
        fsti.FoodClassificationCode,
        fsti.SecondaryClassificationName,
        fsti.TempAnonalyDuration,
        fsti.HumidAnonalyDuration,
        ccfo.ProDate,
        ccfo.FoodName
    FROM monitoring_informationtable mit
    JOIN cold_chain_food_origin ccfo ON mit.TraCode = ccfo.TraCode
    JOIN food_knowledge fsti ON ccfo.FoodClassificationCode = fsti.FoodClassificationCode
    GROUP BY mit.TraCode, fsti.StorTempUpper, fsti.StorTempLower, fsti.FoodClassificationCode,
             fsti.SecondaryClassificationName, fsti.StorHumidUpper, fsti.StorHumidLower,
             fsti.ShelfLife, ccfo.ProDate, fsti.TempAnonalyDuration, fsti.HumidAnonalyDuration, ccfo.FoodName;
    """
    try:
        food_df = pd.read_sql(query, engine)
        food_df["ProDate"] = pd.to_datetime(food_df["ProDate"], errors="coerce")
        numeric_cols = ["StorTempUpper","StorTempLower","StorHumidUpper","StorHumidLower",
                        "ShelfLife","TempAnonalyDuration","HumidAnonalyDuration"]
        for c in numeric_cols:
            food_df[c] = pd.to_numeric(food_df[c], errors="coerce")
        return food_df
    except Exception as e:
        print(f"❌ 获取或处理食品信息时出错:{e}")
        return pd.DataFrame()



# 获取订单路径信息
def get_order_tra_chain(order_number, tra_code, engine):
    if pd.isna(order_number) or pd.isna(tra_code):
        return None
    sql = text("""
        SELECT OrderTraChain
        FROM order_info
        WHERE OrderNumber = :order_number AND TraCode = :tra_code
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={"order_number": str(order_number), "tra_code": str(tra_code)})
        return df.iloc[0]['OrderTraChain'] if not df.empty else None
    except Exception as e:
        print(f"❌ 获取订单路径链时出错 (Order:{order_number}, TraCode:{tra_code}):{e}")
        return None



# 查找路径中的上一条订单
def find_previous_order(order_number, order_tra_chain):
    """查找路径链中的上一个订单号"""
    if not order_tra_chain or pd.isna(order_number):  # 增加 pd.isna 检查
        return None

    order_list = order_tra_chain.split(",")
    # 检查 order_number 是否在列表中 (代码没有此检查,但建议保留)
    if order_number not in order_list:
        # print(f"⚠️ 订单 {order_number} 不在路径链 {order_tra_chain} 中。")
        return None

    try:  # 保留错误处理
        index = order_list.index(order_number)
        if index == 0:
            return False  # 是第一个订单

        previous_order = order_list[index - 1]
        # print(f"上一个元素：{previous_order}")
        return previous_order if previous_order.startswith("O") else False  # 上一个是订单号就返回,否则返回 False
    except ValueError:  # 如果 index() 失败
        return None
    except Exception as e:
        print(f"❌ 查找上一个订单时出错 ({order_number} in {order_tra_chain}):{e}")
        return None


# 判断数据是否是该订单上第一条数据
def is_first_record_simple(order_number, tra_code, rec_time, engine):
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return False
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return False
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text("""
        SELECT 1
        FROM monitoring_informationtable
        WHERE OrderNumber = :order_number
          AND TraCode = :tra_code
          AND RecTime < :rec_time
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "order_number": str(order_number),
            "tra_code": str(tra_code),
            "rec_time": rec_time_str
        })
        return df.empty
    except Exception as e:
        print(f"❌ 检查首条记录时出错 (Order:{order_number}, TraCode:{tra_code}):{e}")
        return False


#查找追溯码对应的储运时间函数
def get_storage_time(tra_code, rec_time, engine):
    """
    查找指定追溯码在 cold_chain_food_origin 表中的储运时间 (StorageTime)。

    参数:
        tra_code (str): 追溯码。
        rec_time (datetime): 当前记录时间，用于验证储运时间不是未来的时间。
        engine: SQLAlchemy 数据库连接引擎。

    返回:
        datetime or None: 找到且有效的储运时间，否则返回 None。
    """
    if pd.isna(tra_code):
        return None

    try:
        # 使用参数化查询以防止SQL注入
        storage_sql = text("""
            SELECT StorageTime
            FROM cold_chain_food_origin
            WHERE TraCode = :tra_code
            LIMIT 1
        """)
        storage_df = pd.read_sql(storage_sql, engine, params={"tra_code": tra_code})

        if not storage_df.empty and pd.notna(storage_df.iloc[0]['StorageTime']):
            storage_time = pd.to_datetime(storage_df.iloc[0]['StorageTime'], errors='coerce')
            # 确保 StorageTime 早于或等于当前记录时间
            if pd.notna(storage_time) and storage_time <= rec_time:
                return storage_time
    except Exception as e:
        print(f"❌ 查询 StorageTime 时出错 (TraCode: {tra_code}): {e}")

    return None # 其他所有情况都返回 None
# 首站查相邻监测数据
def get_previous_first_station_data(tra_code, rec_time, engine):
    """
    获取某追溯码下、无订单号的监测数据中,时间早于当前时间的最近一条（首站数据）的监测时间。
    如果找不到,则回退查询 cold_chain_food_origin 表的 StorageTime。
    (注意：代码未包含 CreateTime 或 ProDate 回退)

    返回：
    - datetime,如果找到
    - None,如果都没找到
    """
    # 保留对 None 或 NaN 的检查
    if pd.isna(tra_code) or pd.isna(rec_time):
        return None

    # 确保 rec_time 是 datetime 对象 (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time):return None

    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text(f"""
        SELECT RecTime
        FROM monitoring_informationtable
        WHERE TraCode = :tra_code
          AND (OrderNumber IS NULL OR OrderNumber = '')
          AND RecTime < :rec_time
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={"tra_code": tra_code, "rec_time": rec_time_str})
        if not df.empty and pd.notna(df.iloc[0]['RecTime']):
            return pd.to_datetime(df.iloc[0]['RecTime'], errors='coerce')
        else:
            # 如果在监控表中找不到，则调用新函数作为回退
            return get_storage_time(tra_code, rec_time, engine)
    except Exception as e:
        print(f"❌ 查询首站监控数据时出错 (TraCode: {tra_code}): {e}")
        # 如果查询监控表出错，仍然尝试获取储运时间
        return get_storage_time(tra_code, rec_time, engine)


# 同一订单中查找当前监测记录再路径上的上一条监测记录
def get_previous_monitor_record(order_number, tra_code, rec_time, engine):
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return None
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return None
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text("""
        SELECT RecTime
        FROM monitoring_informationtable
        WHERE OrderNumber = :order_number
          AND TraCode = :tra_code
          AND RecTime < :rec_time
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "order_number": str(order_number),
            "tra_code": str(tra_code),
            "rec_time": rec_time_str
        })
        return pd.to_datetime(df.iloc[0]['RecTime'], errors='coerce') if not df.empty else None
    except Exception as e:
        print(f"❌ 获取同一订单上一条记录时出错 (Order:{order_number}, TraCode:{tra_code}):{e}")
        return None



# 查找首站预测温度/湿度异常值
def get_previous_first_station_predict_value(tra_code, rec_time, engine, flag="温度"):
    if pd.isna(tra_code) or pd.isna(rec_time):
        return 0.0
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return 0.0
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text(f"""
        SELECT PredictValue
        FROM {prediction_table_name}
        WHERE TraCode = :tra_code
          AND (OrderNumber IS NULL OR OrderNumber = '')
          AND RecTime < :rec_time
          AND PredictFlag = :flag
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "tra_code": str(tra_code),
            "rec_time": rec_time_str,
            "flag": flag
        })
        val = df.iloc[0]['PredictValue'] if not df.empty else 0.0
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0
    except Exception as e:
        print(f"❌ 获取首站先前预测值时出错 (TraCode:{tra_code}, Flag:{flag}):{e}")
        return 0.0



# 同一订单中查找当前监测记录再路径上的上一条监测记录的异常值
def get_previous_monitor_predict_value(order_number, tra_code, rec_time, engine, flag="温度"):
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return 0.0
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return 0.0
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text(f"""
        SELECT PredictValue
        FROM {prediction_table_name}
        WHERE OrderNumber = :order_number
          AND TraCode = :tra_code
          AND RecTime < :rec_time
          AND PredictFlag = :flag
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "order_number": str(order_number),
            "tra_code": str(tra_code),
            "rec_time": rec_time_str,
            "flag": flag
        })
        val = df.iloc[0]['PredictValue'] if not df.empty else 0.0
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0
    except Exception as e:
        print(f"❌ 获取同一订单先前预测值时出错 (Order:{order_number}, TraCode:{tra_code}, Flag:{flag}):{e}")
        return 0.0



# 查找上一条监测数据的时间
def find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine):
    """
    查找上一条监测数据的时间。
    """
    # 处理 NaN 或 None 的 order_number (保留优化)
    current_order = None if pd.isna(order_number) else order_number

    if current_order is None:
        # 首站数据
        return get_previous_first_station_data(tra_code, rec_time, engine)

    # 确保 rec_time 是 datetime (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time): return None  # 无效时间无法判断

    if is_first_record_simple(current_order, tra_code, rec_time, engine):
        # 是订单上的第一条
        previous_order = find_previous_order(current_order, order_tra_chain)
        if previous_order:
            # 往前递归找 (代码的递归调用)
            # 注意：递归可能未正确处理 order_tra_chain 的传递,这里保持原样
            return find_previous_monitor_time(previous_order, rec_time, tra_code, order_tra_chain, engine)
        else:  # previous_order is False or None
            # 已经到路径最前面了,去首站找
            return get_previous_first_station_data(tra_code, rec_time, engine)
    else:
        # 正常情况,直接在当前订单上查上一条
        return get_previous_monitor_record(current_order, tra_code, rec_time, engine)


# 查找上一条监测记录的异常时长
def find_previous_abnormal_value(order_number, rec_time, tra_code, order_tra_chain, engine, flag="温度"):
    """
    查找上一条预测值（支持温度/湿度）。
    """
    # 处理 NaN 或 None 的 order_number (保留优化)
    current_order = None if pd.isna(order_number) else order_number

    if current_order is None:
        return get_previous_first_station_predict_value(tra_code, rec_time, engine, flag)

    # 确保 rec_time 是 datetime (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time): return 0.0

    if is_first_record_simple(current_order, tra_code, rec_time, engine):
        previous_order = find_previous_order(current_order, order_tra_chain)
        if previous_order:
            # 往前递归找 (代码的递归调用)
            return find_previous_abnormal_value(previous_order, rec_time, tra_code, order_tra_chain, engine, flag)
        else:  # previous_order is False or None
            return get_previous_first_station_predict_value(tra_code, rec_time, engine, flag)
    else:
        return get_previous_monitor_predict_value(current_order, tra_code, rec_time, engine, flag)


# 计算温度异常时长
def calculate_temp_predict_value(temp_abnormal, order_number, rec_time, tra_code, order_tra_chain, engine):
    """
    根据是否温度异常计算 temp_predict_value。
    """
    # 确保 rec_time 是 datetime (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time): return 0.0  # Or handle error appropriately

    # 获取上一个时间点和上一个预测值 (使用函数调用)
    last_monitor_time = find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine)
    last_abnormal_value = find_previous_abnormal_value(order_number, rec_time, tra_code, order_tra_chain, engine,
                                                       flag="温度")

    if temp_abnormal:
        # 检查 last_monitor_time 是否有效 (增加检查)
        if last_monitor_time is None or pd.isna(last_monitor_time):
            # print(f"⚠️ 无法找到先前监控时间 (温度计算):order={order_number}, tra={tra_code}, time={rec_time}")
            # 如果是异常,但找不到上一条,代码会出错。这里返回上一个值（通常为0）
            return last_abnormal_value
        # 计算时间差
        try:  # 增加错误处理
            time_delta_seconds = (rec_time - last_monitor_time).total_seconds()
            if time_delta_seconds < 0:
                print(
                    f"⚠️ 计算出负时间差 (温度):{time_delta_seconds}s. Order={order_number}, Tra={tra_code}. 使用0增量。")
                time_delta_minutes = 0.0
            else:
                time_delta_minutes = time_delta_seconds / 60.0
            temp_predict_value = time_delta_minutes + last_abnormal_value
        except TypeError as e:
            print(f"❌ 计算温度时间差时出错:{e}. RecTime={rec_time}, LastTime={last_monitor_time}")
            temp_predict_value = last_abnormal_value  # 出错时返回上一个值
    else:
        # 代码直接获取上一个值
        temp_predict_value = last_abnormal_value

    # 确保返回数值 (增加处理)
    return float(temp_predict_value) if pd.notna(temp_predict_value) else 0.0


# 计算湿度异常时长
def calculate_humidity_predict_value(humidity_abnormal, order_number, rec_time, tra_code, order_tra_chain, engine):
    """
    根据是否湿度异常计算 humidity_predict_value。
    """
    # 确保 rec_time 是 datetime (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time): return 0.0  # Or handle error appropriately

    # 获取上一个时间点和上一个预测值 (使用函数调用)
    last_monitor_time = find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine)
    last_abnormal_value = find_previous_abnormal_value(order_number, rec_time, tra_code, order_tra_chain, engine,
                                                       flag="湿度")

    if humidity_abnormal:
        # 检查 last_monitor_time 是否有效 (增加检查)
        if last_monitor_time is None or pd.isna(last_monitor_time):
            # print(f"⚠️ 无法找到先前监控时间 (湿度计算):order={order_number}, tra={tra_code}, time={rec_time}")
            return last_abnormal_value
        # 计算时间差
        try:  # 增加错误处理
            time_delta_seconds = (rec_time - last_monitor_time).total_seconds()
            if time_delta_seconds < 0:
                print(
                    f"⚠️ 计算出负时间差 (湿度):{time_delta_seconds}s. Order={order_number}, Tra={tra_code}. 使用0增量。")
                time_delta_minutes = 0.0
            else:
                time_delta_minutes = time_delta_seconds / 60.0
            humidity_predict_value = time_delta_minutes + last_abnormal_value
        except TypeError as e:
            print(f"❌ 计算湿度时间差时出错:{e}. RecTime={rec_time}, LastTime={last_monitor_time}")
            humidity_predict_value = last_abnormal_value  # 出错时返回上一个值
    else:
        # 代码直接获取上一个值
        humidity_predict_value = last_abnormal_value

    # 确保返回数值 (增加处理)
    return float(humidity_predict_value) if pd.notna(humidity_predict_value) else 0.0


# 温度异常判断函数
def is_temp_abnormal(temp, lower, upper):
    """
    判断温度是否异常,异常条件：
    - 小于下限 或 大于上限（只要下限/上限存在）

    返回：
        True：异常
        False：正常或无法判断
    """
    # 代码的 try-except 逻辑
    try:
        # 增加对输入是否为 None 或 NaN 的检查 (保留优化)
        if pd.isna(temp): return False
        temp_f = float(temp)
        lower_f = float(lower) if pd.notna(lower) else None
        upper_f = float(upper) if pd.notna(upper) else None
    except Exception:  # 代码捕获所有 Exception
        return False  # 无法转换则视为正常（不异常）

    if lower_f is not None and temp_f < lower_f:
        return True
    if upper_f is not None and temp_f > upper_f:
        return True
    return False


# 湿度上下限解析
def parse_humidity_range(humid_str):
    """
    解析湿度范围字符串,比如 '90-95',返回 (90, 95)。
    如果只有一个值,比如 '90',返回 (90, None)。
    """
    if pd.isna(humid_str):
        return None, None
    try:
        parts = str(humid_str).split("-")
        if len(parts) == 2:
            lower = float(parts[0])
            upper = float(parts[1])
            return lower, upper
        elif len(parts) == 1:
            lower = float(parts[0])
            return lower, None  # 代码这里返回 (lower, None)
    except Exception:
        pass  # 代码忽略异常
    return None, None


# 湿度异常判断函数
def is_humidity_abnormal(humidity, lower, upper):
    """
    判断湿度是否异常,异常条件：
    - 小于下限 或 大于上限（只要下限/上限存在）

    返回：
        True：异常
        False：正常或无法判断
    """
    # 代码的 try-except 逻辑
    try:
        # 增加对输入是否为 None 或 NaN 的检查 (保留优化)
        if pd.isna(humidity): return False
        humidity_f = float(humidity)
        lower_f = float(lower) if pd.notna(lower) else None
        upper_f = float(upper) if pd.notna(upper) else None
    except Exception:  # 代码捕获所有 Exception
        return False  # 无法转换则视为正常（不异常）

    if lower_f is not None and humidity_f < lower_f:
        return True
    if upper_f is not None and humidity_f > upper_f:
        return True
    return False


# 温度风险等级判断
def is_exceeding_temp_threshold(food_class_code, temp_predict_value, engine, temp_abnormal=True):  # 签名包含 temp_abnormal
    """
    判断温度异常时长是否超过对应食品分类的异常阈值。
    """
    if pd.isna(food_class_code) or pd.isna(temp_predict_value):  # 检查
        return False

    # 如果传入了已判断的 temp_abnormal 且为 False,则直接返回 False
    if not temp_abnormal:
        return False

    # 增加检查,如果累积时长为0或负,则不可能超阈值 (保留优化)
    if temp_predict_value <= 0:
        return False

    # 从 food_knowledge 获取阈值 (需要确保 food_knowledge 表和 TempAnonalyDuration 列存在)
    threshold_hours = None
    try:
        query = """
        SELECT TempAnonalyDuration
        FROM food_knowledge
        WHERE FoodClassificationCode = :food_class_code
        LIMIT 1;
        """
        with engine.connect() as conn:
            result = conn.execute(text(query), {"food_class_code": food_class_code}).scalar()  # 使用 scalar 获取单个值
            if result is not None and pd.notna(result):
                threshold_hours = float(result)
    except Exception as e:
        print(f"❌ 查询温度阈值失败：{e}")
        return False  # 查询失败视为不超

    if threshold_hours is None or threshold_hours <= 0:
        # print(f"ℹ️ 未找到有效的食品分类 {food_class_code} 的温度阈值。")
        return False  # 没有阈值或阈值无效,视为不超

    threshold_minutes = threshold_hours * 60
    return temp_predict_value > threshold_minutes


# 湿度风险等级判断
def is_exceeding_humid_threshold(food_class_code, humid_predict_value, engine,
                                 humid_abnormal=True):  # 签名包含 humid_abnormal
    """
    判断湿度异常时长是否超过对应食品分类的异常阈值。
    """
    if pd.isna(food_class_code) or pd.isna(humid_predict_value):  # 检查
        return False

    # 如果传入了已判断的 humid_abnormal 且为 False,则直接返回 False
    if not humid_abnormal:
        return False

    # 增加检查,如果累积时长为0或负,则不可能超阈值 (保留优化)
    if humid_predict_value <= 0:
        return False

    # 从 food_knowledge 获取阈值 (需要确保 food_knowledge 表和 HumidAnonalyDuration 列存在)
    threshold_hours = None
    try:
        query = """
        SELECT HumidAnonalyDuration
        FROM food_knowledge
        WHERE FoodClassificationCode = :food_class_code
        LIMIT 1;
        """
        with engine.connect() as conn:
            result = conn.execute(text(query), {"food_class_code": food_class_code}).scalar()
            if result is not None and pd.notna(result):
                threshold_hours = float(result)
    except Exception as e:
        print(f"❌ 查询湿度阈值失败：{e}")
        return False  # 查询失败视为不超

    if threshold_hours is None or threshold_hours <= 0:
        # print(f"ℹ️ 未找到有效的食品分类 {food_class_code} 的湿度阈值。")
        return False  # 没有阈值或阈值无效,视为不超

    threshold_minutes = threshold_hours * 60
    return humid_predict_value > threshold_minutes


# 过期异常判断
def is_shelf_life_abnormal(pro_date, rec_time, shelf_life):
    """
    判断是否超过保质期（单位：天）。
    """
    try:
        # 检查
        if pd.isna(pro_date) or pd.isna(rec_time) or pd.isna(shelf_life):
            return False

        # 转换成标准格式
        pro_date_dt = pd.to_datetime(pro_date, errors='coerce')
        rec_time_dt = pd.to_datetime(rec_time, errors='coerce')
        # 代码直接用 int(),如果 shelf_life 不能转为 int 会报错
        try:
            shelf_life_days = int(shelf_life)
        except (ValueError, TypeError):
            print(f"⚠️ 无法将保质期转换为整数:{shelf_life}")
            return False  # 无法转换视为未过期

        # 检查转换后的值
        if pd.isna(pro_date_dt) or pd.isna(rec_time_dt):
            return False

        days_elapsed = (rec_time_dt - pro_date_dt).days - shelf_life_days
        return days_elapsed > 0

    except Exception as e:
        print(f"❌ 保质期判断出错:pro_date={pro_date}, rec_time={rec_time}, shelf_life={shelf_life},错误：{e}")  # 打印信息
        return False  # 代码在异常时返回 False


# 过期异常时长计算
def calculate_shelf_life_abnormal_duration(pro_date, rec_time, shelf_life):
    """
    计算过期异常时长（单位：天）。即监测日期距离生产日期的天数再减去保质期。
    """
    try:
        # 检查
        if pd.isna(pro_date) or pd.isna(rec_time):
            return None

        # 转换成标准格式
        pro_date_dt = pd.to_datetime(pro_date, errors='coerce')
        rec_time_dt = pd.to_datetime(rec_time, errors='coerce')
        shelf_life_days_int = None
        if not pd.isna(shelf_life):
            try:
                shelf_life_days_int = int(shelf_life)
            except (ValueError, TypeError):
                print(f"⚠️ 无法将保质期 '{shelf_life}' 转换为整数天数。")

        # 检查转换后的值
        if pd.isna(pro_date_dt) or pd.isna(rec_time_dt):
            return None

        # 计算天数
        days_elapsed = (rec_time_dt - pro_date_dt).days - shelf_life_days_int
        # 代码直接返回 days_elapsed,可能为负数。保留为非负数。
        return days_elapsed

    except Exception as e:
        print(f"❌ 计算过期异常时长失败：{e}")  # 打印信息
        return None  # 代码在异常时返回 None



# ==============================================================================
# --- 统一调度预测模型  ---
# ==============================================================================
def execute_prediction_unit(row, food_info, engine, predictor_cache):
    """
    统一预测调度单元：根据食品分类执行专属模型预测。
    predictor_cache 形如：{'tvbn': TVBNPredictor(...)}。
    返回：list[dict]，可能为空。
    """
    food_class_code = food_info.get("FoodClassificationCode")
    records = []

    if food_class_code == 'C09006':  # 小龙虾
        tvbn_predictor = predictor_cache.get('crayfish_tvbn')
        if tvbn_predictor and getattr(tvbn_predictor, 'model', None) is not None:
            rec_time = row.get("RecTime")
            tra_code = row.get("TraCode")
            order_number = row.get("OrderNumber")

            # 获取运输链上下文
            order_tra_chain = get_order_tra_chain(order_number, tra_code, engine)

            # 获取最近一次的异常值和监测时间
            last_abnormal_value = find_previous_abnormal_value(
                order_number, rec_time, tra_code, order_tra_chain, engine, flag="化学"
            )
            last_monitor_time = find_previous_monitor_time(
                order_number, rec_time, tra_code, order_tra_chain, engine
            )

            # 初始值兜底：如果没找到，默认 8.0 mg/100g
            if last_abnormal_value is None or pd.isna(last_abnormal_value) or last_abnormal_value == 0.0:
                last_abnormal_value = 8.0

            # 兜底监测时间：如果没找到，取冷链入库时间或生产日期
            if not last_monitor_time:
                last_monitor_time = get_storage_time(tra_code, rec_time, engine) or food_info.get("ProDate")
                if not last_monitor_time:  # 如果还是 None
                    last_monitor_time = pd.to_datetime(rec_time) - pd.Timedelta(hours=1)

            # 统一格式为 pandas 时间戳
            if last_monitor_time:
                last_monitor_time = pd.to_datetime(last_monitor_time)

            # === 调用多项式动态预测 ===
            if pd.notna(last_monitor_time) and pd.notna(rec_time):
                predict_value = calculate_dynamic_tvbn_value(
                    last_abnormal_value=last_abnormal_value,
                    last_monitor_time=last_monitor_time,
                    rec_time=rec_time,
                    current_temp=row.get("Temp"),
                    predictor=tvbn_predictor  # ✅ 改成直接传 predictor
                )
            else:
                predict_value = last_abnormal_value  # 如果时间无效，直接返回上一个值
            # 生成存储记录
            tvbn_record = generate_tvbn_record(row, food_info, predict_value, engine)  # flag 默认“化学”

            if tvbn_record:
                records.append(tvbn_record)
    # 其它分类可以在这里扩展
    # TODO: 十四胺/菌落总数

    return records




# ==============================================================================
# ---标志物 风险等级判断---
# ==============================================================================
# flag -> 表名（同时兼容中文/英文入参）
_TABLE_MAP = {
    "chem": "chemicalmarker_risk_level",
    "化学": "chemicalmarker_risk_level",
    "bio":  "biomarker_risk_level",
    "生物":  "biomarker_risk_level",
}
_NEEDED_COLS = ["FoodClassificationCode", "LowRiskLine", "MiddleRiskLine", "HighRiskLine"]
# 惰性缓存：key=规范化flag('chem'/'bio') -> DataFrame(index=FoodClassificationCode)
_threshold_cache: Dict[str, pd.DataFrame] = {}
#加载阈值缓存
def load_thresholds(
    engine,
    flag: str,
    refresh: bool = False,
    read_sql: Callable[[str, Union[object, None]], pd.DataFrame] = pd.read_sql,
) -> pd.DataFrame:
    """
    从数据库把对应表的阈值读入缓存；返回 DataFrame（index=FoodClassificationCode）
    """
    nflag = _norm_flag(flag)
    if (not refresh) and (nflag in _threshold_cache):
        return _threshold_cache[nflag]

    table = _TABLE_MAP[nflag]
    sql = f"SELECT {', '.join(_NEEDED_COLS)} FROM {table}"
    df = read_sql(sql, engine).copy()
    df = df[_NEEDED_COLS].drop_duplicates(subset=["FoodClassificationCode"])
    df.set_index("FoodClassificationCode", inplace=True)
    for c in ("LowRiskLine", "MiddleRiskLine", "HighRiskLine"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _threshold_cache[nflag] = df
    return df
def _norm_flag(flag: str) -> str:
    if flag in ("chem", "化学"):
        return "chem"
    if flag in ("bio", "生物"):
        return "bio"
    raise ValueError(f"flag 只能是 'chem'/'化学' 或 'bio'/'生物'，收到：{flag}")

def get_thresholds(
    food_code: str,
    flag: str,
    engine,
    refresh: bool = False,
    read_sql: Callable[[str, Union[object, None]], pd.DataFrame] = pd.read_sql,
) -> Optional[Tuple[float, float, float]]:
    """
    返回 (low, mid, high)；没有或缺失则返回 None
    """
    df = load_thresholds(engine, flag, refresh=refresh, read_sql=read_sql)
    if food_code not in df.index:
        return None
    row = df.loc[food_code]
    low, mid, high = row["LowRiskLine"], row["MiddleRiskLine"], row["HighRiskLine"]
    if pd.isna(low) or pd.isna(mid) or pd.isna(high):
        return None
    return float(low), float(mid), float(high)

def determine_risk_level(
    food_code: str,
    value: Optional[float],
    flag: str,
    engine,
    *,
    return_detail: bool = False,
    refresh: bool = False,
    read_sql: Callable[[str, Union[object, None]], pd.DataFrame] = pd.read_sql,
):
    """
    通用风险等级判断（适用于化学/生物的任意标志物）：
    规则：value < low → '无'；[low, mid) → '低'；[mid, high) → '中'；>= high → '高'
    value 为 None 返回 '无'；找不到阈值返回 '未知'
    """
    if value is None:
        return ("无", None) if return_detail else "无"

    th = get_thresholds(food_code, flag, engine, refresh=refresh, read_sql=read_sql)
    if th is None:
        return ("未知", None) if return_detail else "未知"

    low, mid, high = th
    if value < low:
        lvl = "无"
    elif value < mid:
        lvl = "低"
    elif value < high:
        lvl = "中"
    else:
        lvl = "高"
    return (lvl, (low, mid, high)) if return_detail else lvl


# ==============================================================================
# --- 生成TVB-N预测记录  ---
# ==============================================================================
def generate_tvbn_record(row, food_info, tvbn_predict_value, engine, flag='化学'):
    """
    【TVB-N记录生成函数】
    根据计算出的TVB-N含量值，生成要插入数据库的记录字典。
    直接使用通用 determine_risk_level 从数据库阈值表判级。
    """
    monitor_num = row.get('MonitorNum')
    if monitor_num is None:
        return None

    food_code = food_info.get("FoodClassificationCode")
    risk_level = determine_risk_level(food_code, tvbn_predict_value, flag, engine)

    return {
        "PredictResultID": f"{monitor_num}11",
        "MonitorNum": monitor_num,
        "OrderNumber": row.get("OrderNumber"),
        "TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),
        "FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),
        "Humid": row.get("Humid"),
        "PredictFlag": flag,                 # 默认“化学”，也可传入“生物”
        "RiskName": "挥发性盐基氮",
        "PredictValue": tvbn_predict_value,
        "Unit": "mg/100g",
        "RiskLevel": risk_level
    }



# 生成温度预测记录
def generate_temperature_abnormal_record(row, food_info, temp_predict_value,
                                         risk_temp_abnormal):  # 签名包含 risk_temp_abnormal
    """生成温度预测记录字典"""
    monitor_num = row.get('MonitorNum')

    if monitor_num is None: return None
    #  RiskLevel 判断
    risk_level = "低" if risk_temp_abnormal else "无"

    stotemplower_val = food_info.get("StorTempLower")  # This is a float or None
    stotempupper_val = food_info.get("StorTempUpper")  # This is a float or None
    # Process lower limit
    if stotemplower_val is None or (isinstance(stotemplower_val, float) and math.isnan(stotemplower_val)):
        display_lower = "    "
    else:
        display_lower = str(stotemplower_val)  # Converts float to string, e.g., 10.0 -> "10.0"
    # Process upper limit
    if stotempupper_val is None or (isinstance(stotempupper_val, float) and math.isnan(stotempupper_val)):
        display_upper = "    "
    else:
        display_upper = str(stotempupper_val)

    return {
        "PredictResultID": f"{monitor_num}02",  # 保留后缀
        "MonitorNum": monitor_num,
        "OrderNumber": row.get("OrderNumber"),
        "TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),
        "FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),
        "Humid": row.get("Humid"),
        "PredictFlag": "温度",
        "RiskName": f"{display_lower}℃~{display_upper}℃",
        "PredictValue": temp_predict_value,
        "Unit": "min",
        "RiskLevel": risk_level  # 使用判断逻辑
    }


# 生成湿度预测记录
def generate_humidity_abnormal_record(row, food_info, humidity_predict_value,
                                      risk_humid_abnormal):  # 签名包含 risk_humid_abnormal
    """生成湿度预测记录字典"""
    monitor_num = row.get('MonitorNum')

    if monitor_num is None: return None
    #  RiskLevel 判断
    risk_level = "低" if risk_humid_abnormal else "无"

    stohumidupper_val = food_info.get("StorHumidUpper")  # This is a float or None
    stohumidlower_val = food_info.get("StorHumidLower")  # This is a float or None
    # Process lower limit
    if stohumidlower_val is None or (isinstance(stohumidlower_val, float) and math.isnan(stohumidlower_val)):
        display_lower = "    "
    else:
        display_lower = str(stohumidlower_val)  # Converts float to string, e.g., 10.0 -> "10.0"
    # Process upper limit
    if stohumidupper_val is None or (isinstance(stohumidupper_val, float) and math.isnan(stohumidupper_val)):
        display_upper = "    "
    else:
        display_upper = str(stohumidupper_val)

    return {
        "PredictResultID": f"{monitor_num}03",  # 保留后缀
        "MonitorNum": monitor_num,
        "OrderNumber": row.get("OrderNumber"),
        "TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),
        "FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),
        "Humid": row.get("Humid"),
        "PredictFlag": "湿度",
        "RiskName": f"{display_lower}%RH~{display_upper}%RH",
        "PredictValue": humidity_predict_value,
        "Unit": "min",
        "RiskLevel": risk_level  # 使用判断逻辑
    }


# 生成过期预测记录
def generate_shelf_life_abnormal_record(row, food_info, shelf_predict_value,
                                        shelf_life_abnormal):  # 签名包含 shelf_life_abnormal
    """生成过期预测记录字典"""
    monitor_num = row.get('MonitorNum')
    shelflife = food_info.get("ShelfLife")
    if monitor_num is None: return None
    #  RiskLevel 判断
    risk_level = "高" if shelf_life_abnormal else "无"

    return {
        "PredictResultID": f"{monitor_num}01",  # 保留后缀
        "MonitorNum": monitor_num,
        "OrderNumber": row.get("OrderNumber"),
        "TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),
        "FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),
        "Humid": row.get("Humid"),
        "PredictFlag": "过期",
        "RiskName": shelflife,
        "PredictValue": shelf_predict_value,
        "Unit": "天",
        "RiskLevel": risk_level  # 使用判断逻辑
    }


# 处理none函数
def replace_nan_with_none(d):
    # 实现只处理 float NaN
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


# 将异常记录插入预测结果表
def insert_results(results, engine):
    """将预测结果列表插入或更新到 risk_prediction_results 表 """
    if not results:
        # print("⚠️ 当前批次无数据可入库") # 打印信息
        return []  # 返回空列表

    # 过滤掉 None 值 (如果 generate 函数返回 None) (保留优化)
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        # print("⚠️ 当前批次无有效数据可入库")
        return []

    if not isinstance(valid_results, list) or not all(isinstance(item, dict) for item in valid_results):
        # 代码会直接抛出 ValueError
        raise ValueError("❌ insert_results 接收的是一个由 dict 构成的列表,例如：[record1, record2]")

    result_df = pd.DataFrame(valid_results)

    # 代码没有预处理和检查 DataFrame

    insert_sql = f"""
    INSERT INTO {prediction_table_name} (
        PredictResultID, MonitorNum, OrderNumber, TraCode, RecTime,
        FoodClassificationCode, Temp, Humid, PredictFlag, RiskName,
        PredictValue, Unit, RiskLevel
    ) VALUES (
        :PredictResultID, :MonitorNum, :OrderNumber, :TraCode, :RecTime,
        :FoodClassificationCode, :Temp, :Humid, :PredictFlag, :RiskName,
        :PredictValue, :Unit, :RiskLevel
    )
    ON DUPLICATE KEY UPDATE
        PredictValue = VALUES(PredictValue),
        -- RiskLevel = VALUES(RiskLevel), # 代码没有更新 RiskLevel
        RecTime = VALUES(RecTime);
        -- 代码只更新 PredictValue 和 RecTime
    """

    # 代码使用 replace_nan_with_none
    insert_data_list = [replace_nan_with_none(d) for d in result_df.to_dict(orient='records')]

    try:
        with engine.begin() as conn:
            for data in insert_data_list:
                # 代码没有检查 data 有效性
                conn.execute(text(insert_sql), data)
        # print(f"✅ 插入记录数：{len(result_df)}") # 打印信息 (可能不准确)
    except Exception as e:
        print("❌ 插入失败：", e)  # 打印信息
        # 代码在失败时不返回任何东西,这里保持返回列表
        return results  # 或者可以返回 [] 表示失败

    return results  # 代码返回传入的 results


# 温湿度过期判断并写入预测表 (逻辑,恢复循环内插入)
def handle_prediction_results(monitor_df, food_df, engine):
    """处理监控数据, 计算风险并写入预测表"""
    if monitor_df.empty:
        print("ℹ️ 无监控数据需要处理。")
        return []

    if food_df.empty:
        print("⚠️ 缺少食品基础信息,无法进行预测处理。")
        return []

    monitor_df['RecTime'] = pd.to_datetime(monitor_df['RecTime'], errors='coerce')
    monitor_df.dropna(subset=['RecTime', 'TraCode'], inplace=True)
    if monitor_df.empty:
        print("ℹ️ 预处理后无有效的监控数据。")
        return []

    # 预建 TraCode -> food_info 映射，避免循环里反复筛选
    food_index = (food_df.drop_duplicates(subset=['TraCode'], keep='first')
                        .set_index('TraCode')
                        .to_dict('index'))

    # 仅初始化一次 TVBN 模型，传递缓存字典
    predictor_cache = {
        'crayfish_tvbn': Crayfish_TVBNPredictor()
    }

    results = []
    monitor_df_sorted = monitor_df.sort_values(by=['TraCode', 'RecTime'])

    for i, row in monitor_df_sorted.iterrows():
        tra_code = row["TraCode"]
        rec_time = row["RecTime"]
        temp = row.get("Temp")
        humid = row.get("Humid")
        order_number = row.get("OrderNumber")

        food_info = food_index.get(tra_code)
        if not food_info:
            continue

        # --- 过滤掉 RecTime < StorageTime 的记录 ---
        storage_time = get_storage_time(tra_code, rec_time, engine)
        if storage_time and rec_time < storage_time:
            continue

        shelf_life = food_info.get("ShelfLife")
        pro_date = food_info.get("ProDate")
        food_class_code = food_info.get("FoodClassificationCode")

        order_tra_chain = get_order_tra_chain(order_number, tra_code, engine)

        # 温度
        temp_abnormal = is_temp_abnormal(temp, food_info.get("StorTempLower"), food_info.get("StorTempUpper"))
        temp_predict_value = calculate_temp_predict_value(temp_abnormal, order_number, rec_time, tra_code,
                                                          order_tra_chain, engine)
        risk_temp_abnormal = is_exceeding_temp_threshold(food_class_code, temp_predict_value, engine,
                                                         temp_abnormal=temp_abnormal)
        temp_record = generate_temperature_abnormal_record(row, food_info, temp_predict_value,
                                                           risk_temp_abnormal=risk_temp_abnormal)

        # 湿度
        humidity_abnormal = is_humidity_abnormal(humid, food_info.get("StorHumidLower"),
                                                 food_info.get("StorHumidUpper"))
        humid_predict_value = calculate_humidity_predict_value(humidity_abnormal, order_number, rec_time, tra_code,
                                                               order_tra_chain, engine)
        risk_humid_abnormal = is_exceeding_humid_threshold(food_class_code, humid_predict_value, engine,
                                                           humid_abnormal=humidity_abnormal)
        humidity_record = generate_humidity_abnormal_record(row, food_info, humid_predict_value,
                                                            risk_humid_abnormal=risk_humid_abnormal)

        # 过期
        shelf_life_abnormal = is_shelf_life_abnormal(pro_date, rec_time, shelf_life)
        shelf_predict_value = calculate_shelf_life_abnormal_duration(pro_date, rec_time, shelf_life)
        shelf_record = generate_shelf_life_abnormal_record(row, food_info, shelf_predict_value,
                                                           shelf_life_abnormal=shelf_life_abnormal)

        # 标志物（TVB-N）
        marker_records = execute_prediction_unit(row, food_info, engine, predictor_cache)

        # 本条需要写入的所有记录
        records_to_insert = [rec for rec in [temp_record, humidity_record, shelf_record] if rec is not None]
        if marker_records:
            records_to_insert.extend(marker_records)
        #print(marker_records)
        if records_to_insert:
            insert_results(records_to_insert, engine)
            results.extend(records_to_insert)

    print(f"✅ 预测结果处理完成。尝试生成 {len(results)} 条预测记录。")
    return results

