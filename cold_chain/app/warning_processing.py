# -*- coding:utf-8 -*-
import time
import pandas as pd
from datetime import datetime # Ensure datetime is imported if not already
from sqlalchemy import text # text is used
import math
from db.database_setup import prediction_table_name, warning_table_name
from app.prediction_logic import replace_nan_with_none


# ==============================================================================
# --- 预警逻辑函数  ---
# ==============================================================================
def _calculate_storage_duration_hours(tra_code, current_rec_time_dt, engine):
    """
    计算在库时长 (小时)。
    """
    first_time = get_first_station_storage_time(tra_code, engine)  # 使用已修改的函数
    storage_duration_hours = 0.0
    if pd.notna(first_time) and pd.notna(current_rec_time_dt):
        if first_time <= current_rec_time_dt:  # 确保 first_time 不是未来的时间
            time_difference_delta = current_rec_time_dt - first_time
            total_seconds_float = time_difference_delta.total_seconds()  # .total_seconds() 返回的就是 float
            hours_float = total_seconds_float / 3600.0
            storage_duration_hours = math.ceil(hours_float)
    return storage_duration_hours


def _build_warning_record(
        result_dict,  # 预测结果行 (字典)
        tra_code,  # 显式传入 TraCode
        order_number,  # 显式传入 OrderNumber
        secondary_class_name,  # 查找到的二级分类名
        food_name,  # 查找到的食品名
        storage_duration_hours,  # 计算好的在库时长 (小时)
        anomaly_duration_hours,  # 计算好的异常时长/值 (小时或原始值)
        warning_content  # 构建好的预警内容
):
    """
    构建通用的预警记录字典 (恢复 tra_code, order_number 参数)。
    RiskType(用PredictFlag), RiskLevel, RiskName 仍从 result_dict 获取。
    """
    # 从 result_dict 获取其他基础信息
    current_rec_time_dt = pd.to_datetime(result_dict.get("RecTime"), errors='coerce')
    food_class_code = result_dict.get("FoodClassificationCode")
    warning_type = result_dict.get("PredictFlag")  # 使用 PredictFlag 作为 RiskType
    warning_level = result_dict.get("RiskLevel")
    marker_name = result_dict.get("RiskName")  # 使用 RiskName

    return {
        "EalyWarningID": result_dict.get("PredictResultID"),
        "MonitorNum": result_dict.get("MonitorNum"),
        "RecTime": current_rec_time_dt,  # datetime 对象
        "TraCode": tra_code,  # 使用传入的参数
        "OrderNumber": order_number,  # 使用传入的参数
        "FoodClassificationCode": food_class_code,
        "SecondaryClassificationName": secondary_class_name,  # 传入
        "FoodName": food_name,  # 传入
        "Temp": result_dict.get("Temp"),
        "Humid": result_dict.get("Humid"),
        "StorageDuration": storage_duration_hours,  # 传入计算值 (小时)
        "RiskLevel": warning_level,  # 使用来自 result_dict 的值
        "AnomalyDuration": anomaly_duration_hours,
        "RiskType": warning_type,  # 使用来自 result_dict 的 PredictFlag
        "RiskName": marker_name,  # 使用来自 result_dict 的 RiskName
        "ResistControlMethod": None,
        "WarningContent": warning_content,  # 传入
        "IsDone": 0
    }


# 唯一业务键函数
def get_unique_business_keys(engine, start_time_dt, end_time_dt):  # 修改参数名以表示 datetime
    """
    从 risk_prediction_results 表中提取指定时间段内唯一的 (TraCode, OrderNumber) 组合。
    将 NaN 或空字符串的 OrderNumber 处理为 None。

    参数:
        engine:SQLAlchemy 数据库连接引擎。
        start_time_dt (datetime):查询的开始时间 (datetime 对象)。
        end_time_dt (datetime):查询的结束时间 (datetime 对象)。

    返回:
        list:包含唯一 (TraCode, OrderNumber) 元组的列表。
              OrderNumber 为 None 表示原始值是 NaN 或空字符串。
    """
    query = text(f"""
        SELECT DISTINCT TraCode, OrderNumber
        FROM {prediction_table_name}
        WHERE RecTime BETWEEN :start_time AND :end_time
    """)
    try:
        # --- 修改点：直接使用传入的 datetime 对象 ---
        if not isinstance(start_time_dt, datetime) or not isinstance(end_time_dt, datetime):
            print(f"❌ 无效的 datetime 对象传递给 get_unique_business_keys:start={start_time_dt}, end={end_time_dt}")
            # 尝试转换,如果失败则返回空列表
            start_time_dt = pd.to_datetime(start_time_dt, errors='coerce')
            end_time_dt = pd.to_datetime(end_time_dt, errors='coerce')
            if pd.isna(start_time_dt) or pd.isna(end_time_dt):
                return []

        df = pd.read_sql(query, engine, params={"start_time": start_time_dt, "end_time": end_time_dt})
        # --- 修改结束 ---
        if df.empty:
            return []

        keys_df = df[['TraCode', 'OrderNumber']].copy()
        # keys_df['OrderNumber'] = keys_df['OrderNumber'].fillna(np.nan).infer_objects(copy=False).replace('', np.nan)
        unique_keys_list = keys_df.drop_duplicates().values.tolist()  # drop_duplicates 仍然有用,以防万一
        return [(key[0], None if pd.isna(key[1]) else key[1]) for key in unique_keys_list]

    except Exception as e:
        print(f"❌ 查询唯一业务键时出错:{e}")
        return []


KNOWN_VALID_MARKER_NAMES = {
    "菌落总数", "大肠杆菌", "蜡样芽孢杆菌", "挥发性盐基氮", "丙二醛", "组胺",
    # Add ALL your valid marker names here
}


# If this list is very dynamic, you might need another strategy,
# but for a fixed set of marker types, this is standard.
# 得到唯一标志物名称（删掉riskname中的无效字符串比如保质期温湿度上下限）
def get_unique_marker_names(engine, start_time_dt: datetime, end_time_dt: datetime):
    """
    从指定表 (prediction_table_name) 中提取指定时间段内唯一的、已知的标志物名称 (RiskName)。
    将 NaN 或空字符串的标志物名称处理为 None, 并过滤掉非已知标志物。

    参数:
        engine:SQLAlchemy 数据库连接引擎。
        start_time_dt (datetime):查询的开始时间 (datetime 对象)。
        end_time_dt (datetime):查询的结束时间 (datetime 对象)。

    返回:
        list:包含已知唯一的标志物名称 (字符串) 的列表。
    """
    # 确保 prediction_table_name 已被定义
    # This check might be better outside if the function is called many times,
    # or prediction_table_name could be an argument to the function.
    if 'prediction_table_name' not in globals() and 'prediction_table_name' not in locals():
        print(f"❌ 错误 (get_unique_marker_names):prediction_table_name 未定义。")
        return []
    # Ensure KNOWN_VALID_MARKER_NAMES is accessible
    if not KNOWN_VALID_MARKER_NAMES:
        print(f"⚠️ 警告 (get_unique_marker_names):KNOWN_VALID_MARKER_NAMES 为空或未定义。可能返回非预期结果。")

    query = text(f"""
        SELECT DISTINCT RiskName
        FROM {prediction_table_name}
        WHERE RecTime BETWEEN :start_time AND :end_time
    """)
    try:
        # Type checking and conversion for datetime inputs (as in your original code)
        # Consider if this is truly necessary if callers guarantee datetime objects per type hints.
        if not isinstance(start_time_dt, datetime) or not isinstance(end_time_dt, datetime):
            print(
                f"⚠️ (get_unique_marker_names) 无效的 datetime 对象类型传递:start_type={type(start_time_dt)}, end_type={type(end_time_dt)}")
            start_time_dt_orig = start_time_dt
            end_time_dt_orig = end_time_dt
            start_time_dt = pd.to_datetime(start_time_dt, errors='coerce')
            end_time_dt = pd.to_datetime(end_time_dt, errors='coerce')
            if pd.isna(start_time_dt) or pd.isna(end_time_dt):
                print(
                    f"❌ (get_unique_marker_names) 转换 datetime 失败:start_orig='{start_time_dt_orig}', end_orig='{end_time_dt_orig}'")
                return []

        df = pd.read_sql(query, engine, params={"start_time": start_time_dt, "end_time": end_time_dt})

        if df.empty:
            return []

        # Standardize missing values (NaN, empty strings) to np.nan first
        unique_markers_series = df["RiskName"].unique()

        # Convert to list, make np.nan to Python None, and ensure strings
        processed_marker_names = []
        for marker in unique_markers_series:
            if pd.isna(marker):
                # We typically don't consider None as a valid marker *name* to filter for
                pass
            else:
                processed_marker_names.append(str(marker))

        # Filter this list against the known valid marker names
        actual_marker_names_list = [
            marker_name for marker_name in processed_marker_names
            if marker_name in KNOWN_VALID_MARKER_NAMES
        ]

        # If you want to see which ones were filtered out (for debugging):
        # potential_markers_from_db = set(processed_marker_names)
        # unknown_markers = potential_markers_from_db - KNOWN_VALID_MARKER_NAMES
        # if unknown_markers:
        #     print(f"ℹ️ (get_unique_marker_names) 发现但已过滤掉的未知 RiskName 值:{unknown_markers}")

        return actual_marker_names_list

    except Exception as e:
        print(f"❌ (get_unique_marker_names) 查询唯一标志物名称时出错:{e}")
        return []


def get_first_station_storage_time(tra_code, engine):
    """
    获取指定 TraCode 在 cold_chain_food_origin 表中的 StorageTime。
    已修改为使用参数化查询。
    """
    if pd.isna(tra_code):
        return None
    storage_time = None
    try:
        # 使用参数化查询
        origin_query = text("""
            SELECT StorageTime
            FROM cold_chain_food_origin
            WHERE TraCode = :tra_code
            LIMIT 1
        """)
        origin_df = pd.read_sql(origin_query, engine, params={"tra_code": tra_code})

        if not origin_df.empty:
            storage_time_val = origin_df.iloc[0].get('StorageTime')
            if pd.notna(storage_time_val):
                storage_time = pd.to_datetime(storage_time_val, errors='coerce')
    except Exception as e:
        print(f"ℹ️ 查询 StorageTime 时出错 (TraCode:{tra_code}):{e}")
        return None
    return storage_time if pd.notna(storage_time) else None


# 从标志物名称得到风险类型函数
def get_risk_type_from_marker_name(marker_name: str):
    """
    根据标志物名称返回其对应的风险类型。

    参数:
        marker_name (str):标志物的名称。

    返回:
        str | None:如果找到匹配,则返回风险类型 ("生物" 或 "化学")；
                     如果标志物名称为空或未在已知列表中找到,则返回 None。
    """
    RISK_TYPE_BIOLOGICAL = "生物"
    RISK_TYPE_CHEMICAL = "化学"

    BIOLOGICAL_MARKERS = {
        "菌落总数",
        "大肠杆菌",
        "蜡样芽孢杆菌"
        # 您可以在此添加更多生物类标志物
    }

    CHEMICAL_MARKERS = {
        "挥发性盐基氮",
        "丙二醛",
        "组胺"
        # 您可以在此添加更多化学类标志物
    }

    if not marker_name or not isinstance(marker_name, str):
        # print(f"⚠️ 警告 (get_risk_type_from_marker_name):传入的标志物名称无效:{marker_name}") # 可选打印
        return None

    cleaned_marker_name = marker_name.strip()
    # 如果您的标志物名称可能存在大小写不一致的问题,可以添加 .lower() 或 .upper()
    # cleaned_marker_name = marker_name.strip().lower() # 示例：转换为小写

    if cleaned_marker_name in BIOLOGICAL_MARKERS:
        return RISK_TYPE_BIOLOGICAL
    elif cleaned_marker_name in CHEMICAL_MARKERS:
        return RISK_TYPE_CHEMICAL
    else:
        print(f"ℹ️ 信息 (get_risk_type_from_marker_name):未找到标志物 '{cleaned_marker_name}' 对应的风险类型。")  # 可选打印
        return None


# 检查预警是否存在 (最终版逻辑) ---
def check_if_warning_exists(tra_code, order_number, warning_type, warning_level, engine, marker_name=None):
    """
    检查是否存在未完成的同类预警。
    - 对于 '过期', '温度', '湿度',检查是否存在该类型且未处理的预警 (忽略风险等级和标志物)。
    - 对于 '生物', '化学',检查是否存在该类型、该标志物名称(RiskName)且相同风险级别的未处理预警。

    参数:
        tra_code (str):追溯码。
        order_number (str or None):订单号。
        warning_type (str):风险类型 ('过期', '温度', '湿度', '生物', '化学')。
        warning_level (str):风险级别 ('高', '中', '低')。
        engine:SQLAlchemy 数据库连接引擎。
        marker_name (str or None):具体的标志物名称 (仅当 warning_type 为 '生物' 或 '化学' 时应提供)。

    返回:
        bool:True 如果存在匹配的未处理预警,False 否则。
    """
    standard_risk_types = ['过期', '温度', '湿度']
    marker_risk_types = ['生物', '化学']

    # 基础查询条件：匹配 TraCode, RiskType, 和未完成状态
    base_query_sql = f"""
        SELECT 1 FROM {warning_table_name}
        WHERE TraCode = :tra_code
          AND RiskType = :warning_type
          AND (IsDone = 0 OR IsDone IS NULL)
    """
    params = {
        "tra_code": tra_code,
        "warning_type": warning_type,
    }

    # --- 修改点：根据 warning_type 添加 RiskLevel 和 RiskName 条件 ---
    if warning_type in standard_risk_types:
        # 对于标准风险,不再添加其他条件
        pass
    elif warning_type in marker_risk_types:
        # 对于标志物风险,需要匹配 RiskLevel 和 RiskName
        base_query_sql += " AND RiskLevel = :warning_level"
        params["warning_level"] = warning_level

        if pd.isna(marker_name):
            print(f"⚠️ 检查标志物预警 ({warning_type}) 时缺少 marker_name。无法精确检查。")
            return True  # 保守假设存在
        base_query_sql += " AND RiskName = :marker_cn"
        params["marker_cn"] = marker_name
    else:
        # 未知的 RiskType
        print(f"⚠️ 未知的 RiskType '{warning_type}' 在 check_if_warning_exists 中遇到。")
        return True  # 保守假设存在
    # --- 修改结束 ---

    # 添加 OrderNumber 条件
    if pd.isna(order_number) or order_number == '':
        base_query_sql += " AND (OrderNumber IS NULL OR OrderNumber = '')"
    else:
        base_query_sql += " AND OrderNumber = :order_number"
        params["order_number"] = order_number

    base_query_sql += " LIMIT 1;"

    try:
        with engine.connect() as conn:
            result = conn.execute(text(base_query_sql), params).scalar()
            # print(f"Debug check_if_warning_exists:SQL={base_query_sql}, Params={params}, Result={result is not None}") # 调试语句
            return result is not None
    except Exception as e:
        print(f"❌ 检查预警是否存在时出错 (表:{warning_table_name}):{e}")
        return True  # 发生错误时,保守假设预警存在,防止重复


# 插入预警数据
def insert_single_warning(warning_record, engine):
    """
    将单个预警记录插入到 early_warning_information 表。
    (此函数来自您的代码,做了少量调整以适应 warning_record 的构建)
    """
    if not warning_record or not isinstance(warning_record, dict): return 0
    if not warning_record.get('EalyWarningID') or pd.isna(warning_record.get('RecTime')): return 0

    insert_sql = text(f"""
    INSERT INTO {warning_table_name} (
        EalyWarningID, MonitorNum, RecTime, TraCode, OrderNumber,
        FoodClassificationCode, SecondaryClassificationName, FoodName, Temp, Humid,
        StorageDuration, RiskLevel, AnomalyDuration, RiskType, RiskName,
        ResistControlMethod, WarningContent, IsDone
    ) VALUES (
        :EalyWarningID, :MonitorNum, :RecTime, :TraCode, :OrderNumber,
        :FoodClassificationCode, :SecondaryClassificationName, :FoodName, :Temp, :Humid,
        :StorageDuration, :RiskLevel, :AnomalyDuration, :RiskType, :RiskName,
        :ResistControlMethod, :WarningContent, :IsDone
    );
    """)
    # 假设 replace_nan_with_none 是您已有的函数
    insert_data = replace_nan_with_none(warning_record.copy())
    insert_data['IsDone'] = int(insert_data.get('IsDone', 0))

    # RecTime 应该已经是 datetime 对象,由 _build_warning_record 处理
    if pd.notna(insert_data.get('RecTime')) and not isinstance(insert_data['RecTime'], datetime):
        insert_data['RecTime'] = pd.to_datetime(insert_data['RecTime'], errors='coerce')
        if pd.isna(insert_data['RecTime']): insert_data['RecTime']

    for key in ['Temp', 'Humid', 'StorageDuration', 'AnomalyDuration']:
        if pd.notna(insert_data.get(key)):
            try:
                insert_data[key] = float(insert_data[key])
            except (ValueError, TypeError):
                insert_data[key]
        elif key == 'StorageDuration' and insert_data.get(key) is None:  # 仅当确实为 None 时设为 0.0
            insert_data[key] = 0.0

    try:
        with engine.begin() as conn:
            result_proxy = conn.execute(insert_sql, insert_data)
            return result_proxy.rowcount
    except Exception as e:
        if "duplicate entry" in str(e).lower() and "primary" in str(e).lower():
            pass
        elif "foreign key constraint fails" in str(e).lower():
            pass
        elif "column 'storageduration' cannot be null" in str(e).lower():
            print(f"❌ StorageDuration 仍然为 NULL:{insert_data}")
        else:
            print(f"❌ 插入预警失败:{e}\n    Data:{insert_data}")
        return 0


def fetch_risk_data(engine,
                    tra_code: str,
                    risk_type: str,
                    order_number: str,  # 假设这是必需的, 如果可选则添加 =None
                    risk_level: str,  # 假设这是必需的, 如果可选则添加 =None
                    start_time_dt: datetime,  # 必需
                    end_time_dt: datetime,  # 必需
                    marker_name: str = None  # 可选参数,现在位置正确
                    ):
    """
    根据不同的风险类型和条件从数据库中读取风险数据,可选择时间范围。
    参数:
        engine:SQLAlchemy 数据库连接引擎。
        tra_code (str):追溯码。
        risk_type (str):风险类型 ("过期", "温度", "湿度", "生物", "化学")。
        order_number (str, optional):订单号。默认为 None。
        risk_level (str, optional):风险等级。现在由调用者为所有类型提供。
        marker_name (str, optional):标志物名称。默认为 None。
        start_time_dt (datetime):查询的开始时间 (包含)。必须提供。
        end_time_dt (datetime):查询的结束时间 (包含)。必须提供。

    返回:
        pandas.DataFrame:查询结果。
    """
    if not tra_code:
        print("❌ 错误:追溯码 (tra_code) 不能为空。")
        return pd.DataFrame()

    # 确保 prediction_table_name 已被定义
    if 'prediction_table_name' not in globals() and 'prediction_table_name' not in locals():
        print("❌ 错误:prediction_table_name 未定义。请在调用此函数前定义它。")
        return pd.DataFrame()

    base_query = f"SELECT * FROM {prediction_table_name} WHERE TraCode = :tra_code"
    params = {"tra_code": tra_code}

    # 处理可选的 OrderNumber
    if order_number and order_number.strip():
        base_query += " AND OrderNumber = :order_number"
        params["order_number"] = order_number
    else:
        base_query += " AND OrderNumber IS NULL"

    # 根据风险类型构建特定查询条件
    if risk_type == '过期':
        base_query += " AND PredictFlag = :risk_type AND RiskLevel = :risk_level_input"
        params["risk_type"] = risk_type
        params["risk_level_input"] = risk_level

    elif risk_type == '温度' or risk_type == '湿度':
        base_query += " AND PredictFlag = :risk_type AND RiskLevel = :risk_level_input"
        params["risk_type"] = risk_type
        params["risk_level_input"] = risk_level

    elif risk_type == '生物' or risk_type == '化学':
        base_query += " AND PredictFlag = :risk_type AND RiskLevel = :risk_level_input AND RiskName = :marker_name_input"
        params["risk_type"] = risk_type
        params["risk_level_input"] = risk_level
        params["marker_name_input"] = marker_name

    else:
        print(f"❌ 错误:未知的风险类型 '{risk_type}'。")
        return pd.DataFrame()

    # --- 新增时间过滤逻辑 ---
    # 假设表中的时间列名为 'RecTime'
    time_column_name = "RecTime"  # 如果您的时间列名不同,请修改这里

    base_query += f" AND {time_column_name} >= :start_time"
    params["start_time"] = start_time_dt

    base_query += f" AND {time_column_name} <= :end_time"
    params["end_time"] = end_time_dt

    try:
        # 1. 获取所有符合条件的记录
        all_matching_df = pd.read_sql(text(base_query), engine, params=params)
        if not all_matching_df.empty:
            # 2. 检查用于排序的时间列是否存在
            try:
                # 3. 确保时间列是 datetime 类型,以便正确排序
                all_matching_df[time_column_name] = pd.to_datetime(all_matching_df[time_column_name])
                # 4. 排序并提取最早的一条记录
                earliest_record_df = all_matching_df.sort_values(by=time_column_name, ascending=True).head(1)

                return earliest_record_df  # 返回包含单条最早记录的DataFrame
            except Exception as e:
                print(f"❌ 在对数据进行排序或提取最早记录时出错:{e}。将返回空DataFrame。")
                return pd.DataFrame()
        else:
            print("ℹ️ 查询完成 (fetch_risk_data),未找到匹配数据。")
            return all_matching_df  # 返回空的 DataFrame
    except Exception as e:
        print(f"❌ 数据库查询失败 (fetch_risk_data):{e}")
        return pd.DataFrame()


# 过期预警判断
def _check_and_handle_expiration_warning(tra_code, order_number, engine, secondary_class_name, food_name, start_time_dt,
                                         end_time_dt):  # Returns True if a new warning was inserted, False otherwise
    """
    检查并处理指定追溯码和订单号在给定时间范围内的过期预警。
    如果发现符合条件的、且尚未记录的过期风险数据（基于最早记录）,则记录新预警。

    参数:
        tra_code (str):追溯码。
        order_number (str):订单号 (如果适用,否则可为 None 或空字符串)。
        engine:SQLAlchemy 数据库连接引擎。
        secondary_class_name (str):食品的二级分类名称。
        food_name (str):食品名称。
        start_time_dt (datetime):查询的开始时间。
        end_time_dt (datetime):查询的结束时间。
    返回:
        bool:True 如果成功生成并插入了新的预警,否则 False。
    """
    print(
        f"--- 开始检查过期预警 ({food_name},OrderNumber:{order_number} TraCode:{tra_code}, 时间:{start_time_dt.strftime('%Y-%m-%d')} to {end_time_dt.strftime('%Y-%m-%d')}) ---")

    target_risk_type = "过期"
    target_risk_level = "高"

    # fetch_risk_data is expected to return a DataFrame with 0 or 1 row (the earliest)
    expiration_data_df = fetch_risk_data(
        engine=engine,
        tra_code=tra_code,
        risk_type=target_risk_type,
        order_number=order_number,
        risk_level=target_risk_level,
        start_time_dt=start_time_dt,
        end_time_dt=end_time_dt
    )

    if not expiration_data_df.empty:
        # expiration_data_df 包含唯一的一条最早记录 (单行DataFrame)
        print(f"ℹ️ 成功获取到最早的一条潜在过期风险数据 (产品 '{food_name}', 追溯码:{tra_code})。")

        # **修正：从单行DataFrame中正确提取数据**
        earliest_data_row_series = expiration_data_df.iloc[0]  # 获取该行作为Pandas Series
        earliest_data_dict = earliest_data_row_series.to_dict()  # 将该行转换为字典

        warning_already_exists = check_if_warning_exists(
            tra_code=tra_code,
            order_number=order_number,
            warning_type=target_risk_type,
            warning_level=target_risk_level,
            engine=engine
        )

        if not warning_already_exists:
            print(f"  该过期预警 (类型:{target_risk_type}, 等级:{target_risk_level}) 尚未记录。准备生成新预警...")

            # 获取预警内容里的需要的字段内容
            monitornum_record = earliest_data_dict.get("MonitorNum")
            order_num_display = order_number if order_number and order_number.strip() else "None"
            temp_record = earliest_data_dict.get('Temp')
            current_rec_time_dt = earliest_data_dict.get('RecTime')
            rectime_record = current_rec_time_dt
            anomaly_value_for_warning_record = earliest_data_dict.get('PredictValue', None)  # Placeholder
            anomaly_value_for_warning_record = math.ceil(
                (anomaly_value_for_warning_record + 365) * 24),  # 传入计算值 (分钟转为小时并向上取整)
            # earliest_rec_time_str 现在已定义
            warning_content = (
                f"监测编号:{monitornum_record},追溯码:{tra_code},订单编号:{order_num_display},食品名称:{food_name},"
                f"温度:{temp_record}℃,记录时间:{rectime_record},过期{anomaly_value_for_warning_record}天,建议处置方法:停止储运。"
            )

            placeholder_storage_duration = _calculate_storage_duration_hours(tra_code, current_rec_time_dt, engine)

            # **修正：从 earliest_data_row_series 获取 "PredictValue"**
            placeholder_anomaly_duration = earliest_data_row_series.get("PredictValue")

            # print(f"  用于构建预警的存储时长:{placeholder_storage_duration} 小时")
            # print(f"  用于构建预警的异常时长/值 (来自 PredictValue):{placeholder_anomaly_duration}")

            # **修正：传递正确的 result_dict**
            warning_record = _build_warning_record(
                result_dict=earliest_data_dict,  # 传递最早记录的字典
                tra_code=tra_code,
                order_number=order_number,
                secondary_class_name=secondary_class_name,
                food_name=food_name,
                storage_duration_hours=placeholder_storage_duration,
                anomaly_duration_hours=placeholder_anomaly_duration,
                warning_content=warning_content
            )

            if warning_record:
                try:
                    insert_result = insert_single_warning(warning_record, engine)
                    if insert_result is not False:  # Catches True and None as success
                        print(f"✅ 新的过期预警已成功插入数据库。")
                        return True
                    else:
                        print(f"❌ 插入新的过期预警失败 (insert_single_warning 返回 False)。")
                        return False
                except Exception as e:
                    print(f"❌ 插入新的过期预警时发生数据库错误:{e}")
                    return False
            else:
                print(f"❌ 构建预警记录失败 (_build_warning_record 返回 None 或空)。")
                return False
        else:
            print(f"ℹ️ 此过期预警 (类型:{target_risk_type}, 等级:{target_risk_level}) 已存在于预警表中,跳过。")
            return True  # 没有生成新的预警 但是存在过期数据此业务键就不在判断温湿度以及标志物
    else:
        print(f"✅ 在指定时间范围内未发现与产品 '{food_name}' (追溯码:{tra_code}) 相关的过期风险数据。")
        return False  # 没有生成新的预警


# 温度预警判断

def _check_and_handle_temperature_warning(tra_code, order_number, engine, secondary_class_name, food_name,
                                          start_time_dt, end_time_dt):
    print(
        f"--- 开始检查温度预警 ({food_name},OrderNumber:{order_number} TraCode:{tra_code}, 时间:{start_time_dt.strftime('%Y-%m-%d')} to {end_time_dt.strftime('%Y-%m-%d')}) ---")

    target_risk_type = "温度"
    target_risk_level = "低"  # Or another appropriate level for temperature
    target_marker_name = "无"

    temperature_data_df = fetch_risk_data(
        engine=engine,
        tra_code=tra_code,
        risk_type=target_risk_type,
        order_number=order_number,
        risk_level=target_risk_level,
        start_time_dt=start_time_dt,
        end_time_dt=end_time_dt
    )

    if not temperature_data_df.empty:
        print(f"ℹ️ 成功获取到最早的一条潜在温度风险数据 (产品 '{food_name}', 追溯码:{tra_code})。")

        earliest_data_row_series = temperature_data_df.iloc[0]
        earliest_data_dict = earliest_data_row_series.to_dict()

        warning_already_exists = check_if_warning_exists(
            tra_code=tra_code,
            order_number=order_number,
            warning_type=target_risk_type,
            warning_level=target_risk_level,
            engine=engine
        )

        if not warning_already_exists:
            print(f"  该温度预警 (类型:{target_risk_type}, 等级:{target_risk_level}) 尚未记录。准备生成新预警...")

            monitornum_record = earliest_data_dict.get("MonitorNum")

            order_num_display = order_number if order_number and order_number.strip() else None
            temp_record = earliest_data_dict.get('Temp')
            current_rec_time_dt = earliest_data_dict.get('RecTime')
            rectime_record = current_rec_time_dt

            anomaly_value_for_warning_record = earliest_data_dict.get('PredictValue', None)  # Try to get duration first
            anomaly_value_for_warning_record = math.ceil(anomaly_value_for_warning_record / 60)  # 传入计算值 (分钟转为小时并向上取整)
            riskname_record = earliest_data_dict.get('RiskName')
            warning_content = (
                f"监测编号:{monitornum_record},追溯码:{tra_code},订单编号:{order_num_display},食品名称:{food_name},温度:{temp_record}℃,"
                f"记录时间:{rectime_record},当前超出储存温度范围({riskname_record}),温度异常时长为{anomaly_value_for_warning_record}小时,建议处置方法:调节温度。"
            )

            placeholder_storage_duration = _calculate_storage_duration_hours(tra_code, current_rec_time_dt,
                                                                             engine)  # Define how to calculate this

            warning_record = _build_warning_record(
                result_dict=earliest_data_dict,
                tra_code=tra_code,
                order_number=order_number,
                secondary_class_name=secondary_class_name,
                food_name=food_name,
                storage_duration_hours=placeholder_storage_duration,
                anomaly_duration_hours=anomaly_value_for_warning_record,  # This can be duration or value
                warning_content=warning_content
            )

            if warning_record:
                try:
                    insert_result = insert_single_warning(warning_record, engine)
                    if insert_result is not False:
                        print(f"✅ 新的温度预警已成功插入数据库。")
                        return True
                    else:
                        print(f"❌ 插入新的温度预警失败 (insert_single_warning 返回 False)。")
                        return False
                except Exception as e:
                    print(f"❌ 插入新的温度预警时发生数据库错误:{e}")
                    return False
            else:
                print(f"❌ 构建温度预警记录失败 (_build_warning_record 返回 None 或空)。")
                return False
        else:
            print(f"ℹ️ 此温度预警 (类型:{target_risk_type}, 等级:{target_risk_level}) 已存在于预警表中,跳过。")
            return False
    else:
        print(f"✅ 在指定时间范围内未发现与产品 '{food_name}' (追溯码:{tra_code}) 相关的温度风险数据。")
        return False


# 湿度预警判断
def _check_and_handle_humidity_warning(tra_code, order_number, engine, secondary_class_name, food_name, start_time_dt,
                                       end_time_dt):
    print(
        f"--- 开始检查湿度预警 ({food_name}, OrderNumber:{order_number}TraCode:{tra_code}, 时间:{start_time_dt.strftime('%Y-%m-%d')} to {end_time_dt.strftime('%Y-%m-%d')}) ---")

    target_risk_type = "湿度"
    target_risk_level = "低"

    humidity_data_df = fetch_risk_data(
        engine=engine,
        tra_code=tra_code,
        risk_type=target_risk_type,
        order_number=order_number,
        risk_level=target_risk_level,
        start_time_dt=start_time_dt,
        end_time_dt=end_time_dt
    )

    if not humidity_data_df.empty:
        print(f"ℹ️ 成功获取到最早的一条潜在湿度风险数据 (产品 '{food_name}', 追溯码:{tra_code})。")

        earliest_data_row_series = humidity_data_df.iloc[0]
        earliest_data_dict = earliest_data_row_series.to_dict()

        earliest_rec_time = earliest_data_row_series.get("RecTime")

        warning_already_exists = check_if_warning_exists(
            tra_code=tra_code,
            order_number=order_number,
            warning_type=target_risk_type,
            warning_level=target_risk_level,
            engine=engine
        )

        if not warning_already_exists:
            print(f"  该湿度预警 (类型:{target_risk_type}, 等级:{target_risk_level}) 尚未记录。准备生成新预警...")

            # --- Determine anomaly_value_or_duration and warning_content ---
            monitornum_record = earliest_data_dict.get("MonitorNum")

            order_num_display = order_number if order_number and order_number.strip() else "None"
            humid_record = earliest_data_dict.get('Humid')
            current_rec_time_dt = earliest_data_dict.get('RecTime')
            rectime_record = current_rec_time_dt
            riskname_record = earliest_data_dict.get('RiskName')
            anomaly_value_for_warning_record = earliest_data_dict.get('PredictValue', None)  # Placeholder
            anomaly_value_for_warning_record = math.ceil(anomaly_value_for_warning_record / 60)  # 传入计算值 (分钟转为小时并向上取整)
            warning_content = (
                f"监测编号:{monitornum_record},追溯码:{tra_code},订单编号:{order_num_display},食品名称:{food_name},湿度:{humid_record}℃,"
                f"记录时间:{rectime_record},当前超出储存湿度范围({riskname_record}),湿度异常时长为{anomaly_value_for_warning_record}小时,建议处置方法:调节湿度。"
            )

            placeholder_storage_duration = _calculate_storage_duration_hours(tra_code, current_rec_time_dt,
                                                                             engine)  # Define how to calculate this

            warning_record = _build_warning_record(
                result_dict=earliest_data_dict,
                tra_code=tra_code,
                order_number=order_number,
                secondary_class_name=secondary_class_name,
                food_name=food_name,
                storage_duration_hours=placeholder_storage_duration,
                anomaly_duration_hours=anomaly_value_for_warning_record,  # This can be duration or value
                warning_content=warning_content
            )

            if warning_record:
                try:
                    insert_result = insert_single_warning(warning_record, engine)
                    if insert_result is not False:
                        print(f"✅ 新的湿度预警已成功插入数据库。")
                        return True
                    else:
                        print(f"❌ 插入新的湿度预警失败 (insert_single_warning 返回 False)。")
                        return False
                except Exception as e:
                    print(f"❌ 插入新的湿度预警时发生数据库错误:{e}")
                    return False
            else:
                print(f"❌ 构建湿度预警记录失败 (_build_warning_record 返回 None 或空)。")
                return False
        else:
            print(f"ℹ️ 此湿度预警 (类型:{target_risk_type}, 等级:{target_risk_level}) 已存在于预警表中,跳过。")
            return False
    else:
        print(f"✅  在指定时间范围内未发现与产品 '{food_name}' (追溯码:{tra_code}) 相关的湿度风险数据。")
        return False


# 标志物预警判断
def _check_and_handle_marker_warning(tra_code: str,
                                     order_number: str,
                                     engine,  # SQLAlchemy 引擎
                                     secondary_class_name: str,
                                     food_name: str,
                                     start_time_dt: datetime,
                                     end_time_dt: datetime
                                     ) -> bool:
    """
    检查并处理指定追溯码、订单号、风险类型在给定时间范围内的标志物预警。
    会针对高、中、低三个风险等级,对每个唯一检测到的标志物分别检查。
    如果发现符合条件的、且尚未记录的风险数据,则记录新预警。
    优先处理高级别风险,若处理（或已存在）则跳过该标志物的低级别检查。
    """
    risk_level_high = "高"
    risk_level_mid = "中"
    risk_level_low = "低"

    # 得到唯一标志物名称
    unique_marker_names = get_unique_marker_names(engine, start_time_dt, end_time_dt)
    print(unique_marker_names)
    # 如果列表为空,或者列表只包含一个 None 元素,则不进行后续步骤
    if not unique_marker_names or unique_marker_names == [None]:
        print(f"ℹ️ 在指定时间范围内未找到任何有效或特定的唯一标志物名称 类型检查。")  # 可选打印
        return False

    any_new_warning_generated_overall = False

    for current_marker_name in unique_marker_names:  # current_marker_name 可能为 None
        risk_type = get_risk_type_from_marker_name(current_marker_name)  # current_marker_name is a string here

        # Step 2:Check if a valid risk_type was found
        if risk_type is None:
            print(
                f"⚠️ ({food_name}, TraCode:{tra_code}) 无法确定标志物 '{current_marker_name}' 的风险类型,跳过此标志物的预警检查。")
            continue  # Skip to the next marker in the loop

        # Now risk_type is guaranteed to be assigned and not None
        print(
            f"--- ({food_name},OrderNumber:{order_number} TraCode:{tra_code}) 检查标志物:'{current_marker_name}' (类型:{risk_type}) ---")

        processed_marker_for_higher_level = False  # 标记此标志物是否已因高级别风险处理过

        # 1. 检查高风险
        marker_data_df_high = fetch_risk_data(engine, tra_code, risk_type, order_number, risk_level_high, start_time_dt,
                                              end_time_dt, current_marker_name)
        if not marker_data_df_high.empty:
            warning_exists_high = check_if_warning_exists(tra_code, order_number, risk_type, risk_level_high, engine,
                                                          current_marker_name)
            if not warning_exists_high:
                if _process_single_marker_level_warning(engine, tra_code, order_number, secondary_class_name, food_name,
                                                        risk_type, current_marker_name, risk_level_high,
                                                        marker_data_df_high.iloc[0].to_dict()):
                    any_new_warning_generated_overall = True
            else:
                print(f"    ℹ️ 高风险预警 (标志物:{current_marker_name}) 已存在。")  # 可选打印
            processed_marker_for_higher_level = True  # 无论新旧,高风险数据存在即处理完毕
            print(f"    ℹ️ 已处理/存在高风险数据,跳过此标志物 '{current_marker_name}' 的中低风险检查。")  # 可选打印
            continue  # 跳到下一个 unique_marker_name

        # if processed_marker_for_higher_level:
        # print(f"    ℹ️ 已处理/存在高风险数据,跳过此标志物 '{current_marker_name}' 的中低风险检查。") # 可选打印
        # continue # 跳到下一个 unique_marker_name

        # 2. 检查中风险 (仅当高风险未处理/不存在时)
        marker_data_df_medium = fetch_risk_data(engine, tra_code, risk_type, order_number, risk_level_mid,
                                                start_time_dt, end_time_dt, current_marker_name)
        if not marker_data_df_medium.empty:
            warning_exists_medium = check_if_warning_exists(tra_code, order_number, risk_type, risk_level_mid, engine,
                                                            current_marker_name)
            if not warning_exists_medium:
                if _process_single_marker_level_warning(engine, tra_code, order_number, secondary_class_name, food_name,
                                                        risk_type, current_marker_name, risk_level_mid,
                                                        marker_data_df_medium.iloc[0].to_dict()):
                    any_new_warning_generated_overall = True
            else:
                print(f"    ℹ️ 中风险预警 (标志物:{current_marker_name}) 已存在。")  # 可选打印
            processed_marker_for_higher_level = True
            print(f"    ℹ️ 已处理/存在中风险数据,跳过此标志物 '{current_marker_name}' 的低风险检查。")  # 可选打印
            continue  # 跳到下一个 unique_marker_name

        # if processed_marker_for_higher_level:

        # 3. 检查低风险 (仅当高和中风险均未处理/不存在时)
        marker_data_df_low = fetch_risk_data(engine, tra_code, risk_type, order_number, risk_level_low, start_time_dt,
                                             end_time_dt, current_marker_name)
        if not marker_data_df_low.empty:
            warning_exists_low = check_if_warning_exists(tra_code, order_number, risk_type, risk_level_low, engine,
                                                         current_marker_name)
            if not warning_exists_low:
                if _process_single_marker_level_warning(engine, tra_code, order_number, secondary_class_name, food_name,
                                                        risk_type, current_marker_name, risk_level_low,
                                                        marker_data_df_low.iloc[0].to_dict()):
                    any_new_warning_generated_overall = True
            # else:print(f"    ℹ️ 低风险预警 (标志物:{current_marker_name}) 已存在。") # 可选打印
        # else:print(f"    ✅ 未发现低风险数据 (标志物:{current_marker_name})。") # 可选打印

    print(f"--- 标志物预警检查完成。整体生成状态:{any_new_warning_generated_overall} ---")  # 可选打印
    return any_new_warning_generated_overall


# 单个标志物预警处理
def _process_single_marker_level_warning(engine, tra_code, order_number, secondary_class_name, food_name,
                                         risk_type, current_marker_name, current_risk_level,
                                         earliest_data_dict  # 移除了 start_time_dt 和 end_time_dt
                                         ) -> bool:
    """
    辅助函数：处理单个标志物、单个风险等级的预警生成和插入。
    """
    earliest_data_row_series = pd.Series(earliest_data_dict)

    # 处理时间
    current_rec_time_dt = earliest_data_row_series.get("RecTime")
    rectime_record = current_rec_time_dt
    monitornum_record = earliest_data_row_series.get('MonitorNum')
    order_num_display = order_number if order_number and order_number.strip() else "None"
    marker_display_name = current_marker_name if current_marker_name is not None else "未指定标志物"
    temp_record = earliest_data_row_series.get('Temp')
    anomaly_value_for_warning_record = earliest_data_row_series.get('PredictValue')

    # 修正：warning_content 不再引用 start_time_dt 和 end_time_dt
    warning_content = (
        f"监测编号:{monitornum_record},追溯码:{tra_code},订单编号:{order_num_display},食品名称:{food_name},温度:{temp_record}℃,"
        f"记录时间:{rectime_record},预测{risk_type}标志物{current_marker_name}为{current_risk_level}风险。建议检测{current_marker_name},建议处置方法:XXX技术。"
    )

    storage_duration = 0.0
    storage_duration = _calculate_storage_duration_hours(tra_code, current_rec_time_dt, engine)

    warning_record = _build_warning_record(
        result_dict=earliest_data_dict,
        tra_code=tra_code,
        order_number=order_number,
        secondary_class_name=secondary_class_name,
        food_name=food_name,
        storage_duration_hours=storage_duration,
        anomaly_duration_hours=anomaly_value_for_warning_record,
        warning_content=warning_content
    )

    if warning_record:
        try:
            insert_result = insert_single_warning(warning_record, engine)
            if insert_result is not False:
                return True
        except Exception:
            pass
    return False


# ==============================================================================
# --- 主处理函数 process_warnings  ---
# ==============================================================================

def process_warnings(start_time_dt: datetime, end_time_dt: datetime, food_df: pd.DataFrame,
                     engine):  # food_df 现在是 DataFrame
    start_process_time = time.time()
    total_warnings_generated = 0

    print(
        f"--- 预警处理开始,查询时间段:{start_time_dt.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time_dt.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # ----1、 获取唯一业务键
    unique_business_keys = get_unique_business_keys(engine, start_time_dt, end_time_dt)

    if not unique_business_keys:
        print(f"--- 未找到任何业务键进行处理。 ---")
    else:
        print(f"--- 成功获取 {len(unique_business_keys)} 个唯一业务键进行处理 ---")
    # ---2、 遍历业务键
    for tra_code, order_number in unique_business_keys:
        # print(f"\n--- 正在处理业务键:TraCode={tra_code}, OrderNumber={order_number or 'None'} ---") # 可选打印

        # **修改：从传入的 food_df DataFrame 中查询 FoodName 和 SecondaryClassName**
        food_name = None
        secondary_class_name = None
        # 获取该追溯码对应的食品信息
        matched_food_rows = food_df[food_df['TraCode'] == tra_code]
        food_row = matched_food_rows.iloc[0]

        food_name = food_row.get('FoodName')  # 使用 .get() 以防列不存在
        secondary_class_name = food_row.get('SecondaryClassificationName')

        if food_name is None or secondary_class_name is None:
            print(
                f"  ⚠️ 未能从 food_df 中为 TraCode='{tra_code}' 获取有效的 FoodName 和/或 SecondaryClassName,跳过此业务键的预警处理。")
            continue

        # ---3、过期预警
        if _check_and_handle_expiration_warning(tra_code, order_number, engine, secondary_class_name, food_name,
                                                start_time_dt, end_time_dt):
            total_warnings_generated += 1
            continue

            # ---4、 温度预警
        if _check_and_handle_temperature_warning(tra_code, order_number, engine, secondary_class_name, food_name,
                                                 start_time_dt, end_time_dt):
            total_warnings_generated += 1
        # ---5、湿度预警
        if _check_and_handle_humidity_warning(tra_code, order_number, engine, secondary_class_name, food_name,
                                              start_time_dt, end_time_dt):
            total_warnings_generated += 1
        # --- 6
        if _check_and_handle_marker_warning(tra_code, order_number, engine, secondary_class_name, food_name,
                                            start_time_dt, end_time_dt):
            total_warnings_generated += 1

    end_process_time = time.time()
    print(
        f"--- 预警处理结束,总耗时:{end_process_time - start_process_time:.2f} 秒。共生成 {total_warnings_generated} 条新预警。 ---")



