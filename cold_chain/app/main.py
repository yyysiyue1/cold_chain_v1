# Standard library imports
import time
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text # Used for monitor_query_sql_template
import logging # <--- 1. 导入 logging 模块

# Your custom modules
from db import database_setup
import prediction_logic

# ==============================================================================
# --- 日志配置 ---
# ==============================================================================
# 2. 配置日志记录器
log_file_name = '../cold_chain_app.log'
logging.basicConfig(
    level=logging.INFO, # 设置记录的最低级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s', # 日志格式
    handlers=[
        logging.FileHandler(log_file_name, mode='a', encoding='utf-8'), # 输出到文件，追加模式
        logging.StreamHandler() # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__) # 获取当前模块的 logger 实例

# ==============================================================================
# --- 主执行逻辑 (循环处理指定时间范围) ---
# ==============================================================================
if __name__ == "__main__":
    script_overall_start_time = time.time()
    logger.info("================ SCRIPT START ================") # 记录脚本开始

    # --- 1. Loop Configuration ---
    loop_processing_start_str = "2025-03-04 11:00:01"
    loop_processing_overall_end_str = "2025-05-27 16:00:00"

    time_format = "%Y-%m-%d %H:%M:%S"
    window_duration = timedelta(minutes=30) # 您的设置是30分钟
    # window_duration = timedelta(days=2) # 之前的2天设置

    try:
        current_window_start_dt = datetime.strptime(loop_processing_start_str, time_format)
        overall_end_dt = datetime.strptime(loop_processing_overall_end_str, time_format)
        logger.info("Starting iterative processing.") # <--- 3. 替换 print
        logger.info(f"First window will process data from: {current_window_start_dt}")
        logger.info(f"Processing will cover data up to (exclusive): {overall_end_dt}")
        logger.info(f"Each window duration: {window_duration}")
    except ValueError as e:
        logger.error(f"Invalid loop date format in configuration: {e}", exc_info=True) # exc_info=True 会记录异常堆栈
        exit()

    engine = None # 初始化 engine
    try:
        engine = database_setup.connect_to_db()
    except Exception as e_db_conn: # 更广泛地捕获连接中的任何异常
        logger.error(f"Critical error during database_setup.connect_to_db(): {e_db_conn}", exc_info=True)
        # engine 会保持为 None

    if not engine:
        logger.error("Database engine not initialized. Exiting.")
        exit()

    # --- 2. 获取食品基础信息 (food_df) - Once before the loop ---
    logger.info("⏳ 正在获取食品基础信息 (一次性)...")
    try:
        food_df = prediction_logic.get_food_info(engine)
        if food_df is None:
            food_df = pd.DataFrame()
    except Exception as e_food:
        logger.error(f"获取食品基础信息失败: {e_food}", exc_info=True)
        food_df = pd.DataFrame()

    if food_df.empty:
        logger.warning("未能获取食品基础信息 (food_df 为空), 预警中的名称字段可能为空。")
        required_food_cols = ['TraCode', 'FoodClassificationCode', 'SecondaryClassificationName', 'FoodName']
        for col in required_food_cols:
            if col not in food_df.columns:
                food_df[col] = None
    else:
        logger.info(f"成功获取 {len(food_df)} 条食品基础信息。")

    monitor_query_sql_template = text(f"""
        SELECT mit.*
        FROM monitoring_informationtable mit
        WHERE mit.RecTime >= :start_time AND mit.RecTime < :end_time
    """)

    iteration_count = 0
    total_monitor_records_processed = 0

    # --- Main Processing Loop ---
    while current_window_start_dt < overall_end_dt:
        iteration_count += 1

        monitor_time_limit = current_window_start_dt
        target_time = current_window_start_dt + window_duration

        if target_time > overall_end_dt:
            target_time = overall_end_dt

        if monitor_time_limit >= target_time:
            logger.info(
                f"Remaining interval [{monitor_time_limit}, {target_time}) is too small or invalid. Loop finished.")
            break

        logger.info(f"--- Iteration {iteration_count}: Processing window [{monitor_time_limit}, {target_time}) ---")
        iteration_local_start_time = time.time()

        logger.info(f"⏳ 正在查询监控数据 for window: {monitor_time_limit} to {target_time}...")
        monitor_df = pd.DataFrame() # 初始化以防查询失败
        try:
            monitor_df = pd.read_sql(
                monitor_query_sql_template,
                engine,
                params={"start_time": monitor_time_limit, "end_time": target_time}
            )
            logger.info(f"查询到 {len(monitor_df)} 条监控记录。")
            total_monitor_records_processed += len(monitor_df)
        except Exception as e_mon:
            logger.error(f"查询监控数据失败: {e_mon}", exc_info=True)
            # monitor_df 仍然是空的 DataFrame

        if not monitor_df.empty:
            if food_df.empty and iteration_count == 1:
                logger.warning("缺少食品基础信息, 预测结果处理可能受影响。")

            logger.info("⏳ 正在处理预测结果并写入数据库...")
            try:
                prediction_logic.handle_prediction_results(monitor_df, food_df, engine)
                logger.info("预测结果处理完成。")
            except Exception as e_handle:
                logger.error(f"处理预测结果时出错: {e_handle}", exc_info=True)
        else:
            logger.info("无监控数据在本窗口, 跳过预测结果处理。")

        # logger.info(f"⏳ 执行预警分析 (process_warnings) for window [{monitor_time_limit}, {target_time})...")
        # try:
        #     warning_processing.process_warnings(monitor_time_limit, target_time, food_df, engine)
        #     logger.info("预警分析执行完成。") # 添加一条完成信息
        # except Exception as e_warn:
        #     logger.error(f"执行预警分析时出错: {e_warn}", exc_info=True)
        #
        # iteration_local_end_time = time.time()
        # logger.info(
        #     f"⏱️ Iteration {iteration_count} 完成, 耗时: {iteration_local_end_time - iteration_local_start_time:.2f} 秒。")

        current_window_start_dt += window_duration

        if current_window_start_dt >= overall_end_dt and iteration_count > 0:
            logger.info(
                f"下一个窗口的起始时间 ({current_window_start_dt}) 已达到或超过总体结束时间 ({overall_end_dt})。循环即将结束。")

        # 您之前的代码中 time.sleep(10) 被注释掉了，我保留了这个注释。
        # 如果需要暂停，可以取消注释 time.sleep()
        # logger.info(f"等待0s然后继续读取数据 (Current time in Germany: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        # time.sleep(0) # 或者 time.sleep(10)

    if iteration_count == 0:
        logger.warning("循环未执行任何迭代。请检查循环起止日期和窗口时长。")
    else:
        logger.info(f"完成 {iteration_count} 个迭代的处理。")
        logger.info(f"总共查询了 {total_monitor_records_processed} 条监控记录。")

    script_overall_end_time = time.time()
    logger.info(f"🎉 全部预警测试流程完成, 总耗时: {script_overall_end_time - script_overall_start_time:.2f} 秒。")
    logger.info("================ SCRIPT END ================")