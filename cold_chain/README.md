# 【项目名称：冷链预测系统】

> 本项目旨在为冷链物流提供数据驱动的解决方案，通过构建预测模型和自动化数据处理，提升冷链管理的效率和可靠性。

## 项目简介

本项目是一个综合性的冷链管理工具，其核心功能包括：
-   **数据处理与预测：** 利用`tvbn`和`advanced_prediction_models`等多种预测模型，对冷链数据进行分析和预测。
-   **数据库交互：** 自动执行数据库设置 (`database_setup.py`)，确保数据存储的完整性。
-   **警告处理：** 包含一个专门的模块 (`warning_processing.py`) 用于处理和响应系统产生的警告。
-   **配置管理：** 所有系统配置都可以通过 `config.ini` 文件进行灵活调整。

## 文件结构说明

-   `main.py`: 项目主程序入口。
-   `advanced_prediction_models.py`: 集中存放动态预测模型代码。
-   `tvbn_predictor.py`: `tvbn` 预测模型的调用方法。
-   `tvbn_predictor_model.pkl`: 训练好的 `tvbn` 模型文件。
-   `train_tvbn_model.py`: 用于训练 `tvbn` 模型的脚本。
-   `database_setup.py`: 数据库初始化和表结构创建脚本。
-   `config.ini`: 项目配置文件，用于数据库连接、API 密钥等设置。
-   `cold_chain_app.log`: 应用程序运行日志。
-   `prediction_logic.py` : 预测处理代码实现，具体对过期温度湿度标志物进行处理。
-   `warning_processing.py`: 负责处理各类风险警告并将写入数据库。
-   `README.md`: 本文件，项目说明。
-   `quick_check_tvbn.py` : 测试动态预测模型的文件

## 安装

要运行本项目，请确保您安装了 Python 3，并安装所有必要的依赖库。

1.  **克隆项目**
    ```bash
    git clone https://github.com/yyysiyue1/cold_chain.git
    cd cold_chain_pyc
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 使用

1.  **配置**
    在运行项目之前，请根据您的环境修改 `config.ini` 文件，例如配置数据库连接信息。

2.  **运行项目**
    通过以下命令运行主程序：
    ```bash
    python main.py #这是运行所有代码
    ```

3.  **其他脚本**
    -   训练模型：
        ```bash
        python train_tvbn_model.py
        ```
        如若使用新的模型可以在当前文件夹下新建文件并训练好 建立新类就可以外部调用，如tvbn_predictor.py一样
      - 
    -   数据库初始化：
        ```bash
        python database_setup.py
        ```
    -    动态预测方法：
    - 动态预测方法以函数的方式放在了advanced_prediction_models.py  贡献者的动态预测方法可集中写在此处

## 贡献

- TODO：其他贡献者可以将模型加入本系统直接实现半小时动态读取数据并处理后写入数据再预警的一条龙服务
  - 重要函数：
  - 1、execute_prediction_unit(row, food_info, engine, predictor_cache)
      """
      统一预测调度单元：根据食品分类执行专属模型预测。
      predictor_cache 形如：{'tvbn': TVBNPredictor(...)}。
      返回：list[dict]，可能为空。
      """
-  更具体的可以去prediction_logic.py中找见该函数理解模仿写
- 2、find_previous_abnormal_value(order_number, rec_time, tra_code, order_tra_chain, engine, flag="化学")
-     """
    查找上一条预测值（支持温度/湿度）。
     """
- 这个是找上一条数据的含量值 对于动态预测方法会有需要
- 3、find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine):
    """
    查找上一条监测数据的时间。
    """
- 这是寻找上一条数据的时间 对于动态预测的步长会有需要

## 许可证 

杨思越  许可证


---

