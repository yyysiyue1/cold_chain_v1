# H:\cold_chain\db\database_setup.py
from __future__ import annotations
from pathlib import Path
from configparser import ConfigParser
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError
import os
import urllib.parse

# ---------------------------
# 1) 定位 config.ini（多路径兜底）
# ---------------------------
def _load_config() -> tuple[ConfigParser, Path]:
    candidates: list[Path] = []
    env_path = os.getenv("COLD_CHAIN_CONFIG")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve()             # .../db/database_setup.py
    project_root = here.parents[1]              # .../cold_chain
    candidates += [
        here.with_name("config.ini"),           # .../db/config.ini   ← 你现在的路径
        project_root / "config.ini",            # 兼容老路径
        Path.cwd() / "db" / "config.ini",       # 从根运行时的相对路径
        Path.cwd() / "config.ini",
    ]

    cfg = ConfigParser()
    for p in candidates:
        if p.is_file():
            cfg.read(p, encoding="utf-8")
            if not cfg.has_section("database"):
                # 找到了文件但缺少 [database]
                raise RuntimeError(f"Loaded config {p} but missing [database] section")
            print(f"[database_setup] Using config: {p}")
            return cfg, p
    raise FileNotFoundError(
        "config.ini not found. Tried:\n" + "\n".join(str(x) for x in candidates)
    )

_config, _cfg_path = _load_config()

# ---------------------------
# 2) 读取配置（字段名区分）
#    注意你的 ini 用的是 db_name，而不是 name
# ---------------------------
db_user = _config.get("database", "user")
raw_password = _config.get("database", "password")
db_password = urllib.parse.quote_plus(raw_password)  # 转义特殊字符
db_host = _config.get("database", "host")
db_port = _config.getint("database", "port", fallback=3306)
db_name = _config.get("database", "db_name")         # 和 ini 对齐

# 表名常量
warning_table_name = "early_warning_information"
prediction_table_name = "risk_prediction_results"

_engine = None  # 模块级缓存

def connect_to_db():
    """建立并测试数据库连接，返回 engine；失败返回 None。"""
    global _engine
    try:
        url = URL.create(
            drivername="mysql+pymysql",
            username=db_user,
            password=raw_password,   # 这里交给 SQLAlchemy 处理，已能安全转义
            host=db_host,
            port=db_port,
            database=db_name,
        )
        _engine = create_engine(url, pool_pre_ping=True, future=True)

        # 测试连接
        with _engine.connect() as conn:
            print("✅ 数据库连接成功!")

        # 检查表（指定 schema 更稳）
        inspector = inspect(_engine)
        def _check(tbl: str):
            exists = inspector.has_table(tbl, schema=db_name)
            print(f"{'✅' if exists else '⚠️'} 表 {tbl} {'存在' if exists else '不存在'}。")
            return exists

        _check(warning_table_name)
        _check(prediction_table_name)

        return _engine

    except OperationalError as e:
        print(f"❌ 数据库连接失败或数据库不存在: {e}")
        _engine = None
        return None
    except Exception as e:
        print(f"❌ 数据库连接时发生未知错误: {e}")
        _engine = None
        return None

# 便于外部 import 使用的只读属性
engine = property(lambda: _engine)

if __name__ == "__main__":
    print("Attempting to connect to the database...")
    eng = connect_to_db()
    print("Database engine initialized successfully." if eng else "Failed to initialize database engine.")
