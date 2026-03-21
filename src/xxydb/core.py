"""
xxydb — A股轻量数据库封装

基于 Parquet (Hive分区) + DuckDB，提供简洁的写入/查询 API。
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd


class xxydb:
    """
    轻量存储管理。

    usage:
        db = xxydb(path='duckdb测试')
        db.write_data(data, id='cn_stock_bar1d', date_col='date', partitioning='年')
        result = db.query("SELECT * FROM cn_stock_bar1d WHERE date > '2024-01-01'").df()
    """

    _PART_MAP = {"年": "year", "月": "month", "日": "day"}

    def __init__(self, path: Optional[str] = None):
        self.base_dir = Path(path) if path else Path.cwd()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._config_path = self.base_dir / "tables_config.json"
        self._db_path = self.base_dir / "stock_database.duckdb"
        self._config = self._load_config()
        self._con = duckdb.connect(str(self._db_path))

        # 启动时为已有表创建视图
        self._refresh_all_views()

    # ──────────────────────────────────────────
    # 上下文管理器
    # ──────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        return f"xxydb(path={str(self.base_dir)!r}, tables={len(self._config)})"

    # ──────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────

    def write_data(
        self,
        data: pd.DataFrame,
        id: str,
        date_col: str = "date",
        partitioning: Optional[str] = "年",
        unique_together: Optional[List[str]] = None,
        rewrite: bool = True,
    ):
        """
        将 DataFrame 写入存储，支持历史批量和增量更新。

        参数:
            data:             要写入的 DataFrame
            id:               表名 / 存储 ID
            date_col:         日期列名，默认 "date"
            partitioning:     分区粒度，"年" / "月" / "日" / None(不分区)
            unique_together:  主键列表，指定后自动去重；None 则不去重
            rewrite:          True = 保留最新数据(覆盖)；False = 保留旧数据
        """
        if date_col not in data.columns:
            raise ValueError(f"日期列 '{date_col}' 不存在于 DataFrame 中")

        if partitioning is not None:
            part_key = self._PART_MAP.get(partitioning)
            if part_key is None:
                raise ValueError(f"不支持的分区粒度: {partitioning}，可选: 年、月、日")
        else:
            part_key = None

        # 更新配置
        self._ensure_config(id, date_col, part_key, unique_together)

        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        table_dir = self.base_dir / id

        if part_key is None:
            # 不分区：全部写入单个文件
            table_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = table_dir / "data.parquet"
            self._write_parquet(parquet_path, df, unique_together, rewrite)
        else:
            # 按分区分组写入
            df["_pk"] = df[date_col].apply(lambda d: self._partition_key(d, part_key))
            for pk, group in df.groupby("_pk"):
                group = group.drop(columns=["_pk"])
                sub_dir = table_dir / self._partition_dir(part_key, pk)
                sub_dir.mkdir(parents=True, exist_ok=True)
                parquet_path = sub_dir / "data.parquet"
                self._write_parquet(parquet_path, group, unique_together, rewrite)

        # 刷新该表的视图
        self._create_view(id)

    def _write_parquet(self, parquet_path, df_new, unique_together, rewrite):
        """写入单个 parquet 文件，处理合并去重。"""
        if parquet_path.exists():
            df_old = pd.read_parquet(parquet_path)
            if unique_together:
                df_merged = self._merge_dedup(df_old, df_new, unique_together, rewrite)
            else:
                df_merged = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_merged = df_new

        if unique_together:
            df_merged = df_merged.sort_values(unique_together).reset_index(drop=True)
        else:
            df_merged = df_merged.reset_index(drop=True)

        df_merged.to_parquet(parquet_path, index=False, compression="zstd")
        print(f"  [写入] {parquet_path}  ({len(df_merged)} 行)")

    def query(self, sql: str):
        """
        执行 SQL 查询，返回 DuckDB 的结果对象。
        调用 .df() 可转为 pandas DataFrame。
        """
        return self._con.execute(sql)

    def tables(self) -> list:
        """返回已注册的所有表名列表。"""
        return list(self._config.keys())

    def delete(self, id: str):
        """删除指定表：移除数据文件、DuckDB 视图和配置。"""
        folder = self.base_dir / id
        if folder.exists():
            shutil.rmtree(folder)

        try:
            self._con.execute(f"DROP VIEW IF EXISTS {id}")
        except Exception:
            pass

        if id in self._config:
            del self._config[id]
            self._save_config()

        print(f"  [删除] 表 '{id}' 已移除")

    def close(self):
        """关闭 DuckDB 连接。"""
        self._con.close()

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    @staticmethod
    def _partition_key(dt: pd.Timestamp, part_key: str):
        if part_key == "day":
            return (dt.year, dt.month, dt.day)
        if part_key == "month":
            return (dt.year, dt.month)
        return (dt.year,)

    @staticmethod
    def _partition_dir(part_key: str, pk: tuple) -> str:
        if part_key == "day":
            return f"year={pk[0]}/month={pk[1]:02d}/day={pk[2]:02d}"
        if part_key == "month":
            return f"year={pk[0]}/month={pk[1]:02d}"
        return f"year={pk[0]}"

    @staticmethod
    def _merge_dedup(
        df_old: pd.DataFrame,
        df_new: pd.DataFrame,
        keys: List[str],
        rewrite: bool,
    ) -> pd.DataFrame:
        """合并去重。rewrite=True 保留新数据，False 保留旧数据。"""
        if rewrite:
            # 新数据在后，drop_duplicates keep='last' 保留新的
            merged = pd.concat([df_old, df_new], ignore_index=True)
            return merged.drop_duplicates(subset=keys, keep="last")
        else:
            # 旧数据在前，drop_duplicates keep='first' 保留旧的
            merged = pd.concat([df_old, df_new], ignore_index=True)
            return merged.drop_duplicates(subset=keys, keep="first")

    def _load_config(self) -> dict:
        if self._config_path.exists():
            with open(self._config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_config(self):
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=4)

    def _ensure_config(
        self, table_id: str, date_col: str, part_key: str, unique_together
    ):
        """如果表不在配置中则自动添加。"""
        if table_id not in self._config:
            self._config[table_id] = {
                "partition_by": part_key,
                "date_column": date_col,
                "unique_keys": unique_together or [],
                "source_folder": table_id,
            }
            self._save_config()

    def _create_view(self, table_id: str):
        """为指定表创建/刷新 DuckDB 视图。"""
        folder = self.base_dir / table_id
        glob_pattern = str(folder / "**" / "*.parquet").replace("\\", "/")

        sample = next(folder.rglob("*.parquet"), None)
        if sample is None:
            return

        cfg = self._config.get(table_id, {})
        part_key = cfg.get("partition_by")

        if part_key:
            # 有分区：开启 hive_partitioning，但只 SELECT 原始列
            sample_path = str(sample).replace("\\", "/")
            cols = self._con.execute(
                f"SELECT name FROM parquet_schema('{sample_path}') WHERE num_children IS NULL"
            ).fetchall()
            col_names = ", ".join(f'"{c[0]}"' for c in cols)

            sql = f"""
                CREATE OR REPLACE VIEW {table_id} AS
                SELECT {col_names} FROM read_parquet(
                    '{glob_pattern}',
                    hive_partitioning = true,
                    union_by_name = true
                );
            """
        else:
            # 无分区：直接读取
            sql = f"""
                CREATE OR REPLACE VIEW {table_id} AS
                SELECT * FROM read_parquet(
                    '{glob_pattern}',
                    union_by_name = true
                );
            """
        self._con.execute(sql)

    def _refresh_all_views(self):
        """启动时为所有已配置的表刷新视图。"""
        for table_id in self._config:
            folder = self.base_dir / table_id
            if folder.exists():
                self._create_view(table_id)
