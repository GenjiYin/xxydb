"""
xxydb — A股轻量数据库封装

基于 Parquet (Hive分区) + DuckDB，提供简洁的写入/查询 API。
"""

import json
import os
import re
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

    def __init__(
        self,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_dir = Path(path) if path else Path.cwd()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._config_path = self.base_dir / "tables_config.json"
        self._config = self._load_config()
        self._con = duckdb.connect()  # 内存连接，避免多进程文件锁冲突
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

        # 注册内置 SQL 算子（中性化等），随连接生效
        self._register_macros()

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
        schema: Optional[dict] = None,
    ):
        """
        将 DataFrame 写入存储，支持历史批量和增量更新。

        参数:
            data:             要写入的 DataFrame
            id:               表名 / 存储 ID
            date_col:         日期列名，默认 "date"
            partitioning:     分区粒度，"年" / "月" / "日" / None(不分区)
            unique_together:  主键列表，指定后自动去重；None 则不去重
            rewrite:          True = 全量覆盖(删除旧数据后写入新数据)；False = 增量合并(有冲突时保留新值)
            schema:           字段描述，如 {"close": {"desc": "收盘价", "type": "float"}}
                              未提供时自动从 DataFrame 推断类型
        """
        if date_col not in data.columns:
            raise ValueError(f"日期列 '{date_col}' 不存在于 DataFrame 中")

        if partitioning is not None:
            part_key = self._PART_MAP.get(partitioning)
            if part_key is None:
                raise ValueError(f"不支持的分区粒度: {partitioning}，可选: 年、月、日")
        else:
            part_key = None

        # 更新配置（含字段描述）
        merged_schema = self._infer_schema(data)
        if schema:
            for col, info in schema.items():
                if col in merged_schema:
                    merged_schema[col].update(info)
                else:
                    merged_schema[col] = info
        self._ensure_config(id, date_col, part_key, unique_together, merged_schema)

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
        if rewrite or not parquet_path.exists():
            # rewrite=True：全量覆盖，直接用新数据，不读旧数据
            df_merged = df_new
        else:
            # rewrite=False：增量合并，新数据优先覆盖旧数据中的冲突行
            df_old = pd.read_parquet(parquet_path)
            if unique_together:
                df_merged = self._merge_dedup(df_old, df_new, unique_together)
            else:
                df_merged = pd.concat([df_old, df_new], ignore_index=True)

        if unique_together:
            df_merged = df_merged.sort_values(unique_together).reset_index(drop=True)
        else:
            df_merged = df_merged.reset_index(drop=True)

        df_merged.to_parquet(parquet_path, index=False, compression="zstd")
        print(f"  [写入] {parquet_path}  ({len(df_merged)} 行)")

    def query(self, sql: str, filters: Optional[dict] = None):
        """
        执行 SQL 查询，返回 DuckDB 的结果对象。
        调用 .df() 可转为 pandas DataFrame。

        参数:
            sql:      SQL 查询语句
            filters:  列筛选条件字典，会自动下推到所有引用了该列的表，
                      无需在 SQL（含每个 CTE 子句）里重复书写 WHERE。

                      值的类型决定筛选语义：
                        - tuple (起, 止)：区间，双闭（含两端），即 col >= 起 AND col <= 止
                                          任一端传 None 表示该端开放，如 ("2020-01-01", None)
                        - list  [a, b]   ：枚举，即 col IN (a, b)
                        - 标量            ：等值，即 col = 值

                      示例：
                        db.query("WITH t AS (...) SELECT ...", filters={
                            "date": ("2020-01-01", "2020-12-31"),  # 取 2020 一整年
                            "instrument": ["000001", "000002"],
                        })

                      筛选只作用于实际包含该列的表，不含该列的表不受影响。
        """
        if not filters:
            return self._con.execute(sql)

        affected = []
        try:
            for table_id in self._config:
                if not (self.base_dir / table_id).exists():
                    continue
                cols = self._table_columns(table_id)
                where = self._build_filter_where(filters, cols)
                if where:
                    self._create_view(table_id, where_clause=where)
                    affected.append(table_id)
            # 完整物化结果，避免还原视图后惰性取数读到已还原的视图
            arrow_tbl = self._con.execute(sql).fetch_arrow_table()
        finally:
            for table_id in affected:
                self._create_view(table_id)
        return self._con.from_arrow(arrow_tbl)

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

    def set_schema(self, id: str, schema: dict):
        """
        为已有表设置或更新字段描述，无需重新写入数据。

        参数:
            id:      表名
            schema:  字段描述，如 {"close": {"desc": "收盘价", "type": "float"}}
                     已有字段的描述会被合并更新，新字段会追加。
        """
        if id not in self._config:
            raise ValueError(f"表 '{id}' 不存在")
        existing = self._config[id].get("schema", {})
        for col, info in schema.items():
            if col in existing:
                existing[col].update(info)
            else:
                existing[col] = info
        self._config[id]["schema"] = existing
        self._save_config()

    def describe(self, id: str) -> pd.DataFrame:
        """
        返回指定表的字段描述信息。

        返回 DataFrame 包含: 字段、物理类型、说明、是否主键。
        """
        cfg = self._config.get(id)
        if cfg is None:
            raise ValueError(f"表 '{id}' 不存在")

        folder = self.base_dir / id
        sample = next(folder.rglob("*.parquet"), None)
        if sample is None:
            return pd.DataFrame(columns=["字段", "物理类型", "说明", "是否主键"])

        sample_path = str(sample).replace("\\", "/")
        cols = self._con.execute(
            f"SELECT name, type FROM parquet_schema('{sample_path}') "
            "WHERE num_children IS NULL"
        ).fetchall()

        schema = cfg.get("schema", {})
        unique_keys = cfg.get("unique_keys", [])
        rows = []
        for name, dtype in cols:
            info = schema.get(name, {})
            rows.append({
                "字段": name,
                "物理类型": dtype,
                "说明": info.get("desc", ""),
                "是否主键": name in unique_keys,
            })
        return pd.DataFrame(rows)

    def ask(
        self,
        question: str,
        *,
        return_df: bool = True,
        model: Optional[str] = None,
    ):
        """
        用自然语言查询数据库，AI 自动生成 SQL 并执行。

        参数:
            question:   自然语言问题，如 "2024年收盘价最高的前10只股票"
            return_df:  True 返回查询结果 DataFrame，False 返回生成的 SQL 字符串
            model:      模型名称，不传则使用构造函数中指定的模型
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "使用 ask() 需要安装 openai，请运行: pip install xxydb[ai]"
            )

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "未提供 API Key，请通过构造函数参数 api_key "
                "或环境变量 OPENAI_API_KEY 设置"
            )

        if not self._config:
            raise ValueError("数据库中没有任何表，请先写入数据")

        model = model or self._model
        if not model:
            raise ValueError(
                "未指定模型，请通过构造函数参数 model 或 ask(model=...) 设置"
            )

        system_prompt = self._build_schema_prompt()
        client_kwargs = {"api_key": api_key}
        base_url = self._base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        )

        raw = resp.choices[0].message.content.strip()
        sql = self._extract_sql(raw)
        self._validate_sql(sql)

        if not return_df:
            return sql

        return self._con.execute(sql).df()

    def close(self):
        """关闭 DuckDB 连接。"""
        self._con.close()

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    def _build_schema_prompt(self) -> str:
        """将所有表的 schema 元数据组装为系统提示词。"""
        lines = [
            "你是一个 DuckDB SQL 生成助手。根据用户的自然语言问题，生成准确的 SQL 查询。",
            "",
            "可用的表及其字段：",
        ]
        for table_id, cfg in self._config.items():
            lines.append(f"\n## {table_id}")
            schema = cfg.get("schema", {})
            unique_keys = cfg.get("unique_keys", [])
            for col, info in schema.items():
                dtype = info.get("type", "")
                desc = info.get("desc", "")
                pk = " [主键]" if col in unique_keys else ""
                parts = [f"  - {col}"]
                if dtype:
                    parts.append(f"({dtype})")
                if desc:
                    parts.append(f": {desc}")
                if pk:
                    parts.append(pk)
                lines.append(" ".join(parts))
        lines.extend([
            "",
            "规则：",
            "1. 只生成 SELECT 语句（可使用 WITH/CTE），禁止 INSERT/UPDATE/DELETE/DROP 等操作",
            "2. 只返回纯 SQL，不要添加解释、注释或 markdown 格式",
            "3. 使用 DuckDB SQL 语法",
            "4. 日期比较使用标准格式如 '2024-01-01'",
        ])
        return "\n".join(lines)

    @staticmethod
    def _extract_sql(text: str) -> str:
        """从 AI 返回的文本中提取 SQL 语句（兼容 markdown 代码块）。"""
        m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _validate_sql(sql: str):
        """校验 SQL 安全性：只允许 SELECT / WITH 开头，禁止修改类操作。"""
        normalized = sql.strip().upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            raise ValueError(f"AI 生成了非查询语句，已拒绝执行: {sql[:100]}")
        forbidden = [
            "INSERT", "UPDATE", "DELETE", "DROP",
            "ALTER", "CREATE", "TRUNCATE", "REPLACE",
        ]
        for kw in forbidden:
            if re.search(rf"\b{kw}\b", normalized):
                raise ValueError(f"SQL 包含禁止的操作 '{kw}'，已拒绝执行")

    @staticmethod
    def _infer_schema(df: pd.DataFrame) -> dict:
        """从 DataFrame 自动推断字段类型信息，desc 留空待人工补充。"""
        schema = {}
        for col in df.columns:
            schema[col] = {"type": str(df[col].dtype), "desc": ""}
        return schema

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
    ) -> pd.DataFrame:
        """合并去重。新数据在后，keep='last' 保留新数据（仅在 rewrite=False 时调用）。"""
        merged = pd.concat([df_old, df_new], ignore_index=True)
        return merged.drop_duplicates(subset=keys, keep="last")

    def _load_config(self) -> dict:
        if self._config_path.exists():
            with open(self._config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_config(self):
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=4)

    def _ensure_config(
        self, table_id: str, date_col: str, part_key: str, unique_together,
        schema: Optional[dict] = None,
    ):
        """如果表不在配置中则自动添加；已有表则合并更新 schema。"""
        if table_id not in self._config:
            self._config[table_id] = {
                "partition_by": part_key,
                "date_column": date_col,
                "unique_keys": unique_together or [],
                "source_folder": table_id,
                "schema": schema or {},
            }
        else:
            # 合并新 schema 到已有配置（新字段追加，已有字段补充）
            existing = self._config[table_id].get("schema", {})
            if schema:
                for col, info in schema.items():
                    if col in existing:
                        existing[col].update(info)
                    else:
                        existing[col] = info
            self._config[table_id]["schema"] = existing
        self._save_config()

    def _table_columns(self, table_id: str) -> List[str]:
        """返回某张表的原始列名（不含 hive 分区列）。"""
        folder = self.base_dir / table_id
        sample = next(folder.rglob("*.parquet"), None)
        if sample is None:
            return []
        sample_path = str(sample).replace("\\", "/")
        cols = self._con.execute(
            f"SELECT name FROM parquet_schema('{sample_path}') "
            "WHERE num_children IS NULL"
        ).fetchall()
        return [c[0] for c in cols]

    @staticmethod
    def _sql_literal(value) -> str:
        """把 Python 值转为安全的 SQL 字面量（数字裸写，其余加引号并转义）。"""
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    @classmethod
    def _build_filter_where(cls, filters: dict, cols: List[str]) -> str:
        """
        根据 filters 字典和表的实际列，构造 WHERE 条件（不含 WHERE 关键字）。
        只对表中真实存在的列生成条件，其它列忽略。

        值语义：
            tuple (起, 止) -> 区间，双闭（含两端）；端点为 None 时该端开放
            list  [...]    -> IN 枚举
            其它标量        -> 等值
        """
        conds = []
        for col, val in filters.items():
            if col not in cols:
                continue
            qcol = f'"{col}"'
            if isinstance(val, tuple):
                lo, hi = val
                if lo is not None:
                    conds.append(f"{qcol} >= {cls._sql_literal(lo)}")
                if hi is not None:
                    conds.append(f"{qcol} <= {cls._sql_literal(hi)}")
            elif isinstance(val, list):
                if not val:
                    conds.append("FALSE")  # 空枚举 -> 无结果
                else:
                    items = ", ".join(cls._sql_literal(v) for v in val)
                    conds.append(f"{qcol} IN ({items})")
            else:
                conds.append(f"{qcol} = {cls._sql_literal(val)}")
        return " AND ".join(conds)

    def _create_view(self, table_id: str, where_clause: Optional[str] = None):
        """为指定表创建/刷新 DuckDB 视图。

        where_clause 非空时，在视图定义里附加过滤条件（用于 query 的 filters 下推）。
        """
        folder = self.base_dir / table_id
        glob_pattern = str(folder / "**" / "*.parquet").replace("\\", "/")

        sample = next(folder.rglob("*.parquet"), None)
        if sample is None:
            return

        cfg = self._config.get(table_id, {})
        part_key = cfg.get("partition_by")

        where_sql = f"\n                WHERE {where_clause}" if where_clause else ""

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
                ){where_sql};
            """
        else:
            # 无分区：直接读取
            sql = f"""
                CREATE OR REPLACE VIEW {table_id} AS
                SELECT * FROM read_parquet(
                    '{glob_pattern}',
                    union_by_name = true
                ){where_sql};
            """
        self._con.execute(sql)

    def _refresh_all_views(self):
        """启动时为所有已配置的表刷新视图。"""
        for table_id in self._config:
            folder = self.base_dir / table_id
            if folder.exists():
                self._create_view(table_id)

    def _register_macros(self):
        """注册内置 SQL 算子（随连接生效，query 中可直接调用）。

        —— neutralize：因子中性化（市值 + 行业等）——
        原理为 FWL 定理：对「任意个离散哑变量 + 至多一个连续控制变量」，
        「组内去均值 + 截面过原点一元回归取残差」与完整多元 OLS 残差逐元素相等
        （非近似，仅有 ~1e-12 级浮点舍入）。是表算子（table macro），作用在整张
        （子）表上，输出原表所有列 + 一列 factor_neutral（中性化后的因子）。

        参数：
          tbl  : 输入表名字符串（可以是外层 WITH 定义的 CTE 名，如 't1'）
          y    : 因子列（被中性化）
          x1   : 连续控制变量列（如 LN(市值)）
          x2   : 离散控制变量列（行业名，字符或数字皆可，须离散）
          grp  : 截面分组键（通常是 date，逐日截面各自中性化）

        典型用法（逐日截面，对市值 + 行业中性化）：

            WITH t1 AS (
                SELECT date, instrument, factor, ln_mcap, industry FROM ...
            )
            SELECT date, instrument, factor_neutral
            FROM neutralize('t1', factor, ln_mcap, industry, date)

        注意：
          - 第一个参数是表名字符串（DuckDB 表算子只接表名，不接子查询）；
            取数逻辑写在外层 WITH，把 CTE 名字符串传进来即可。
          - 只支持一个连续控制变量；再加连续变量需矩阵求逆，超出纯 SQL 能力。
          - 传入前应过滤 y / x1 / x2 的 NULL，否则会污染组均值。
          - 某截面 x1 在各组内方差全为 0 时，NULLIF 使该日残差置 NULL（行为明确）。
        """
        self._con.execute(
            """
            CREATE OR REPLACE MACRO neutralize(tbl, y, x1, x2, grp) AS TABLE
            WITH _dm AS (
                SELECT *,
                    (y)  - AVG(y)  OVER (PARTITION BY grp, x2) AS _f_dm,
                    (x1) - AVG(x1) OVER (PARTITION BY grp, x2) AS _x_dm
                FROM query_table(tbl)
            )
            SELECT * EXCLUDE (_f_dm, _x_dm),
                _f_dm - (
                    SUM(_f_dm * _x_dm) OVER (PARTITION BY grp)
                    / NULLIF(SUM(_x_dm * _x_dm) OVER (PARTITION BY grp), 0)
                ) * _x_dm AS factor_neutral
            FROM _dm;
            """
        )
