# xxydb

轻量 A 股数据库封装，基于 Parquet (Hive 分区) + DuckDB，提供简洁的写入/查询 API。

## 安装

```bash
pip install xxydb

# 如需 AI 自然语言查询功能
pip install xxydb[ai]
```

本地开发安装：

```bash
git clone https://github.com/xxydb/xxydb.git
cd xxydb
pip install -e ".[ai]"
```

## 快速开始

```python
import pandas as pd
from xxydb import xxydb

# 初始化（指定数据存储路径，AI 参数可选）
db = xxydb(path="./my_data")

# 写入数据（按年分区）
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=100),
    "code": ["000001"] * 100,
    "close": range(100),
})
db.write_data(df, id="daily_bar", date_col="date", partitioning="年",
              unique_together=["date", "code"])

# SQL 查询
result = db.query("SELECT * FROM daily_bar WHERE close > 50").df()
print(result)

# 查看所有表
print(db.tables())

# 删除表
db.delete("daily_bar")

# 关闭连接
db.close()
```

### 上下文管理器

```python
with xxydb(path="./my_data") as db:
    db.write_data(df, id="daily_bar", date_col="date")
    result = db.query("SELECT * FROM daily_bar").df()
# 连接自动关闭
```

## 分区模式

| 参数值 | 目录结构 |
|--------|---------|
| `"年"` | `table/year=2024/data.parquet` |
| `"月"` | `table/year=2024/month=01/data.parquet` |
| `"日"` | `table/year=2024/month=01/day=15/data.parquet` |
| `None` | `table/data.parquet`（不分区） |

## Schema（字段描述）

xxydb 支持为每张表管理字段描述信息，方便记录各列的含义。

### 写入时指定 schema

```python
schema = {
    "date":  {"desc": "交易日期"},
    "code":  {"desc": "股票代码（6位）"},
    "close": {"desc": "日收盘价（元）"},
}
db.write_data(df, id="daily_bar", date_col="date", partitioning="年",
              unique_together=["date", "code"], schema=schema)
```

未提供 `schema` 时，会自动从 DataFrame 推断字段类型（`desc` 留空）。手动传入的 `schema` 会与自动推断结果合并，已有字段的描述会被更新，新字段会追加。

### 单独设置 schema

对已有表补充或修改字段描述，无需重新写入数据：

```python
db.set_schema("daily_bar", {
    "close": {"desc": "日收盘价（前复权，元）"},
})
```

### 查看表结构

`describe()` 返回一个 DataFrame，包含字段名、物理类型、说明、是否主键：

```python
print(db.describe("daily_bar"))
#     字段    物理类型          说明  是否主键
# 0   date  BYTE_ARRAY      交易日期      True
# 1   code  BYTE_ARRAY  股票代码（6位）     True
# 2  close      DOUBLE   日收盘价（元）    False
```

## AI 自然语言查询

安装 `xxydb[ai]` 后，可以用自然语言直接查询数据，AI 会根据表结构自动生成 SQL 并执行。

支持所有兼容 OpenAI 协议的模型服务商（OpenAI、Deepseek、通义千问、Moonshot、Ollama 等）。

### 基本用法

```python
db = xxydb(
    path="./my_data",
    api_key="sk-xxx",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)

# 返回 DataFrame
df = db.ask("2024年收盘价最高的前10只股票")

# 只返回 SQL，不执行
sql = db.ask("2024年收盘价最高的前10只股票", return_df=False)
```

### 配置方式

AI 相关参数（`api_key`、`base_url`、`model`）支持两种配置方式：

**方式一：构造函数传参**

```python
db = xxydb("./my_data", api_key="sk-xxx", base_url="https://api.deepseek.com", model="deepseek-chat")
```

**方式二：环境变量**

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com
```

```python
db = xxydb("./my_data", model="deepseek-chat")
```

`ask()` 调用时也可以临时指定 `model` 覆盖默认值：

```python
df = db.ask("月均成交量排名", model="deepseek-reasoner")
```

### 提高准确率

AI 依赖表的 schema 信息来理解字段含义。字段描述越完善，生成的 SQL 越准确：

```python
db.set_schema("daily_bar", {
    "date":  {"desc": "交易日期"},
    "code":  {"desc": "股票代码（6位）"},
    "close": {"desc": "日收盘价（前复权，元）"},
    "vol":   {"desc": "成交量（手）"},
})

# schema 完善后，AI 能正确理解"成交量"指的是 vol 列
df = db.ask("最近一个月日均成交量最大的股票")
```

## API 参考

### `write_data(data, id, date_col="date", partitioning="年", unique_together=None, rewrite=True, schema=None)`

将 DataFrame 写入存储。

| 参数 | 说明 |
|------|------|
| `data` | 要写入的 DataFrame |
| `id` | 表名 |
| `date_col` | 日期列名，默认 `"date"` |
| `partitioning` | 分区粒度：`"年"` / `"月"` / `"日"` / `None` |
| `unique_together` | 主键列表，指定后自动去重；`None` 不去重 |
| `rewrite` | `True` 保留最新数据（覆盖），`False` 保留旧数据 |
| `schema` | 字段描述字典，如 `{"close": {"desc": "收盘价"}}` |

### `ask(question, *, return_df=True, model=None)`

用自然语言查询数据库（需安装 `xxydb[ai]`）。

| 参数 | 说明 |
|------|------|
| `question` | 自然语言问题 |
| `return_df` | `True` 返回 DataFrame，`False` 返回生成的 SQL 字符串 |
| `model` | 模型名称，不传则使用构造函数中指定的模型 |

### `query(sql, filters=None)`

执行 SQL 查询，返回 DuckDB 结果对象（调用 `.df()` 转为 DataFrame）。

`filters` 用于把列筛选条件一次性下推到所有引用了该列的表，无需在 SQL（尤其是多个 CTE 子句）里重复书写 `WHERE`。对按日期分区的表，时间条件还能触发分区裁剪，跳过范围外的 parquet 文件。

值的类型决定筛选语义：

| 值类型 | 含义 | 生成的条件 |
|--------|------|-----------|
| `tuple` `(起, 止)` | 区间，双闭（含两端） | `col >= 起 AND col <= 止` |
| `list` `[a, b]` | 枚举 | `col IN (a, b)` |
| 标量 | 等值 | `col = 值` |

区间任一端传 `None` 表示该端开放，如 `("2020-01-01", None)` 只限定起点。筛选只作用于实际包含该列的表，不含该列的表不受影响。

```python
# 取 2020 一整年的个股数据，CTE 里无需重复 WHERE date ...
sql = """
WITH ret AS (SELECT instrument, AVG(pct_change) m FROM bar GROUP BY instrument),
     vol AS (SELECT instrument, STDDEV(pct_change) s FROM bar GROUP BY instrument)
SELECT ret.instrument, ret.m, vol.s FROM ret JOIN vol USING(instrument)
"""
df = db.query(sql, filters={
    "date": ("2020-01-01", "2020-12-31"),   # 双闭，含两端，正好 2020 全年
    "instrument": ["000001", "000002"],     # 只看这两只
})
```

## 内置 SQL 算子

初始化 `xxydb` 后，连接中自动注册了若干可在 `query()` 里直接调用的算子，无需自己定义。

### `neutralize` — 因子中性化（市值 + 行业等）

对因子做横截面 OLS 回归、剥离连续变量（如市值）与离散变量（如行业）的影响，取残差作为「纯净」因子。基于 FWL 定理用纯窗口函数实现，与完整多元 OLS 残差逐元素相等（非近似，仅 ~1e-12 级浮点舍入）。

这是一个**表算子**，写在 `FROM` 里，输出原表所有列 + 一列 `factor_neutral`：

```python
sql = """
WITH t1 AS (
    SELECT d.date, d.instrument,
           d.close                    AS factor,      -- 待中性化的因子
           LN(v.total_market_cap)     AS ln_mcap,     -- 连续控制变量
           i.industry_level1_name     AS industry     -- 离散控制变量
    FROM daily_bar d
    JOIN valuation v USING(date, instrument)
    JOIN stock_industry_component i USING(date, instrument)
    WHERE v.total_market_cap > 0 AND d.close IS NOT NULL
      AND i.industry_level1_name IS NOT NULL
)
SELECT date, instrument, factor_neutral
FROM neutralize('t1', factor, ln_mcap, industry, date)
"""
df = db.query(sql).df()
```

参数：

| 参数 | 说明 |
|------|------|
| `tbl` | 输入表名字符串（可以是外层 `WITH` 定义的 CTE 名，如 `'t1'`） |
| `y` | 因子列（被中性化） |
| `x1` | 连续控制变量列（如 `LN(市值)`） |
| `x2` | 离散控制变量列（行业名，字符或数字皆可，须离散） |
| `grp` | 截面分组键（通常是 `date`，逐日截面各自中性化） |

注意：

- 第一个参数是**表名字符串**（DuckDB 表算子只接表名、不接子查询）；取数逻辑写在外层 `WITH`，把 CTE 名字符串传进来即可。
- 只支持**一个**连续控制变量；再加连续变量需矩阵求逆，超出纯 SQL 能力，应转 Python。
- 传入前应过滤 `y` / `x1` / `x2` 的 NULL，否则会污染组均值。
- 某截面 `x1` 在各组内方差全为 0 时，该日残差置 NULL（行为明确，不静默出错）。

### `tables()`

返回所有已注册的表名列表。

### `describe(id)`

返回指定表的字段描述 DataFrame（字段、物理类型、说明、是否主键）。

### `set_schema(id, schema)`

为已有表设置或更新字段描述，无需重新写入数据。

### `delete(id)`

删除指定表（数据文件、DuckDB 视图、配置）。

### `close()`

关闭 DuckDB 连接。也可通过 `with` 语句自动管理。

## License

MIT
