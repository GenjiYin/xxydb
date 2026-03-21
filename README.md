# xxydb

轻量 A 股数据库封装，基于 Parquet (Hive 分区) + DuckDB，提供简洁的写入/查询 API。

## 安装

```bash
pip install xxydb
```

本地开发安装：

```bash
git clone https://github.com/xxydb/xxydb.git
cd xxydb
pip install -e .
```

## 快速开始

```python
import pandas as pd
from xxydb import xxydb

# 初始化（指定数据存储路径）
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

## License

MIT
