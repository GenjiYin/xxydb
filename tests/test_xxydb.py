import shutil
from pathlib import Path

import pandas as pd
import pytest

from xxydb import xxydb


@pytest.fixture
def tmp_db(tmp_path):
    """创建一个临时目录下的 xxydb 实例，测试结束后自动清理。"""
    db = xxydb(path=str(tmp_path / "testdb"))
    yield db
    db.close()


@pytest.fixture
def sample_df():
    """生成跨两年的示例 DataFrame。"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-06-01", "2023-12-15", "2024-03-10", "2024-07-20"]
            ),
            "code": ["000001", "000002", "000001", "000002"],
            "close": [10.0, 20.0, 30.0, 40.0],
        }
    )


# ──────────────────────────────────────────
# 基本功能
# ──────────────────────────────────────────


class TestInit:
    def test_empty_database(self, tmp_db):
        assert tmp_db.tables() == []

    def test_base_dir_created(self, tmp_db):
        assert tmp_db.base_dir.exists()


class TestWritePartitioned:
    def test_write_and_query(self, tmp_db, sample_df):
        tmp_db.write_data(
            sample_df,
            id="bar",
            date_col="date",
            partitioning="年",
            unique_together=["date", "code"],
        )
        result = tmp_db.query("SELECT * FROM bar").df()
        assert len(result) == 4
        assert set(result.columns) == {"date", "code", "close"}

    def test_partition_dirs_created(self, tmp_db, sample_df):
        tmp_db.write_data(sample_df, id="bar", date_col="date", partitioning="年")
        table_dir = tmp_db.base_dir / "bar"
        years = sorted(p.name for p in table_dir.iterdir() if p.is_dir())
        assert years == ["year=2023", "year=2024"]

    def test_month_partition(self, tmp_db, sample_df):
        tmp_db.write_data(sample_df, id="bar", date_col="date", partitioning="月")
        table_dir = tmp_db.base_dir / "bar"
        parquet_files = list(table_dir.rglob("*.parquet"))
        # 4 rows span 4 distinct year-month combos
        assert len(parquet_files) == 4


class TestWriteUnpartitioned:
    def test_write_no_partition(self, tmp_db, sample_df):
        tmp_db.write_data(
            sample_df, id="flat", date_col="date", partitioning=None
        )
        result = tmp_db.query("SELECT * FROM flat").df()
        assert len(result) == 4

    def test_single_parquet_file(self, tmp_db, sample_df):
        tmp_db.write_data(
            sample_df, id="flat", date_col="date", partitioning=None
        )
        parquet_files = list((tmp_db.base_dir / "flat").rglob("*.parquet"))
        assert len(parquet_files) == 1


class TestDedup:
    def test_rewrite_true_keeps_new(self, tmp_db):
        df1 = pd.DataFrame(
            {"date": ["2024-01-01"], "code": ["000001"], "close": [10.0]}
        )
        df2 = pd.DataFrame(
            {"date": ["2024-01-01"], "code": ["000001"], "close": [99.0]}
        )
        tmp_db.write_data(
            df1, id="t", date_col="date", partitioning="年",
            unique_together=["date", "code"],
        )
        tmp_db.write_data(
            df2, id="t", date_col="date", partitioning="年",
            unique_together=["date", "code"], rewrite=True,
        )
        result = tmp_db.query("SELECT close FROM t").df()
        assert result["close"].iloc[0] == pytest.approx(99.0)

    def test_rewrite_false_keeps_old(self, tmp_db):
        df1 = pd.DataFrame(
            {"date": ["2024-01-01"], "code": ["000001"], "close": [10.0]}
        )
        df2 = pd.DataFrame(
            {"date": ["2024-01-01"], "code": ["000001"], "close": [99.0]}
        )
        tmp_db.write_data(
            df1, id="t", date_col="date", partitioning="年",
            unique_together=["date", "code"],
        )
        tmp_db.write_data(
            df2, id="t", date_col="date", partitioning="年",
            unique_together=["date", "code"], rewrite=False,
        )
        result = tmp_db.query("SELECT close FROM t").df()
        assert result["close"].iloc[0] == pytest.approx(10.0)


class TestDelete:
    def test_delete_removes_table(self, tmp_db, sample_df):
        tmp_db.write_data(sample_df, id="todel", date_col="date", partitioning="年")
        assert "todel" in tmp_db.tables()
        tmp_db.delete("todel")
        assert "todel" not in tmp_db.tables()
        assert not (tmp_db.base_dir / "todel").exists()

    def test_delete_nonexistent_no_error(self, tmp_db):
        tmp_db.delete("nonexistent")  # should not raise


class TestTables:
    def test_tables_lists_written(self, tmp_db, sample_df):
        tmp_db.write_data(sample_df, id="a", date_col="date", partitioning=None)
        tmp_db.write_data(sample_df, id="b", date_col="date", partitioning="年")
        assert sorted(tmp_db.tables()) == ["a", "b"]


# ──────────────────────────────────────────
# 上下文管理器
# ──────────────────────────────────────────


class TestContextManager:
    def test_with_statement(self, tmp_path):
        with xxydb(path=str(tmp_path / "ctx")) as db:
            assert db.tables() == []
            db.write_data(
                pd.DataFrame({"date": ["2024-01-01"], "v": [1]}),
                id="x",
                date_col="date",
                partitioning=None,
            )
            assert db.tables() == ["x"]
        # 连接已关闭，再次操作应报错
        with pytest.raises(Exception):
            db.query("SELECT 1")

    def test_repr(self, tmp_db, sample_df):
        tmp_db.write_data(sample_df, id="bar", date_col="date", partitioning="年")
        r = repr(tmp_db)
        assert "xxydb" in r
        assert "tables=1" in r


# ──────────────────────────────────────────
# 异常
# ──────────────────────────────────────────


class TestSchema:
    def test_auto_infer_schema(self, tmp_db, sample_df):
        """写入时不传 schema，自动推断 dtype 并记录。"""
        tmp_db.write_data(sample_df, id="bar", date_col="date", partitioning="年",
                          unique_together=["date", "code"])
        cfg = tmp_db._config["bar"]
        assert "schema" in cfg
        assert "date" in cfg["schema"]
        assert "code" in cfg["schema"]
        assert "close" in cfg["schema"]
        # 自动推断时 desc 为空
        assert cfg["schema"]["close"]["desc"] == ""
        # type 应来自 pandas dtype
        assert cfg["schema"]["close"]["type"] != ""

    def test_write_with_schema(self, tmp_db, sample_df):
        """写入时传入 schema，desc 应被正确保存。"""
        schema = {
            "date": {"desc": "交易日期"},
            "code": {"desc": "股票代码（6位）"},
            "close": {"desc": "日收盘价（元）"},
        }
        tmp_db.write_data(sample_df, id="bar", date_col="date", partitioning="年",
                          unique_together=["date", "code"], schema=schema)
        cfg = tmp_db._config["bar"]["schema"]
        assert cfg["date"]["desc"] == "交易日期"
        assert cfg["code"]["desc"] == "股票代码（6位）"
        assert cfg["close"]["desc"] == "日收盘价（元）"
        # 自动推断的 type 也应保留
        assert "type" in cfg["close"]

    def test_set_schema_updates_existing(self, tmp_db, sample_df):
        """set_schema 可以为已有表补充或更新字段描述。"""
        tmp_db.write_data(sample_df, id="bar", date_col="date", partitioning="年")
        # 初始无描述
        assert tmp_db._config["bar"]["schema"]["close"]["desc"] == ""
        # 补充描述
        tmp_db.set_schema("bar", {"close": {"desc": "日收盘价（元）"}})
        assert tmp_db._config["bar"]["schema"]["close"]["desc"] == "日收盘价（元）"
        # 原有 type 不丢失
        assert "type" in tmp_db._config["bar"]["schema"]["close"]

    def test_set_schema_nonexistent_table(self, tmp_db):
        """对不存在的表调用 set_schema 应报错。"""
        with pytest.raises(ValueError, match="不存在"):
            tmp_db.set_schema("ghost", {"x": {"desc": "test"}})

    def test_describe_returns_dataframe(self, tmp_db, sample_df):
        """describe 返回包含字段描述的 DataFrame。"""
        schema = {
            "date": {"desc": "交易日期"},
            "code": {"desc": "股票代码"},
            "close": {"desc": "收盘价"},
        }
        tmp_db.write_data(sample_df, id="bar", date_col="date", partitioning="年",
                          unique_together=["date", "code"], schema=schema)
        desc = tmp_db.describe("bar")
        assert list(desc.columns) == ["字段", "物理类型", "说明", "是否主键"]
        assert len(desc) == 3
        # 检查主键标记
        code_row = desc[desc["字段"] == "code"].iloc[0]
        assert code_row["是否主键"] is True
        assert code_row["说明"] == "股票代码"
        close_row = desc[desc["字段"] == "close"].iloc[0]
        assert close_row["是否主键"] is False

    def test_describe_nonexistent_table(self, tmp_db):
        """对不存在的表调用 describe 应报错。"""
        with pytest.raises(ValueError, match="不存在"):
            tmp_db.describe("ghost")

    def test_schema_merge_on_second_write(self, tmp_db):
        """第二次写入时 schema 应合并，不覆盖已有描述。"""
        df1 = pd.DataFrame({"date": ["2024-01-01"], "code": ["000001"], "close": [10.0]})
        schema1 = {"close": {"desc": "收盘价"}}
        tmp_db.write_data(df1, id="t", date_col="date", partitioning="年", schema=schema1)
        assert tmp_db._config["t"]["schema"]["close"]["desc"] == "收盘价"

        # 第二次写入，不传 close 的 schema，但传 code 的
        df2 = pd.DataFrame({"date": ["2024-02-01"], "code": ["000002"], "close": [20.0]})
        schema2 = {"code": {"desc": "股票代码"}}
        tmp_db.write_data(df2, id="t", date_col="date", partitioning="年", schema=schema2)
        # close 的描述应保留
        assert tmp_db._config["t"]["schema"]["close"]["desc"] == "收盘价"
        # code 的描述应更新
        assert tmp_db._config["t"]["schema"]["code"]["desc"] == "股票代码"


@pytest.fixture
def panel_df():
    """跨 2019-2021 的面板数据，多只股票。"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-12-31", "2020-01-01", "2020-06-15",
                 "2020-12-31", "2021-01-01", "2021-06-01"]
            ),
            "instrument": ["000001", "000001", "000002",
                           "000001", "000002", "000001"],
            "close": [9.0, 10.0, 20.0, 12.0, 21.0, 30.0],
        }
    )


class TestQueryFilters:
    def _write(self, db, df):
        db.write_data(df, id="bar", date_col="date", partitioning="年",
                      unique_together=["date", "instrument"])

    def test_no_filters_regression(self, tmp_db, panel_df):
        """不传 filters 时行为不变。"""
        self._write(tmp_db, panel_df)
        assert len(tmp_db.query("SELECT * FROM bar").df()) == 6

    def test_tuple_range_left_closed_right_open(self, tmp_db, panel_df):
        """tuple 区间，左闭右开，取 2020 整年。"""
        self._write(tmp_db, panel_df)
        r = tmp_db.query("SELECT * FROM bar",
                         filters={"date": ("2020-01-01", "2021-01-01")}).df()
        assert len(r) == 3
        assert r["date"].min().strftime("%Y-%m-%d") == "2020-01-01"
        assert r["date"].max().strftime("%Y-%m-%d") == "2020-12-31"

    def test_list_in_enum(self, tmp_db, panel_df):
        """list 值生成 IN 枚举。"""
        self._write(tmp_db, panel_df)
        r = tmp_db.query("SELECT * FROM bar",
                         filters={"instrument": ["000002"]}).df()
        assert len(r) == 2
        assert set(r["instrument"]) == {"000002"}

    def test_scalar_equals(self, tmp_db, panel_df):
        """标量值生成等值条件。"""
        self._write(tmp_db, panel_df)
        r = tmp_db.query("SELECT * FROM bar",
                         filters={"instrument": "000001"}).df()
        assert len(r) == 4

    def test_combined_filters(self, tmp_db, panel_df):
        """区间 + 枚举组合。"""
        self._write(tmp_db, panel_df)
        r = tmp_db.query(
            "SELECT * FROM bar",
            filters={"date": ("2020-01-01", "2021-01-01"),
                     "instrument": ["000001"]},
        ).df()
        assert len(r) == 2

    def test_filter_applies_to_all_ctes(self, tmp_db, panel_df):
        """filters 下推到视图，多个 CTE 子句无需重复 WHERE。"""
        self._write(tmp_db, panel_df)
        sql = """
            WITH a AS (SELECT instrument, AVG(close) avg_c FROM bar GROUP BY instrument),
                 b AS (SELECT instrument, MAX(close) max_c FROM bar GROUP BY instrument)
            SELECT a.instrument, a.avg_c, b.max_c
            FROM a JOIN b USING(instrument) ORDER BY instrument
        """
        r = tmp_db.query(sql, filters={"date": ("2020-01-01", "2021-01-01")}).df()
        # 只应统计 2020 的 3 行
        row1 = r[r["instrument"] == "000001"].iloc[0]
        assert row1["avg_c"] == pytest.approx(11.0)
        assert row1["max_c"] == pytest.approx(12.0)

    def test_open_ended_range(self, tmp_db, panel_df):
        """区间端点为 None 表示该端开放。"""
        self._write(tmp_db, panel_df)
        r = tmp_db.query("SELECT * FROM bar",
                         filters={"date": ("2021-01-01", None)}).df()
        assert len(r) == 2

    def test_view_restored_after_filtered_query(self, tmp_db, panel_df):
        """filters 查询后视图应还原为无过滤状态。"""
        self._write(tmp_db, panel_df)
        tmp_db.query("SELECT * FROM bar",
                     filters={"date": ("2020-01-01", "2021-01-01")}).df()
        assert len(tmp_db.query("SELECT * FROM bar").df()) == 6

    def test_missing_column_ignored(self, tmp_db, panel_df):
        """filters 中表里不存在的列被忽略。"""
        self._write(tmp_db, panel_df)
        db2 = tmp_db
        db2.write_data(pd.DataFrame({"date": pd.to_datetime(["2020-05-01"]),
                                     "val": [1]}),
                       id="other", date_col="date", partitioning="年")
        # other 表没有 instrument 列，filters 不应影响它
        r = db2.query("SELECT * FROM other",
                      filters={"instrument": ["000001"]}).df()
        assert len(r) == 1

    def test_filter_value_escaped(self, tmp_db, panel_df):
        """filters 值中的引号被转义，注入失效。"""
        self._write(tmp_db, panel_df)
        r = tmp_db.query("SELECT * FROM bar",
                         filters={"instrument": ["x' OR '1'='1"]}).df()
        assert len(r) == 0

    def test_empty_list_no_results(self, tmp_db, panel_df):
        """空枚举列表返回空结果。"""
        self._write(tmp_db, panel_df)
        r = tmp_db.query("SELECT * FROM bar",
                         filters={"instrument": []}).df()
        assert len(r) == 0


class TestErrors:
    def test_missing_date_col(self, tmp_db):
        df = pd.DataFrame({"not_date": ["2024-01-01"], "v": [1]})
        with pytest.raises(ValueError, match="日期列"):
            tmp_db.write_data(df, id="err", date_col="date")

    def test_invalid_partitioning(self, tmp_db):
        df = pd.DataFrame({"date": ["2024-01-01"], "v": [1]})
        with pytest.raises(ValueError, match="不支持的分区粒度"):
            tmp_db.write_data(df, id="err", date_col="date", partitioning="周")
