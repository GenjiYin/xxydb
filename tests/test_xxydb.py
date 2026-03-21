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


class TestErrors:
    def test_missing_date_col(self, tmp_db):
        df = pd.DataFrame({"not_date": ["2024-01-01"], "v": [1]})
        with pytest.raises(ValueError, match="日期列"):
            tmp_db.write_data(df, id="err", date_col="date")

    def test_invalid_partitioning(self, tmp_db):
        df = pd.DataFrame({"date": ["2024-01-01"], "v": [1]})
        with pytest.raises(ValueError, match="不支持的分区粒度"):
            tmp_db.write_data(df, id="err", date_col="date", partitioning="周")
