"""Tests for CSV Loader."""

import pytest
from data.collectors.csv_loader import CSVLoader


@pytest.fixture
def sample_csv(tmp_path):
    path = str(tmp_path / "test.csv")
    with open(path, "w") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for i in range(200):
            f.write(
                f"2025-01-06 09:{15 + i // 60:02d}:{i % 60:02d},"
                f"{25000 + i},{25005 + i},{24995 + i},{25002 + i},{10000 + i}\n"
            )
    return path


class TestCSVLoader:
    def test_load_basic(self, sample_csv):
        loader = CSVLoader()
        candles = loader.load(sample_csv)
        assert len(candles) == 200

    def test_to_arrays(self, sample_csv):
        loader = CSVLoader()
        candles = loader.load(sample_csv)
        c, h, lo, o, v = loader.to_arrays(candles)
        assert len(c) == 200
        assert c[0] == 25002.0

    def test_missing_file(self):
        loader = CSVLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.csv")
