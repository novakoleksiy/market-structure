"""Shared Parquet-backed cache helpers for market data fetchers."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

DEFAULT_EXPECTED_COLS = frozenset({"open", "high", "low", "close", "volume"})


class ParquetCacheAdapter:
    """Read and write OHLCV DataFrames under a structured cache root."""

    def __init__(
        self,
        root: Path,
        expected_cols: set[str] | frozenset[str] = DEFAULT_EXPECTED_COLS,
    ) -> None:
        self.root = root
        self.expected_cols = set(expected_cols)

    def path_for(self, *parts: str) -> Path:
        *parents, leaf = parts
        return self.root.joinpath(*parents, f"{leaf}.parquet")

    def read(self, *parts: str) -> pd.DataFrame | None:
        return self.read_path(self.path_for(*parts))

    def read_path(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path, engine="fastparquet")
            if not self.expected_cols.issubset(df.columns):
                path.unlink()
                return None
            return df
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def write(self, df: pd.DataFrame, *parts: str) -> Path:
        return self.write_path(self.path_for(*parts), df)

    def write_path(self, path: Path, df: pd.DataFrame) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            df.to_parquet(tmp_path, engine="fastparquet")
            os.replace(tmp_path, path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return path
