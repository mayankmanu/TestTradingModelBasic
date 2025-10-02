import os
from typing import Dict, List, Optional, Tuple, Iterator

import pandas as pd


REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


class DataStream:
    """
    Streams OHLCV CSV files from a directory in chunks (by rows).
    Maintains state for resume: current file index and row offset.
    """

    def __init__(self, data_dir: str, chunk_size: int = 10000):
        self.data_dir = data_dir
        self.chunk_size = int(chunk_size)
        self.files = self._discover_csvs(self.data_dir)
        self.file_idx = 0
        self.row_offset = 0

    @staticmethod
    def _discover_csvs(path: str) -> List[str]:
        files = []
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(".csv"):
                files.append(os.path.join(path, name))
        return files

    def state_dict(self) -> Dict:
        return {
            "file_idx": self.file_idx,
            "row_offset": self.row_offset,
            "chunk_size": self.chunk_size,
        }

    def load_state_dict(self, state: Dict) -> None:
        if not state:
            return
        self.file_idx = int(state.get("file_idx", 0))
        self.row_offset = int(state.get("row_offset", 0))
        # Respect user-provided chunk_size on init; don't override it

    def _read_file(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        # Ensure required columns exist
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        # Sort by date for consistency
        df = df.sort_values("date").reset_index(drop=True)
        # Numeric conversions
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

    def iter_chunks(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        """
        Yields (symbol, dataframe_chunk). Symbol is the file stem.
        Maintains internal indices so resume can continue from last state.
        """
        while self.file_idx < len(self.files):
            path = self.files[self.file_idx]
            symbol = os.path.splitext(os.path.basename(path))[0]
            df = self._read_file(path)
            n = len(df)

            # Yield chunks from current row_offset
            start = self.row_offset
            while start < n:
                end = min(start + self.chunk_size, n)
                chunk = df.iloc[start:end].reset_index(drop=True)
                yield symbol, chunk
                # Update state after yielding so resume continues next
                start = end
                self.row_offset = start

            # Move to next file
            self.file_idx += 1
            self.row_offset = 0

    def reset(self) -> None:
        self.file_idx = 0
        self.row_offset = 0
