"""FICO score quantization and rating map for Task 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "Task 3 and 4 Loan Data.csv"
DEFAULT_NUM_BUCKETS = 10
EPS = 1e-10

_rating_map = None


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@dataclass
class BucketResult:
    """Optimal bucket boundaries and metadata."""

    method: str
    num_buckets: int
    boundaries: list[float]
    mse: float | None = None
    log_likelihood: float | None = None

    def rating_for_fico(self, fico: float) -> int:
        """Return rating where 1 = best credit (highest FICO bucket)."""
        bucket_idx = int(np.searchsorted(self.boundaries, fico, side="right"))
        return self.num_buckets - bucket_idx


def _aggregate_fico(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate FICO scores for efficient DP."""
    grouped = (
        df.groupby("fico_score", as_index=False)
        .agg(count=("default", "size"), defaults=("default", "sum"))
        .sort_values("fico_score")
        .reset_index(drop=True)
    )
    return grouped


def _segment_mse(scores: np.ndarray, counts: np.ndarray, start: int, end: int) -> float:
    seg_scores = scores[start:end]
    seg_counts = counts[start:end]
    total = seg_counts.sum()
    if total == 0:
        return 0.0
    mean = np.sum(seg_scores * seg_counts) / total
    return float(np.sum(seg_counts * (seg_scores - mean) ** 2))


def _segment_log_likelihood(counts: np.ndarray, defaults: np.ndarray, start: int, end: int) -> float:
    n = int(counts[start:end].sum())
    if n == 0:
        return 0.0
    k = int(defaults[start:end].sum())
    p = np.clip(k / n, EPS, 1 - EPS)
    return float(k * np.log(p) + (n - k) * np.log(1 - p))

