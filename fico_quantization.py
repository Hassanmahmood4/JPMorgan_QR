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


def _optimal_partition(
    grouped: pd.DataFrame,
    num_buckets: int,
    objective: Literal["mse", "log_likelihood"],
) -> tuple[list[float], float]:
    scores = grouped["fico_score"].to_numpy()
    counts = grouped["count"].to_numpy()
    defaults = grouped["defaults"].to_numpy()
    m = len(scores)

    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1")
    if num_buckets > m:
        raise ValueError("num_buckets cannot exceed number of unique FICO scores")

    seg_cost = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            if objective == "mse":
                seg_cost[i, j] = _segment_mse(scores, counts, i, j + 1)
            else:
                seg_cost[i, j] = _segment_log_likelihood(counts, defaults, i, j + 1)

    neg_inf = -np.inf
    pos_inf = np.inf
    dp = np.full((m + 1, num_buckets + 1), pos_inf if objective == "mse" else neg_inf)
    split = np.zeros((m + 1, num_buckets + 1), dtype=int)
    dp[0, 0] = 0.0

    for b in range(1, num_buckets + 1):
        for i in range(b, m + 1):
            for j in range(b - 1, i):
                cost = dp[j, b - 1] + seg_cost[j, i - 1]
                if objective == "mse":
                    if cost < dp[i, b]:
                        dp[i, b] = cost
                        split[i, b] = j
                else:
                    if cost > dp[i, b]:
                        dp[i, b] = cost
                        split[i, b] = j

    boundary_indices: list[int] = []
    i, b = m, num_buckets
    while b > 0:
        j = int(split[i, b])
        if j > 0:
            boundary_indices.append(j - 1)
        i, b = j, b - 1

    boundaries = sorted({float(scores[idx]) for idx in boundary_indices})
    return boundaries, float(dp[m, num_buckets])


def generate_mse_buckets(
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    df: pd.DataFrame | None = None,
) -> BucketResult:
    """Find bucket boundaries that minimize mean squared error."""
    if df is None:
        df = load_data()
    grouped = _aggregate_fico(df)
    boundaries, mse = _optimal_partition(grouped, num_buckets, "mse")
    return BucketResult(method="mse", num_buckets=num_buckets, boundaries=boundaries, mse=mse)


def generate_log_likelihood_buckets(
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    df: pd.DataFrame | None = None,
) -> BucketResult:
    """Find bucket boundaries that maximize log-likelihood of defaults."""
    if df is None:
        df = load_data()
    grouped = _aggregate_fico(df)
    boundaries, log_likelihood = _optimal_partition(grouped, num_buckets, "log_likelihood")
    return BucketResult(
        method="log_likelihood",
        num_buckets=num_buckets,
        boundaries=boundaries,
        log_likelihood=log_likelihood,
    )


def get_rating_map(method: str = "log_likelihood", num_buckets: int = DEFAULT_NUM_BUCKETS) -> BucketResult:
    global _rating_map
    if _rating_map is None or _rating_map.method != method or _rating_map.num_buckets != num_buckets:
        _rating_map = (
            generate_mse_buckets(num_buckets)
            if method == "mse"
            else generate_log_likelihood_buckets(num_buckets)
        )
    return _rating_map


def map_rating(
    fico_score: float,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    method: str = "log_likelihood",
) -> int:
    """
    Map a FICO score to a credit rating (1 = best, num_buckets = worst).
    """
    return get_rating_map(method=method, num_buckets=num_buckets).rating_for_fico(float(fico_score))


def bucket_summary(df: pd.DataFrame, bucket_result: BucketResult) -> pd.DataFrame:
    ratings = df["fico_score"].apply(bucket_result.rating_for_fico)
    return (
        df.assign(rating=ratings)
        .groupby("rating")
        .agg(
            count=("fico_score", "count"),
            fico_min=("fico_score", "min"),
            fico_max=("fico_score", "max"),
            default_rate=("default", "mean"),
        )
        .sort_index()
        .reset_index()
    )
