from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    duration_s: float = 120.0
    sample_rate_hz: float = 10.0
    random_seed: int = 42


def build_time_array(duration_s: float, sample_rate_hz: float) -> np.ndarray:
    n_samples = int(duration_s * sample_rate_hz)
    return np.arange(n_samples) / sample_rate_hz


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.zeros_like(values)
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy()
    )


def safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt)


def normalise_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    denom = np.max(np.abs(values))
    if denom == 0:
        return values.copy()
    return values / denom


def save_dataframe(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> None:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=index)


def describe_dataframe(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return df.describe().T
    return df[list(columns)].describe().T
