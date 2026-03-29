from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    SignalConfig,
    ensure_directory,
    rolling_mean,
    rolling_std,
    safe_gradient,
    save_dataframe,
)


def estimate_amplitude(signal: np.ndarray) -> float:
    return 0.5 * (np.max(signal) - np.min(signal))


def estimate_phase_offset(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    corr = np.correlate(
        signal_a - np.mean(signal_a),
        signal_b - np.mean(signal_b),
        mode="full",
    )
    lag = np.argmax(corr) - (len(signal_a) - 1)
    return float(lag)


def summarise_case(case_df: pd.DataFrame, sample_rate_hz: float) -> dict:
    dt = 1.0 / sample_rate_hz

    p1 = case_df["petal_1"].to_numpy()
    p2 = case_df["petal_2"].to_numpy()
    p3 = case_df["petal_3"].to_numpy()
    v = case_df["voltage_v"].to_numpy()
    i = case_df["current_a"].to_numpy()
    p = case_df["power_w"].to_numpy()
    t = case_df["temperature_c"].to_numpy()

    dp1 = safe_gradient(p1, dt)
    dp2 = safe_gradient(p2, dt)
    dp3 = safe_gradient(p3, dt)
    dpower = safe_gradient(p, dt)
    dtemp = safe_gradient(t, dt)

    rolling_power_mean = rolling_mean(p, window=max(3, int(sample_rate_hz * 3)))
    rolling_power_std = rolling_std(p, window=max(3, int(sample_rate_hz * 3)))

    return {
        "case_id": case_df["case_id"].iloc[0],
        "state_id": int(case_df["state_id"].iloc[0]),
        "state_label": case_df["state_label"].iloc[0],
        "amp_petal_1": estimate_amplitude(p1),
        "amp_petal_2": estimate_amplitude(p2),
        "amp_petal_3": estimate_amplitude(p3),
        "mean_petal_1": float(np.mean(p1)),
        "mean_petal_2": float(np.mean(p2)),
        "mean_petal_3": float(np.mean(p3)),
        "std_petal_1": float(np.std(p1)),
        "std_petal_2": float(np.std(p2)),
        "std_petal_3": float(np.std(p3)),
        "phase_offset_12": estimate_phase_offset(p1, p2),
        "phase_offset_23": estimate_phase_offset(p2, p3),
        "phase_offset_13": estimate_phase_offset(p1, p3),
        "mean_abs_dp1": float(np.mean(np.abs(dp1))),
        "mean_abs_dp2": float(np.mean(np.abs(dp2))),
        "mean_abs_dp3": float(np.mean(np.abs(dp3))),
        "max_abs_dp1": float(np.max(np.abs(dp1))),
        "max_abs_dp2": float(np.max(np.abs(dp2))),
        "max_abs_dp3": float(np.max(np.abs(dp3))),
        "mean_voltage_v": float(np.mean(v)),
        "std_voltage_v": float(np.std(v)),
        "mean_current_a": float(np.mean(i)),
        "std_current_a": float(np.std(i)),
        "mean_power_w": float(np.mean(p)),
        "std_power_w": float(np.std(p)),
        "max_power_w": float(np.max(p)),
        "mean_dpower": float(np.mean(dpower)),
        "max_abs_dpower": float(np.max(np.abs(dpower))),
        "rolling_power_mean_final": float(rolling_power_mean[-1]),
        "rolling_power_std_final": float(rolling_power_std[-1]),
        "mean_temperature_c": float(np.mean(t)),
        "max_temperature_c": float(np.max(t)),
        "mean_dtemp": float(np.mean(dtemp)),
        "energy_final_j": float(case_df["energy_j"].iloc[-1]),
        "fold_state_final": float(case_df["fold_state"].iloc[-1]),
    }


def build_feature_table(
    raw_signal_csv: str | Path = "data/synthetic_signals/pelagia_raw_signals.csv",
    output_csv: str | Path = "results/pelagia_features.csv",
    sample_rate_hz: float = 10.0,
) -> pd.DataFrame:
    df = pd.read_csv(raw_signal_csv)
    feature_rows = []

    for _, case_df in df.groupby("case_id"):
        feature_rows.append(summarise_case(case_df, sample_rate_hz=sample_rate_hz))

    feature_df = pd.DataFrame(feature_rows)
    save_dataframe(feature_df, output_csv)
    return feature_df


if __name__ == "__main__":
    features = build_feature_table()
    print(features.head())
    print(features.shape)
