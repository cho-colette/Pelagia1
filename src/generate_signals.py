from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import SignalConfig, build_time_array, ensure_directory, save_dataframe


STATE_LABELS = {
    0: "normal_harvesting",
    1: "anomaly_inspection",
    2: "adaptive_operation",
    3: "protective_mode",
}


def simulate_petal_motion(
    t: np.ndarray,
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    noise_scale: float,
    trend_scale: float = 0.0,
) -> np.ndarray:
    base = amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad)
    harmonic = 0.25 * amplitude * np.sin(2.0 * np.pi * 2.2 * frequency_hz * t + 0.5 * phase_rad)
    trend = trend_scale * (t / max(t[-1], 1e-9))
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(t))
    return base + harmonic + trend + noise


def simulate_temperature(
    t: np.ndarray,
    baseline_temp: float,
    load_factor: float,
    anomaly_boost: float = 0.0,
) -> np.ndarray:
    drift = 1.2 * load_factor * (t / max(t[-1], 1e-9))
    oscillation = 0.4 * np.sin(2.0 * np.pi * 0.01 * t)
    noise = np.random.normal(0.0, 0.12, size=len(t))
    return baseline_temp + drift + oscillation + anomaly_boost + noise


def simulate_voltage_current(
    petal_1: np.ndarray,
    petal_2: np.ndarray,
    petal_3: np.ndarray,
    state_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    motion_combo = 0.45 * np.abs(petal_1) + 0.30 * np.abs(petal_2) + 0.25 * np.abs(petal_3)

    base_voltage = 10.0 + 5.0 * motion_combo
    base_current = 1.1 + 1.8 * motion_combo

    if state_id == 1:
        base_current *= 0.72
        base_voltage *= 0.88
    elif state_id == 2:
        base_voltage *= 0.95
        base_current *= 0.92
    elif state_id == 3:
        base_voltage *= 0.62
        base_current *= 0.45

    voltage_noise = np.random.normal(0.0, 0.18, size=len(motion_combo))
    current_noise = np.random.normal(0.0, 0.08, size=len(motion_combo))

    voltage = base_voltage + voltage_noise
    current = base_current + current_noise
    return voltage, current


def generate_state_case(config: SignalConfig, state_id: int) -> pd.DataFrame:
    t = build_time_array(config.duration_s, config.sample_rate_hz)

    if state_id == 0:
        amp = 0.18
        freq = 0.22
        phases = [0.0, 0.18, -0.12]
        noise = 0.015
        trend = 0.00
        anomaly_boost = 0.0
    elif state_id == 1:
        amp = 0.12
        freq = 0.19
        phases = [0.0, 0.72, -0.55]
        noise = 0.03
        trend = 0.015
        anomaly_boost = 1.8
    elif state_id == 2:
        amp = 0.09
        freq = 0.15
        phases = [0.0, 0.28, -0.18]
        noise = 0.02
        trend = 0.025
        anomaly_boost = 0.6
    else:
        amp = 0.06
        freq = 0.10
        phases = [0.0, 0.10, -0.08]
        noise = 0.01
        trend = -0.01
        anomaly_boost = 2.5

    p1 = simulate_petal_motion(t, amp, freq, phases[0], noise, trend_scale=trend)
    p2 = simulate_petal_motion(t, amp * 0.96, freq * 1.03, phases[1], noise, trend_scale=trend)
    p3 = simulate_petal_motion(t, amp * 1.04, freq * 0.98, phases[2], noise, trend_scale=trend)

    voltage, current = simulate_voltage_current(p1, p2, p3, state_id)
    power = voltage * current
    energy = np.cumsum(power) / config.sample_rate_hz
    temperature = simulate_temperature(
        t=t,
        baseline_temp=21.0,
        load_factor=float(np.mean(np.abs(power)) / max(np.max(np.abs(power)), 1e-9)),
        anomaly_boost=anomaly_boost,
    )

    fold_state = np.zeros_like(t)

    return pd.DataFrame(
        {
            "time_s": t,
            "petal_1": p1,
            "petal_2": p2,
            "petal_3": p3,
            "voltage_v": voltage,
            "current_a": current,
            "power_w": power,
            "energy_j": energy,
            "temperature_c": temperature,
            "fold_state": fold_state,
            "state_id": state_id,
            "state_label": STATE_LABELS[state_id],
        }
    )


def generate_dataset(
    config: SignalConfig,
    samples_per_state: int = 12,
    output_dir: str | Path = "data/synthetic_signals",
) -> pd.DataFrame:
    np.random.seed(config.random_seed)
    frames = []

    for state_id in STATE_LABELS:
        for sample_idx in range(samples_per_state):
            case = generate_state_case(config, state_id)
            case["case_id"] = f"state_{state_id:01d}_sample_{sample_idx:03d}"
            frames.append(case)

    dataset = pd.concat(frames, ignore_index=True)

    output_dir = ensure_directory(output_dir)
    save_dataframe(dataset, output_dir / "pelagia_raw_signals.csv")
    return dataset


if __name__ == "__main__":
    cfg = SignalConfig(duration_s=120.0, sample_rate_hz=10.0, random_seed=42)
    df = generate_dataset(cfg, samples_per_state=12)
    print(df.head())
    print(df["state_label"].value_counts())
