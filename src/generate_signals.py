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


def _safe_time_scale(t: np.ndarray) -> np.ndarray:
    return t / max(float(t[-1]), 1e-9)


def add_sensor_artifacts(
    signal: np.ndarray,
    rng: np.random.Generator,
    drift_scale: float = 0.0,
    spike_probability: float = 0.15,
    spike_scale: float = 0.0,
    dropout_probability: float = 0.08,
    dropout_scale: float = 0.0,
) -> np.ndarray:
    y = np.array(signal, dtype=float, copy=True)
    n = len(y)

    if n == 0:
        return y

    if drift_scale > 0.0:
        endpoint = rng.normal(0.0, drift_scale)
        y += np.linspace(0.0, endpoint, n)

    if spike_scale > 0.0 and rng.random() < spike_probability and n > 12:
        start = int(rng.integers(5, n - 5))
        width = int(rng.integers(2, min(8, n - start)))
        y[start : start + width] += rng.normal(spike_scale, 0.35 * spike_scale, size=width)

    if dropout_scale > 0.0 and rng.random() < dropout_probability and n > 12:
        start = int(rng.integers(5, n - 5))
        width = int(rng.integers(2, min(8, n - start)))
        y[start : start + width] -= abs(rng.normal(dropout_scale, 0.30 * dropout_scale, size=width))

    return y


def simulate_petal_motion(
    t: np.ndarray,
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    noise_scale: float,
    rng: np.random.Generator,
    trend_scale: float = 0.0,
    drift_scale: float = 0.0,
    spike_scale: float = 0.0,
) -> np.ndarray:
    tau = _safe_time_scale(t)

    base = amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad)
    harmonic = 0.25 * amplitude * np.sin(2.0 * np.pi * 2.2 * frequency_hz * t + 0.5 * phase_rad)
    low_freq_mod = 0.10 * amplitude * np.sin(2.0 * np.pi * 0.02 * t + 0.3 * phase_rad)
    trend = trend_scale * tau
    noise = rng.normal(loc=0.0, scale=noise_scale, size=len(t))

    signal = base + harmonic + low_freq_mod + trend + noise
    signal = add_sensor_artifacts(
        signal,
        rng=rng,
        drift_scale=drift_scale,
        spike_probability=0.18,
        spike_scale=spike_scale,
        dropout_probability=0.06,
        dropout_scale=0.45 * spike_scale if spike_scale > 0.0 else 0.0,
    )
    return signal


def simulate_temperature(
    t: np.ndarray,
    baseline_temp: float,
    load_factor: float,
    rng: np.random.Generator,
    anomaly_boost: float = 0.0,
    temp_noise_scale: float = 0.12,
    drift_scale: float = 0.25,
) -> np.ndarray:
    tau = _safe_time_scale(t)

    drift = 1.2 * load_factor * tau
    oscillation = 0.4 * np.sin(2.0 * np.pi * 0.01 * t)
    ambient_cycle = 0.15 * np.sin(2.0 * np.pi * 0.004 * t + 0.7)
    noise = rng.normal(0.0, temp_noise_scale, size=len(t))

    temp = baseline_temp + drift + oscillation + ambient_cycle + anomaly_boost + noise
    temp = add_sensor_artifacts(
        temp,
        rng=rng,
        drift_scale=drift_scale,
        spike_probability=0.10,
        spike_scale=0.35,
        dropout_probability=0.04,
        dropout_scale=0.20,
    )
    return temp


def simulate_voltage_current(
    petal_1: np.ndarray,
    petal_2: np.ndarray,
    petal_3: np.ndarray,
    state_id: int,
    rng: np.random.Generator,
    voltage_noise_scale: float = 0.18,
    current_noise_scale: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    motion_combo = 0.45 * np.abs(petal_1) + 0.30 * np.abs(petal_2) + 0.25 * np.abs(petal_3)

    base_voltage = 10.0 + 5.0 * motion_combo
    base_current = 1.1 + 1.8 * motion_combo

    if state_id == 1:
        base_current *= 0.82
        base_voltage *= 0.92
    elif state_id == 2:
        base_voltage *= 0.96
        base_current *= 0.94
    elif state_id == 3:
        base_voltage *= 0.72
        base_current *= 0.58

    base_voltage *= rng.normal(1.0, 0.02)
    base_current *= rng.normal(1.0, 0.03)

    voltage_noise = rng.normal(0.0, voltage_noise_scale, size=len(motion_combo))
    current_noise = rng.normal(0.0, current_noise_scale, size=len(motion_combo))

    voltage = base_voltage + voltage_noise
    current = base_current + current_noise

    voltage = add_sensor_artifacts(
        voltage,
        rng=rng,
        drift_scale=0.12,
        spike_probability=0.10,
        spike_scale=0.30,
        dropout_probability=0.08,
        dropout_scale=0.45,
    )
    current = add_sensor_artifacts(
        current,
        rng=rng,
        drift_scale=0.05,
        spike_probability=0.08,
        spike_scale=0.10,
        dropout_probability=0.08,
        dropout_scale=0.16,
    )

    return voltage, current


def _sample_state_parameters(state_id: int, rng: np.random.Generator) -> dict[str, float | list[float]]:
    if state_id == 0:
        return {
            "amp": float(rng.normal(0.18, 0.018)),
            "freq": float(rng.normal(0.22, 0.015)),
            "phases": [
                float(rng.normal(0.0, 0.06)),
                float(rng.normal(0.18, 0.08)),
                float(rng.normal(-0.12, 0.08)),
            ],
            "noise": float(abs(rng.normal(0.018, 0.004))),
            "trend": float(rng.normal(0.000, 0.004)),
            "anomaly_boost": float(rng.normal(0.0, 0.08)),
            "petal_drift": 0.010,
            "petal_spike": 0.015,
        }
    if state_id == 1:
        return {
            "amp": float(rng.normal(0.135, 0.018)),
            "freq": float(rng.normal(0.195, 0.016)),
            "phases": [
                float(rng.normal(0.0, 0.08)),
                float(rng.normal(0.72, 0.12)),
                float(rng.normal(-0.55, 0.12)),
            ],
            "noise": float(abs(rng.normal(0.030, 0.006))),
            "trend": float(rng.normal(0.015, 0.006)),
            "anomaly_boost": float(rng.normal(1.6, 0.25)),
            "petal_drift": 0.018,
            "petal_spike": 0.025,
        }
    if state_id == 2:
        return {
            "amp": float(rng.normal(0.10, 0.015)),
            "freq": float(rng.normal(0.155, 0.014)),
            "phases": [
                float(rng.normal(0.0, 0.06)),
                float(rng.normal(0.28, 0.08)),
                float(rng.normal(-0.18, 0.08)),
            ],
            "noise": float(abs(rng.normal(0.022, 0.005))),
            "trend": float(rng.normal(0.022, 0.006)),
            "anomaly_boost": float(rng.normal(0.7, 0.18)),
            "petal_drift": 0.015,
            "petal_spike": 0.018,
        }

    return {
        "amp": float(rng.normal(0.07, 0.012)),
        "freq": float(rng.normal(0.105, 0.012)),
        "phases": [
            float(rng.normal(0.0, 0.05)),
            float(rng.normal(0.10, 0.06)),
            float(rng.normal(-0.08, 0.06)),
        ],
        "noise": float(abs(rng.normal(0.014, 0.003))),
        "trend": float(rng.normal(-0.010, 0.004)),
        "anomaly_boost": float(rng.normal(2.2, 0.30)),
        "petal_drift": 0.012,
        "petal_spike": 0.012,
    }


def generate_state_case(config: SignalConfig, state_id: int, sample_idx: int = 0) -> pd.DataFrame:
    t = build_time_array(config.duration_s, config.sample_rate_hz)

    # Deterministic default for reproducible paper results.
    # For a more realistic stochastic mode later, you could use:
    # rng = np.random.default_rng()
    case_seed = int(config.random_seed + 1000 * state_id + sample_idx)
    rng = np.random.default_rng(case_seed)

    params = _sample_state_parameters(state_id, rng)

    amp = float(params["amp"])
    freq = float(params["freq"])
    phases = list(params["phases"])
    noise = float(params["noise"])
    trend = float(params["trend"])
    anomaly_boost = float(params["anomaly_boost"])
    petal_drift = float(params["petal_drift"])
    petal_spike = float(params["petal_spike"])

    p1 = simulate_petal_motion(
        t,
        amp,
        freq,
        phases[0],
        noise,
        rng=rng,
        trend_scale=trend,
        drift_scale=petal_drift,
        spike_scale=petal_spike,
    )
    p2 = simulate_petal_motion(
        t,
        amp * rng.normal(0.96, 0.02),
        freq * rng.normal(1.03, 0.015),
        phases[1],
        noise,
        rng=rng,
        trend_scale=trend,
        drift_scale=petal_drift,
        spike_scale=petal_spike,
    )
    p3 = simulate_petal_motion(
        t,
        amp * rng.normal(1.04, 0.02),
        freq * rng.normal(0.98, 0.015),
        phases[2],
        noise,
        rng=rng,
        trend_scale=trend,
        drift_scale=petal_drift,
        spike_scale=petal_spike,
    )

    voltage, current = simulate_voltage_current(p1, p2, p3, state_id, rng=rng)
    power = voltage * current
    energy = np.cumsum(power) / config.sample_rate_hz

    temperature = simulate_temperature(
        t=t,
        baseline_temp=float(rng.normal(21.0, 0.25)),
        load_factor=float(np.mean(np.abs(power)) / max(np.max(np.abs(power)), 1e-9)),
        rng=rng,
        anomaly_boost=anomaly_boost,
        temp_noise_scale=0.14 if state_id in (1, 3) else 0.12,
        drift_scale=0.30 if state_id in (1, 3) else 0.18,
    )

    fold_state = np.zeros_like(t)
    if state_id == 3:
        fold_state[:] = 1.0
    elif state_id == 2:
        fold_state[int(0.65 * len(t)) :] = 0.5

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
    samples_per_state: int = 120,
    output_dir: str | Path = "data/synthetic_signals",
) -> pd.DataFrame:
    np.random.seed(config.random_seed)
    frames = []

    for state_id in STATE_LABELS:
        for sample_idx in range(samples_per_state):
            case = generate_state_case(config, state_id, sample_idx=sample_idx)
            case["case_id"] = f"state_{state_id:01d}_sample_{sample_idx:03d}"
            frames.append(case)

    dataset = pd.concat(frames, ignore_index=True)

    output_dir = ensure_directory(output_dir)
    save_dataframe(dataset, output_dir / "pelagia_raw_signals.csv")
    return dataset


if __name__ == "__main__":
    cfg = SignalConfig(duration_s=120.0, sample_rate_hz=10.0, random_seed=42)
    df = generate_dataset(cfg, samples_per_state=120)
    print(df.head())
    print(df["state_label"].value_counts())
