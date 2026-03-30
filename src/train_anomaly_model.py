from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils import ensure_directory, save_dataframe


ANOMALY_FEATURE_COLUMNS = [
    "amp_petal_1",
    "amp_petal_2",
    "amp_petal_3",
    "phase_offset_12",
    "phase_offset_23",
    "phase_offset_13",
    "mean_abs_dp1",
    "mean_abs_dp2",
    "mean_abs_dp3",
    "mean_power_w",
    "std_power_w",
    "max_abs_dpower",
    "mean_temperature_c",
    "max_temperature_c",
    "mean_dtemp",
]


def train_anomaly_model(
    feature_csv: str | Path = "results/pelagia_features.csv",
    model_dir: str | Path = "results/models",
    random_state: int = 42,
    contamination: float = 0.08,
) -> dict:
    df = pd.read_csv(feature_csv)

    # Train only on the nominal operating regime
    normal_df = df[df["state_label"] == "normal_harvesting"].copy()
    X_train = normal_df[ANOMALY_FEATURE_COLUMNS].copy()
    X_all = df[ANOMALY_FEATURE_COLUMNS].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_train_scaled)

    anomaly_score = model.decision_function(X_all_scaled)
    anomaly_flag = model.predict(X_all_scaled)  # 1 normal, -1 anomaly

    results = df[["case_id", "state_label"]].copy()
    results["anomaly_score"] = anomaly_score
    results["anomaly_flag"] = anomaly_flag
    results["flag_label"] = results["anomaly_flag"].map({1: "normal_like", -1: "anomalous"})

    # Lower score means more anomalous, so sort ascending for easier inspection
    results["anomaly_rank"] = results["anomaly_score"].rank(method="dense", ascending=True).astype(int)

    model_dir = ensure_directory(model_dir)
    joblib.dump(model, model_dir / "pelagia_isolation_forest.joblib")
    joblib.dump(scaler, model_dir / "pelagia_anomaly_scaler.joblib")

    save_dataframe(results, model_dir / "anomaly_results.csv")

    summary = (
        results.groupby(["state_label", "flag_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["state_label", "flag_label"])
        .reset_index(drop=True)
    )
    save_dataframe(summary, model_dir / "anomaly_summary.csv")

    # Helpful extra export for paper/debugging
    top_anomalies = (
        results.sort_values("anomaly_score", ascending=True)
        .head(10)
        .reset_index(drop=True)
    )
    save_dataframe(top_anomalies, model_dir / "top_anomalous_cases.csv")

    return {
        "model": model,
        "scaler": scaler,
        "results": results,
        "summary": summary,
    }


if __name__ == "__main__":
    outputs = train_anomaly_model()
    print(outputs["summary"])
