from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

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
) -> dict:
    df = pd.read_csv(feature_csv)

    # Train on normal cases only
    normal_df = df[df["state_label"] == "normal_harvesting"].copy()
    X_train = normal_df[ANOMALY_FEATURE_COLUMNS]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.20,
        random_state=random_state,
    )
    model.fit(X_train)

    X_all = df[ANOMALY_FEATURE_COLUMNS]
    anomaly_score = model.decision_function(X_all)
    anomaly_flag = model.predict(X_all)  # 1 normal, -1 anomaly

    results = df[["case_id", "state_label"]].copy()
    results["anomaly_score"] = anomaly_score
    results["anomaly_flag"] = anomaly_flag
    results["flag_label"] = results["anomaly_flag"].map({1: "normal_like", -1: "anomalous"})

    model_dir = ensure_directory(model_dir)
    joblib.dump(model, model_dir / "pelagia_isolation_forest.joblib")
    save_dataframe(results, model_dir / "anomaly_results.csv")

    summary = (
        results.groupby(["state_label", "flag_label"])
        .size()
        .reset_index(name="count")
    )
    save_dataframe(summary, model_dir / "anomaly_summary.csv")

    return {
        "model": model,
        "results": results,
        "summary": summary,
    }


if __name__ == "__main__":
    outputs = train_anomaly_model()
    print(outputs["summary"])
