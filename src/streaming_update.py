from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from train_classifier import FEATURE_COLUMNS, TARGET_COLUMN
from utils import ensure_directory, save_dataframe


def run_streaming_update_demo(
    feature_csv: str | Path = "results/pelagia_features.csv",
    output_dir: str | Path = "results/streaming",
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(feature_csv).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    classes = np.array(sorted(df[TARGET_COLUMN].unique()))
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    clf = SGDClassifier(
        loss="log_loss",
        max_iter=1,
        tol=None,
        random_state=random_state,
    )

    records = []
    batch_size = 8
    seen = 0

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        X_batch = X[start:end]
        y_batch = y[start:end]

        if start == 0:
            clf.partial_fit(X_batch, y_batch, classes=classes)
        else:
            clf.partial_fit(X_batch, y_batch)

        seen += len(X_batch)

        y_pred = clf.predict(X[:end])
        running_accuracy = float(np.mean(y_pred == y[:end]))

        records.append(
            {
                "samples_seen": seen,
                "batch_end_index": end,
                "running_accuracy": running_accuracy,
            }
        )

    output_dir = ensure_directory(output_dir)
    metrics_df = pd.DataFrame(records)
    save_dataframe(metrics_df, output_dir / "streaming_metrics.csv")

    return {
        "model": clf,
        "metrics_df": metrics_df,
    }


if __name__ == "__main__":
    outputs = run_streaming_update_demo()
    print(outputs["metrics_df"].tail())
