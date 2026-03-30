from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from utils import ensure_directory, save_dataframe


FEATURE_COLUMNS = [
    "amp_petal_1",
    "amp_petal_2",
    "amp_petal_3",
    "std_petal_1",
    "std_petal_2",
    "std_petal_3",
    "phase_offset_12",
    "phase_offset_23",
    "phase_offset_13",
    "mean_abs_dp1",
    "mean_abs_dp2",
    "mean_abs_dp3",
    "mean_voltage_v",
    "mean_current_a",
    "mean_power_w",
    "std_power_w",
    "max_power_w",
    "mean_dpower",
    "max_abs_dpower",
    "rolling_power_mean_final",
    "rolling_power_std_final",
    "mean_temperature_c",
    "max_temperature_c",
    "mean_dtemp",
    "energy_final_j",
]

TARGET_COLUMN = "state_label"


def train_classifier(
    feature_csv: str | Path = "results/pelagia_features.csv",
    model_dir: str | Path = "results/models",
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(feature_csv)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    class_labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    clf = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=2,
        min_samples_split=4,
        ccp_alpha=0.001,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels,
    )

    model_dir = ensure_directory(model_dir)
    joblib.dump(clf, model_dir / "pelagia_decision_tree.joblib")

    rules_text = export_text(clf, feature_names=FEATURE_COLUMNS)
    (model_dir / "pelagia_tree_rules.txt").write_text(rules_text, encoding="utf-8")

    report_df = pd.DataFrame(report).transpose()
    save_dataframe(report_df, model_dir / "classification_report.csv")

    confusion_df = pd.DataFrame(
        confusion,
        index=class_labels,
        columns=class_labels,
    )
    save_dataframe(confusion_df, model_dir / "confusion_matrix.csv")

    predictions_df = X_test.copy()
    predictions_df["true_label"] = y_test.values
    predictions_df["predicted_label"] = y_pred
    predictions_df["correct"] = predictions_df["true_label"] == predictions_df["predicted_label"]
    save_dataframe(predictions_df, model_dir / "test_predictions.csv")

    metrics = {
        "accuracy": float(accuracy),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "classes": class_labels,
    }

    metrics_df = pd.DataFrame([metrics])
    save_dataframe(confusion_df, model_dir / "confusion_matrix.csv", index=True)

    return {
        "model": clf,
        "metrics": metrics,
        "report_df": report_df,
        "confusion_df": confusion_df,
        "rules_text": rules_text,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "class_labels": class_labels,
        "predictions_df": predictions_df,
    }


if __name__ == "__main__":
    outputs = train_classifier()
    print(outputs["metrics"])
    print()
    print(outputs["rules_text"])
