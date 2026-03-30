from __future__ import annotations

from generate_signals import generate_dataset
from extract_features import build_feature_table
from train_classifier import train_classifier
from train_anomaly_model import train_anomaly_model
from streaming_update import run_streaming_update_demo
from plot_branches import plot_decision_tree_figure, plot_confusion_matrix_heatmap
from utils import SignalConfig


def main() -> dict:
    config = SignalConfig(duration_s=120.0, sample_rate_hz=10.0, random_seed=42)

    print("1. Generating synthetic signals...")
    dataset = generate_dataset(config=config, samples_per_state=12)
    print(f"Generated rows: {len(dataset)}")
    print(dataset[['case_id', 'state_label']].drop_duplicates()['state_label'].value_counts().sort_index())
    print()

    print("2. Extracting features...")
    feature_df = build_feature_table(sample_rate_hz=config.sample_rate_hz)
    print(f"Feature rows: {len(feature_df)}")
    print()

    print("3. Training classifier...")
    classifier_outputs = train_classifier()
    print("Classifier metrics:")
    print(classifier_outputs["metrics"])
    print()

    print("4. Training anomaly model...")
    anomaly_outputs = train_anomaly_model()
    print("Anomaly summary:")
    print(anomaly_outputs["summary"])
    print()

    if "results" in anomaly_outputs:
        print("Most anomalous cases:")
        print(
            anomaly_outputs["results"]
            .sort_values("anomaly_score", ascending=True)
            [["case_id", "state_label", "anomaly_score", "flag_label"]]
            .head(10)
        )
        print()

    print("5. Running streaming update demo...")
    streaming_outputs = run_streaming_update_demo()
    print("Latest streaming metrics:")
    print(streaming_outputs["metrics_df"].tail())
    print()

    print("6. Saving figures...")
    tree_path = plot_decision_tree_figure()
    cm_path = plot_confusion_matrix_heatmap()
    print(f"Saved: {tree_path}")
    print(f"Saved: {cm_path}")
    print()

    print("Pipeline complete.")

    return {
        "config": config,
        "dataset": dataset,
        "feature_df": feature_df,
        "classifier_outputs": classifier_outputs,
        "anomaly_outputs": anomaly_outputs,
        "streaming_outputs": streaming_outputs,
        "tree_path": tree_path,
        "cm_path": cm_path,
    }


if __name__ == "__main__":
    main()
