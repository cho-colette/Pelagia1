from __future__ import annotations

from generate_signals import generate_dataset
from extract_features import build_feature_table
from train_classifier import train_classifier
from train_anomaly_model import train_anomaly_model
from streaming_update import run_streaming_update_demo
from plot_branches import plot_decision_tree_figure, plot_confusion_matrix_heatmap
from utils import SignalConfig


def main() -> None:
    config = SignalConfig(duration_s=120.0, sample_rate_hz=10.0, random_seed=42)

    print("1. Generating synthetic signals...")
    generate_dataset(config=config, samples_per_state=12)

    print("2. Extracting features...")
    build_feature_table(sample_rate_hz=config.sample_rate_hz)

    print("3. Training classifier...")
    classifier_outputs = train_classifier()
    print(classifier_outputs["metrics"])

    print("4. Training anomaly model...")
    anomaly_outputs = train_anomaly_model()
    print(anomaly_outputs["summary"])

    print("5. Running streaming update demo...")
    streaming_outputs = run_streaming_update_demo()
    print(streaming_outputs["metrics_df"].tail())

    print("6. Saving figures...")
    tree_path = plot_decision_tree_figure()
    cm_path = plot_confusion_matrix_heatmap()
    print(f"Saved: {tree_path}")
    print(f"Saved: {cm_path}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
