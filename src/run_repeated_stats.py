from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import SignalConfig, ensure_directory, save_dataframe
from generate_signals import generate_dataset
from extract_features import build_feature_table
from train_classifier import train_classifier


def run_repeated_stats(
    n_runs: int = 10,
    samples_per_state: int = 120,
    duration_s: float = 120.0,
    sample_rate_hz: float = 10.0,
    base_seed: int = 42,
    stats_dir: str | Path = "results/stats",
) -> dict:
    stats_dir = ensure_directory(stats_dir)

    all_rows: list[dict] = []
    confusion_rows: list[dict] = []

    for run_idx in range(n_runs):
        run_id = run_idx + 1
        run_seed = base_seed + run_idx

        run_dir = ensure_directory(stats_dir / f"run_{run_id:02d}")
        raw_dir = ensure_directory(run_dir / "data")
        model_dir = ensure_directory(run_dir / "models")

        print(f"\n=== Run {run_id}/{n_runs} | seed={run_seed} ===")

        config = SignalConfig(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            random_seed=run_seed,
        )

        print("1. Generating synthetic signals...")
        raw_df = generate_dataset(
            config=config,
            samples_per_state=samples_per_state,
            output_dir=raw_dir,
        )

        print("2. Extracting features...")
        feature_csv = run_dir / "pelagia_features.csv"
        feature_df = build_feature_table(
            raw_signal_csv=raw_dir / "pelagia_raw_signals.csv",
            output_csv=feature_csv,
            sample_rate_hz=config.sample_rate_hz,
        )

        print("3. Training classifier...")
        classifier_outputs = train_classifier(
            feature_csv=feature_csv,
            model_dir=model_dir,
            random_state=run_seed,
        )

        metrics = classifier_outputs["metrics"]
        report_df = classifier_outputs["report_df"].copy()
        confusion_df = classifier_outputs["confusion_df"].copy()

        row = {
            "run_id": run_id,
            "seed": run_seed,
            "samples_per_state": samples_per_state,
            "n_total_cases": int(feature_df.shape[0]),
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
            "n_features": int(metrics["n_features"]),
            "accuracy": float(metrics["accuracy"]),
        }

        if "macro avg" in report_df.index:
            row["macro_precision"] = float(report_df.loc["macro avg", "precision"])
            row["macro_recall"] = float(report_df.loc["macro avg", "recall"])
            row["macro_f1"] = float(report_df.loc["macro avg", "f1-score"])
        else:
            row["macro_precision"] = np.nan
            row["macro_recall"] = np.nan
            row["macro_f1"] = np.nan

        if "weighted avg" in report_df.index:
            row["weighted_precision"] = float(report_df.loc["weighted avg", "precision"])
            row["weighted_recall"] = float(report_df.loc["weighted avg", "recall"])
            row["weighted_f1"] = float(report_df.loc["weighted avg", "f1-score"])
        else:
            row["weighted_precision"] = np.nan
            row["weighted_recall"] = np.nan
            row["weighted_f1"] = np.nan

        all_rows.append(row)

        for true_label in confusion_df.index:
            for pred_label in confusion_df.columns:
                confusion_rows.append(
                    {
                        "run_id": run_id,
                        "seed": run_seed,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(confusion_df.loc[true_label, pred_label]),
                    }
                )

        print(
            f"Accuracy: {row['accuracy']:.4f} | "
            f"Macro F1: {row['macro_f1']:.4f} | "
            f"Test cases: {row['n_test']}"
        )

    runs_df = pd.DataFrame(all_rows)
    confusion_long_df = pd.DataFrame(confusion_rows)

    summary = {
        "n_runs": int(n_runs),
        "samples_per_state": int(samples_per_state),
        "mean_accuracy": float(runs_df["accuracy"].mean()),
        "std_accuracy": float(runs_df["accuracy"].std(ddof=1)) if len(runs_df) > 1 else 0.0,
        "min_accuracy": float(runs_df["accuracy"].min()),
        "max_accuracy": float(runs_df["accuracy"].max()),
        "mean_macro_f1": float(runs_df["macro_f1"].mean()),
        "std_macro_f1": float(runs_df["macro_f1"].std(ddof=1)) if len(runs_df) > 1 else 0.0,
        "mean_weighted_f1": float(runs_df["weighted_f1"].mean()),
        "std_weighted_f1": float(runs_df["weighted_f1"].std(ddof=1)) if len(runs_df) > 1 else 0.0,
        "mean_n_test": float(runs_df["n_test"].mean()),
    }

    summary_df = pd.DataFrame([summary])

    save_dataframe(runs_df, stats_dir / "repeated_run_metrics.csv")
    save_dataframe(confusion_long_df, stats_dir / "repeated_run_confusions_long.csv")
    save_dataframe(summary_df, stats_dir / "repeated_run_summary.csv")

    summary_txt = (
        f"Repeated stochastic evaluation\n"
        f"Runs: {summary['n_runs']}\n"
        f"Samples per state: {summary['samples_per_state']}\n"
        f"Mean accuracy: {summary['mean_accuracy']:.4f}\n"
        f"Std accuracy: {summary['std_accuracy']:.4f}\n"
        f"Min accuracy: {summary['min_accuracy']:.4f}\n"
        f"Max accuracy: {summary['max_accuracy']:.4f}\n"
        f"Mean macro F1: {summary['mean_macro_f1']:.4f}\n"
        f"Std macro F1: {summary['std_macro_f1']:.4f}\n"
        f"Mean weighted F1: {summary['mean_weighted_f1']:.4f}\n"
        f"Std weighted F1: {summary['std_weighted_f1']:.4f}\n"
        f"Mean test cases: {summary['mean_n_test']:.1f}\n"
    )
    (Path(stats_dir) / "repeated_run_summary.txt").write_text(summary_txt, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(runs_df["run_id"], runs_df["accuracy"], marker="o")
    ax.axhline(summary["mean_accuracy"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy")
    ax.set_title("Repeated stochastic runs: classifier accuracy")
    ax.set_xticks(runs_df["run_id"].tolist())
    fig.tight_layout()
    fig.savefig(Path(stats_dir) / "repeated_run_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    mean_confusion = (
        confusion_long_df.groupby(["true_label", "predicted_label"], as_index=False)["count"]
        .mean()
        .pivot(index="true_label", columns="predicted_label", values="count")
        .fillna(0.0)
    )
    mean_confusion.to_csv(Path(stats_dir) / "mean_confusion_matrix.csv", index=True)

    print("\n=== Summary across runs ===")
    print(summary_txt)

    return {
        "runs_df": runs_df,
        "summary": summary,
        "summary_df": summary_df,
        "confusion_long_df": confusion_long_df,
        "mean_confusion_df": mean_confusion,
    }


if __name__ == "__main__":
    outputs = run_repeated_stats(
        n_runs=10,
        samples_per_state=120,
        duration_s=120.0,
        sample_rate_hz=10.0,
        base_seed=42,
        stats_dir="results/stats",
    )
    print(outputs["summary"])
