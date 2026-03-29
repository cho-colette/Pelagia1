from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from train_classifier import FEATURE_COLUMNS
from utils import ensure_directory


def plot_decision_tree_figure(
    model_path: str | Path = "results/models/pelagia_decision_tree.joblib",
    output_path: str | Path = "figures/pelagia_decision_tree.png",
    dpi: int = 200,
) -> Path:
    clf = joblib.load(model_path)

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=FEATURE_COLUMNS,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix_heatmap(
    confusion_csv: str | Path = "results/models/confusion_matrix.csv",
    output_path: str | Path = "figures/pelagia_confusion_matrix.png",
    dpi: int = 200,
) -> Path:
    cm = pd.read_csv(confusion_csv, index_col=0)

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm.values)

    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Pelagia classifier confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.iloc[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    tree_path = plot_decision_tree_figure()
    cm_path = plot_confusion_matrix_heatmap()
    print(f"Saved: {tree_path}")
    print(f"Saved: {cm_path}")
