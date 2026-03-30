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
    figsize: tuple[int, int] = (32, 18),
    fontsize: int = 26,
) -> Path:
    clf = joblib.load(model_path)

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    class_names = [str(c).replace("_", " ") for c in clf.classes_]

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        clf,
        feature_names=FEATURE_COLUMNS,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=fontsize,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix_heatmap(
    confusion_csv: str | Path = "results/models/confusion_matrix.csv",
    output_path: str | Path = "figures/pelagia_confusion_matrix.png",
    dpi: int = 300,
    figsize: tuple[int, int] = (11, 9),
    label_fontsize: int = 18,
    tick_fontsize: int = 15,
    value_fontsize: int = 16,
    title_fontsize: int = 22,
) -> Path:
    cm = pd.read_csv(confusion_csv, index_col=0)

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm.values, cmap="viridis")

    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(
        [str(c).replace("_", " ") for c in cm.columns],
        rotation=30,
        ha="right",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(
        [str(i).replace("_", " ") for i in cm.index],
        fontsize=tick_fontsize,
    )

    ax.set_xlabel("Predicted label", fontsize=label_fontsize)
    ax.set_ylabel("True label", fontsize=label_fontsize)
    ax.set_title("Pelagia classifier confusion matrix", fontsize=title_fontsize)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=value_fontsize,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    tree_path = plot_decision_tree_figure()
    cm_path = plot_confusion_matrix_heatmap()
    print(f"Saved: {tree_path}")
    print(f"Saved: {cm_path}")
