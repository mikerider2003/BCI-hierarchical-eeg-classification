# scripts/visualization/cms.py
# python -m scripts.visualization.cms

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay


def load_results(path):
    path = Path(path) / "results.json"


    with open(path, "r") as f:
        results = json.load(f)

    return results

def plot_confusion_matrix(confusion, classes, title, save_path=None, normalize=False):
    cm = np.array(confusion)

    if normalize:
        # Normalize by true class, meaning each row sums to 1
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        xticks_rotation=45,
        colorbar=True,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_confusion_matrices_side_by_side(
    confusion_1,
    confusion_2,
    classes,
    title_1,
    title_2,
    main_title,
    save_path=None,
    normalize=False,
):
    short_classes = _short_class_labels(classes)
    class_colors = [_class_group_color(class_name) for class_name in classes]

    cm1 = np.array(confusion_1)
    cm2 = np.array(confusion_2)

    if normalize:
        row_sums_1 = cm1.sum(axis=1, keepdims=True)
        row_sums_2 = cm2.sum(axis=1, keepdims=True)

        cm1 = np.divide(cm1, row_sums_1, where=row_sums_1 != 0)
        cm2 = np.divide(cm2, row_sums_2, where=row_sums_2 != 0)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")


    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm1,
        display_labels=short_classes,
    )

    disp1.plot(
        ax=axes[0],
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        colorbar=False,
    )

    axes[0].set_title(title_1, pad=4, fontsize=10)
    axes[0].set_xlabel("Predicted label", labelpad=2, fontsize=9)
    axes[0].set_ylabel("True label", labelpad=2, fontsize=9)
    axes[0].tick_params(axis="both", labelsize=8, pad=1)

    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm2,
        display_labels=short_classes,
    )

    disp2.plot(
        ax=axes[1],
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        colorbar=False,
    )

    axes[1].set_title(title_2, pad=4, fontsize=10)
    axes[1].set_xlabel("Predicted label", labelpad=2, fontsize=9)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="both", labelsize=8, pad=1)
    axes[1].tick_params(axis="y", labelleft=False)


    _color_tick_labels(axes[0], class_colors, color_y=True)
    _color_tick_labels(axes[1], class_colors, color_y=False)

    # legend_handles = [
    #     Patch(facecolor=color, edgecolor="none", label=f"{short}: {full}")
    #     for short, full, color in zip(short_classes, classes, class_colors)
    # ]

    # fig.legend(
    #     handles=legend_handles,
    #     loc="upper right",
    #     ncol=2,
    #     # fontsize=8,
    # )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def _short_class_labels(classes):
    labels = []
    for class_name in classes:
        parts = class_name.replace("-", "_").split("/")
        if len(parts) > 1:
            labels.append("".join(part[0].upper() for part in parts[:-1]) + _initials(parts[-1]))
        else:
            labels.append(_initials(parts[0]))
    return labels


def _initials(text):
    return "".join(part[0].upper() for part in text.split("_") if part)


def _color_tick_labels(ax, colors, color_y):
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)

    if color_y:
        for label, color in zip(ax.get_yticklabels(), colors):
            label.set_color(color)


def _class_group_color(class_name):
    if class_name.startswith("smartphone"):
        return "tab:blue"
    return "tab:red"

def main():
    EEG_PATH = Path("outputs/eegnet_baseline/")
    HCNN_PATH = Path("outputs/hierarchical_eeg_cnn/")

    eegnet_results = load_results(EEG_PATH)
    hcnn_results = load_results(HCNN_PATH)

    print("EEGNet results:")
    print(eegnet_results)

    print("\nHierarchical CNN results:")
    print(hcnn_results)

    classes = eegnet_results["classes"]

    # Test confusion matrices
    # plot_confusion_matrix(
    #     confusion=eegnet_results["test"]["confusion"],
    #     classes=classes,
    #     title="EEGNet Test Confusion Matrix",
    #     save_path=EEG_PATH / "eegnet_test_cm.png",
    #     normalize=False,
    # )


    # plot_confusion_matrix(
    #     confusion=hcnn_results["test"]["confusion"],
    #     classes=classes,
    #     title="Hierarchical CNN Test Confusion Matrix",
    #     save_path=HCNN_PATH / "hierarchical_cnn_test_cm.png",
    #     normalize=False,
    # )

    plot_confusion_matrices_side_by_side(
        confusion_1=eegnet_results["test"]["confusion"],
        confusion_2=hcnn_results["test"]["confusion"],
        classes=classes,
        title_1="EEGNet",
        title_2="Hierarchical CNN",
        main_title="Test Confusion Matrices",
        save_path=Path("outputs") / "test_confusion_matrices_comparison.png",
        normalize=False,
    )


if __name__ == "__main__":
    main()
