# scripts/results/cms.py
# python -m scripts.results.cms

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
    short_classes = ["SG", "SR", "SSV", "V"]
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

    axes[0].set_title(title_1, pad=4, fontsize=11)
    axes[0].set_xlabel("Predicted label", labelpad=2, fontsize=11)
    axes[0].set_ylabel("True label", labelpad=2, fontsize=11)
    axes[0].tick_params(axis="both", labelsize=11, pad=1)

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

    axes[1].set_title(title_2, pad=4, fontsize=11)
    axes[1].set_xlabel("Predicted label", labelpad=2, fontsize=11)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="both", labelsize=11, pad=1)
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




def plot_confusion_matrices_row(
    confusions,
    titles,
    classes,
    save_path=None,
    normalize=False,
):
    """Plot any number of test confusion matrices in a single row.

    Shares the styling of the two-panel helper: short class labels, group-coloured
    ticks (smartphone = blue, video = red), and y-axis labels only on the leftmost
    panel.
    """
    short_classes = ["SG", "SR", "SSV", "V"]
    class_colors = [_class_group_color(class_name) for class_name in classes]
    n = len(confusions)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), layout="constrained")
    if n == 1:
        axes = [axes]

    for i, (confusion, title) in enumerate(zip(confusions, titles)):
        cm = np.array(confusion)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, where=row_sums != 0)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_classes)
        disp.plot(
            ax=axes[i],
            cmap="Blues",
            values_format=".2f" if normalize else "d",
            colorbar=False,
        )

        axes[i].set_title(title, pad=4, fontsize=11)
        axes[i].set_xlabel("Predicted label", labelpad=2, fontsize=11)
        axes[i].tick_params(axis="both", labelsize=11, pad=1)
        if i == 0:
            axes[i].set_ylabel("True label", labelpad=2, fontsize=11)
        else:
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="y", labelleft=False)

        _color_tick_labels(axes[i], class_colors, color_y=(i == 0))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def _color_tick_labels(ax, colors, color_y):
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)

    if color_y:
        for label, color in zip(ax.get_yticklabels(), colors):
            label.set_color(color)


def _class_group_color(class_name):
    return {
        "smartphone/gaming": "tab:blue",
        "smartphone/reading": "tab:blue",
        "smartphone/short_videos": "tab:blue",
        "video": "tab:red",
    }[class_name]

def main():
    EEG_PATH = Path("outputs/eegnet_baseline/")
    HCNN_PATH = Path("outputs/hierarchical_eeg_cnn/")
    FUSED_PATH = Path("outputs/multimodal_fused/")

    eegnet_results = load_results(EEG_PATH)
    hcnn_results = load_results(HCNN_PATH)
    fused_results = load_results(FUSED_PATH)

    for name, res in [("EEGNet", eegnet_results),
                      ("Hierarchical CNN", hcnn_results),
                      ("Multimodal (EEG+Physio)", fused_results)]:
        t = res["test"]
        print(f"{name}: test acc={t['accuracy']:.3f} macroF1={t['macro_f1']:.3f}")

    classes = eegnet_results["classes"]

    plot_confusion_matrices_row(
        confusions=[
            eegnet_results["test"]["confusion"],
            hcnn_results["test"]["confusion"],
            fused_results["test"]["confusion"],
        ],
        titles=["EEGNet", "Hierarchical CNN", "EEG + Physio"],
        classes=classes,
        save_path=Path("outputs") / "test_confusion_matrices_comparison.png",
        normalize=False,
    )


if __name__ == "__main__":
    main()
