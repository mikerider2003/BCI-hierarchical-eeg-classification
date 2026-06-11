# scripts/results/summary.py
# python -m scripts.results.summary

import json
from pathlib import Path


def load_results(path):
    path = Path(path) / "results.json"

    with open(path, "r") as f:
        results = json.load(f)

    return results


def get_validation_summary(results):
    summary = results["cv_summary"]

    return {
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "macro_f1_mean": summary["macro_f1_mean"],
        "macro_f1_std": summary["macro_f1_std"],
    }


def get_test_summary(results):
    test = results["test"]

    return {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
    }


def print_model_summary(model_name, results):
    validation = get_validation_summary(results)
    test = get_test_summary(results)

    print(model_name)
    print("-" * len(model_name))

    print(
        f"Validation accuracy: "
        f"{validation['accuracy_mean']:.3f} ± {validation['accuracy_std']:.3f}"
    )
    print(
        f"Validation macro-F1: "
        f"{validation['macro_f1_mean']:.3f} ± {validation['macro_f1_std']:.3f}"
    )

    print(f"Test accuracy: {test['accuracy']:.3f}")
    print(f"Test macro-F1: {test['macro_f1']:.3f}")
    print()


def main():
    EEG_PATH = Path("outputs/eegnet_baseline/")
    HCNN_PATH = Path("outputs/hierarchical_eeg_cnn/")

    eegnet_results = load_results(EEG_PATH)
    hcnn_results = load_results(HCNN_PATH)

    print()
    print("Results summary")
    print("===============")
    print()

    print_model_summary("EEGNet", eegnet_results)
    print_model_summary("Hierarchical CNN", hcnn_results)


if __name__ == "__main__":
    main()