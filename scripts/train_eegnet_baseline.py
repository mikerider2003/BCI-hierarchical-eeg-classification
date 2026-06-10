"""Train the flat 4-class EEGNet baseline (cross-subject).

Uses the precomputed subject-disjoint StratifiedGroupKFold splits. For each
fold the model trains on the train subjects and is early-stopped on the
val-fold macro-F1; we report mean +/- std across folds. A final model is then
trained on the full train/val pool and evaluated once on the held-out test
subjects.

Scaling: volts -> microvolts, then per-channel z-score using TRAIN statistics
only (recomputed per fold) to avoid leakage. Class imbalance is handled with
inverse-frequency weighted cross-entropy.

Run:
    python scripts/train_eegnet_baseline.py
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.eegnet import EEGNet  # noqa: E402

DATA_PATH = "data/ds007537/all_subjects_preprocessed.pkl"
SPLITS_PATH = "outputs/groupkfold_splits.pkl"
OUT_DIR = Path("outputs/eegnet_baseline")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(X, y, batch_size, shuffle, device):
    ds = TensorDataset(
        torch.from_numpy(X).float().unsqueeze(1),  # (N, 1, C, T)
        torch.from_numpy(y).long(),
    )
    # pin_memory only helps CUDA; keep simple and device-agnostic
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def standardize(X_train, *others):
    """Per-channel z-score from train stats. X shape (N, C, T)."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-7
    out = [(X_train - mean) / std]
    out += [(X - mean) / std for X in others]
    return out


def class_weights(y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return trues, preds


def metrics(trues, preds, n_classes):
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "macro_f1": float(f1_score(trues, preds, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "per_class_f1": f1_score(trues, preds, average=None, labels=list(range(n_classes)), zero_division=0).tolist(),
        "confusion": confusion_matrix(trues, preds, labels=list(range(n_classes))).tolist(),
    }


def train_one(model, train_loader, val_loader, criterion, device, args, label=""):
    """Train with early stopping on val macro-F1. Returns best state dict + history.

    Prints a per-epoch progress line: epoch/total (%), mean train loss, val
    macro-F1, the best macro-F1 so far, and the early-stopping patience counter.
    """
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            model.apply_max_norm()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        pct = 100 * (epoch + 1) / args.epochs

        if val_loader is None:
            print(f"  [{label}] epoch {epoch+1:3d}/{args.epochs} ({pct:5.1f}%)  "
                  f"train_loss={train_loss:.4f}", flush=True)
            continue

        trues, preds = evaluate(model, val_loader, device)
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        improved = f1 > best_f1
        if improved:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(f"  [{label}] epoch {epoch+1:3d}/{args.epochs} ({pct:5.1f}%)  "
              f"train_loss={train_loss:.4f}  val_f1={f1:.4f}  "
              f"best={best_f1:.4f}{' *' if improved else ''}  patience={patience}/{args.patience}",
              flush=True)
        if patience >= args.patience:
            print(f"  [{label}] early stop at epoch {epoch+1} (best val_f1={best_f1:.4f})", flush=True)
            break
    return best_state, best_f1


def build_model(args, n_classes, n_channels, n_samples, device):
    return EEGNet(
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        dropout=args.dropout,
    ).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    data = pickle.load(open(DATA_PATH, "rb"))
    splits = pickle.load(open(SPLITS_PATH, "rb"))

    X = (data["X"].astype(np.float32)) * 1e6  # volts -> microvolts
    classes = list(splits["classes"])
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in np.array(data["y"]).astype(str)], dtype=np.int64)
    n_channels, n_samples = X.shape[1], X.shape[2]
    print(f"X {X.shape}, classes {classes}")

    # ---- Cross-validation over precomputed folds ----
    n_folds = len(splits["folds"])
    fold_results = []
    for fi, fold in enumerate(splits["folds"]):
        print(f"\n=== Fold {fi+1}/{n_folds} ===", flush=True)
        tr, va = fold["train_idx"], fold["val_idx"]
        Xtr, Xva = standardize(X[tr], X[va])
        ytr, yva = y[tr], y[va]

        model = build_model(args, n_classes, n_channels, n_samples, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes, device))
        train_loader = make_loader(Xtr, ytr, args.batch_size, True, device)
        val_loader = make_loader(Xva, yva, args.batch_size, False, device)

        best_state, best_f1 = train_one(model, train_loader, val_loader, criterion, device, args,
                                        label=f"fold {fi+1}/{n_folds}")
        model.load_state_dict(best_state)
        trues, preds = evaluate(model, val_loader, device)
        m = metrics(trues, preds, n_classes)
        m["fold"] = fold["fold"]
        fold_results.append(m)
        print(f"Fold {fold['fold']}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"perClassF1={[round(f,3) for f in m['per_class_f1']]}")

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    print(f"\nCV: acc {np.mean(accs):.3f}+/-{np.std(accs):.3f} | "
          f"macroF1 {np.mean(f1s):.3f}+/-{np.std(f1s):.3f}")

    # ---- Final model: train on full train/val pool, evaluate on test ----
    # Hold out a small stratified slice of the pool for early stopping.
    tv = splits["train_val_idx"]
    te = splits["test_idx"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tv))
    n_es = max(1, int(0.1 * len(tv)))
    es_idx, fit_idx = tv[perm[:n_es]], tv[perm[n_es:]]

    Xfit, Xes, Xte = standardize(X[fit_idx], X[es_idx], X[te])
    yfit, yes, yte = y[fit_idx], y[es_idx], y[te]

    print("\n=== Final model (train on full pool, eval on test) ===", flush=True)
    final = build_model(args, n_classes, n_channels, n_samples, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(yfit, n_classes, device))
    best_state, _ = train_one(
        final,
        make_loader(Xfit, yfit, args.batch_size, True, device),
        make_loader(Xes, yes, args.batch_size, False, device),
        criterion, device, args, label="final",
    )
    final.load_state_dict(best_state)
    trues, preds = evaluate(final, make_loader(Xte, yte, args.batch_size, False, device), device)
    test_m = metrics(trues, preds, n_classes)
    print(f"\nTEST: acc={test_m['accuracy']:.3f} macroF1={test_m['macro_f1']:.3f} "
          f"perClassF1={[round(f,3) for f in test_m['per_class_f1']]}")
    print("Test confusion (rows=true, cols=pred):")
    for c, row in zip(classes, test_m["confusion"]):
        print(f"  {c:28s} {row}")

    results = {
        "classes": classes,
        "args": vars(args),
        "cv_folds": fold_results,
        "cv_summary": {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        },
        "test": test_m,
    }
    json.dump(results, open(OUT_DIR / "results.json", "w"), indent=2)
    torch.save(final.state_dict(), OUT_DIR / "final_model.pt")
    print(f"\nSaved -> {OUT_DIR}/results.json, {OUT_DIR}/final_model.pt")


if __name__ == "__main__":
    main()
