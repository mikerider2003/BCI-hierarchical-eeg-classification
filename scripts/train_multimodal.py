"""Train the multimodal EEG + GSR/PPG model (cross-subject).

Mirrors train_eegnet_baseline.py exactly (same precomputed StratifiedGroupKFold
folds, same per-fold train-only standardization, inverse-frequency weighted
cross-entropy, early stopping on val macro-F1, final-model-on-test protocol) so
the numbers are directly comparable to the flat EEGNet baseline. The only change
is the data: EEG epochs and the row-aligned GSR/PPG epochs are loaded together
and fed to a two-branch fusion model.

The EEG pkl and the physio pkl are aligned 1:1 (same epoch order, identical y and
groups -- guaranteed by preprocess_physio.py), so the same split indices apply to
both arrays unchanged.

Modes (ablations):
    --mode fused    EEG + physio  (default; tests the multimodal hypothesis)
    --mode eeg      EEG only       (sanity check vs the baseline)
    --mode physio   GSR/PPG only

Scaling: EEG volts -> microvolts then per-channel z-score (train stats only).
Physio channels are z-scored separately (their own train stats) because GSR and
PPG live on entirely different scales from EEG and from each other.

Run:
    python scripts/train_multimodal.py                 # fused
    python scripts/train_multimodal.py --mode physio   # ablation
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
from models.multimodal_eegnet import MultimodalEEGNet  # noqa: E402

EEG_PATH = "data/ds007537/all_subjects_preprocessed.pkl"
PHYSIO_PATH = "data/ds007537/all_subjects_physio.pkl"
SPLITS_PATH = "outputs/groupkfold_splits.pkl"


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


def make_loader(Xe, Xp, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(Xe).float().unsqueeze(1),  # (N, 1, C, T)
        torch.from_numpy(Xp).float(),               # (N, 2, T)
        torch.from_numpy(y).long(),
    )
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
    for xe, xp, yb in loader:
        logits = model(xe.to(device), xp.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def metrics(trues, preds, n_classes):
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "macro_f1": float(f1_score(trues, preds, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "per_class_f1": f1_score(trues, preds, average=None, labels=list(range(n_classes)), zero_division=0).tolist(),
        "confusion": confusion_matrix(trues, preds, labels=list(range(n_classes))).tolist(),
    }


def train_one(model, train_loader, val_loader, criterion, device, args, label=""):
    """Train with early stopping on val macro-F1. Returns best state dict + best f1."""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for xe, xp, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xe.to(device), xp.to(device)), yb.to(device))
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


def build_model(args, n_classes, n_eeg_channels, n_physio_channels, n_samples, device):
    return MultimodalEEGNet(
        n_classes=n_classes,
        n_eeg_channels=n_eeg_channels,
        n_samples=n_samples,
        n_physio_channels=n_physio_channels,
        dropout=args.dropout,
        use_eeg=(args.mode in ("fused", "eeg")),
        use_physio=(args.mode in ("fused", "physio")),
    ).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fused", "eeg", "physio"], default="fused")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(f"outputs/multimodal_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | mode: {args.mode}")

    eeg = pickle.load(open(EEG_PATH, "rb"))
    phys = pickle.load(open(PHYSIO_PATH, "rb"))
    splits = pickle.load(open(SPLITS_PATH, "rb"))

    # Alignment is the whole premise of reusing the EEG splits: assert it loudly.
    if not (np.array_equal(eeg["groups"], phys["groups"]) and np.array_equal(eeg["y"], phys["y"])):
        raise RuntimeError("EEG and physio pkls are not row-aligned (groups/y differ). "
                           "Re-run scripts/preprocess_physio.py.")

    Xe = eeg["X"].astype(np.float32) * 1e6   # volts -> microvolts
    Xp = phys["X"].astype(np.float32)        # already filtered; standardized below
    classes = list(splits["classes"])
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in np.array(eeg["y"]).astype(str)], dtype=np.int64)
    n_eeg_channels, n_samples = Xe.shape[1], Xe.shape[2]
    n_physio_channels = Xp.shape[1]
    print(f"EEG {Xe.shape} | physio {Xp.shape} | classes {classes}")

    # ---- Cross-validation over precomputed folds ----
    n_folds = len(splits["folds"])
    fold_results = []
    for fi, fold in enumerate(splits["folds"]):
        print(f"\n=== Fold {fi+1}/{n_folds} ===", flush=True)
        tr, va = fold["train_idx"], fold["val_idx"]
        Xetr, Xeva = standardize(Xe[tr], Xe[va])
        Xptr, Xpva = standardize(Xp[tr], Xp[va])
        ytr, yva = y[tr], y[va]

        model = build_model(args, n_classes, n_eeg_channels, n_physio_channels, n_samples, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes, device))
        train_loader = make_loader(Xetr, Xptr, ytr, args.batch_size, True)
        val_loader = make_loader(Xeva, Xpva, yva, args.batch_size, False)

        best_state, _ = train_one(model, train_loader, val_loader, criterion, device, args,
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

    # ---- Final model: train on full train/val pool, evaluate once on test ----
    tv = splits["train_val_idx"]
    te = splits["test_idx"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tv))
    n_es = max(1, int(0.1 * len(tv)))
    es_idx, fit_idx = tv[perm[:n_es]], tv[perm[n_es:]]

    Xefit, Xees, Xete = standardize(Xe[fit_idx], Xe[es_idx], Xe[te])
    Xpfit, Xpes, Xpte = standardize(Xp[fit_idx], Xp[es_idx], Xp[te])
    yfit, yes, yte = y[fit_idx], y[es_idx], y[te]

    print("\n=== Final model (train on full pool, eval on test) ===", flush=True)
    final = build_model(args, n_classes, n_eeg_channels, n_physio_channels, n_samples, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(yfit, n_classes, device))
    best_state, _ = train_one(
        final,
        make_loader(Xefit, Xpfit, yfit, args.batch_size, True),
        make_loader(Xees, Xpes, yes, args.batch_size, False),
        criterion, device, args, label="final",
    )
    final.load_state_dict(best_state)
    trues, preds = evaluate(final, make_loader(Xete, Xpte, yte, args.batch_size, False), device)
    test_m = metrics(trues, preds, n_classes)
    print(f"\nTEST: acc={test_m['accuracy']:.3f} macroF1={test_m['macro_f1']:.3f} "
          f"perClassF1={[round(f,3) for f in test_m['per_class_f1']]}")
    print("Test confusion (rows=true, cols=pred):")
    for c, row in zip(classes, test_m["confusion"]):
        print(f"  {c:28s} {row}")

    results = {
        "mode": args.mode,
        "classes": classes,
        "args": vars(args),
        "cv_folds": fold_results,
        "cv_summary": {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        },
        "test": test_m,
    }
    json.dump(results, open(out_dir / "results.json", "w"), indent=2)
    torch.save(final.state_dict(), out_dir / "final_model.pt")
    print(f"\nSaved -> {out_dir}/results.json, {out_dir}/final_model.pt")


if __name__ == "__main__":
    main()
