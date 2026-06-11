"""Train a two-stage Hierarchical EEG-CNN baseline (cross-subject).

This mirrors the flat EEGNet baseline protocol as closely as possible:
- use the same precomputed subject-disjoint splits
- convert volts to microvolts
- standardize per channel using TRAIN statistics only
- early-stop on validation macro-F1
- evaluate once on the held-out test subjects

Run:
    python scripts/train_hierarchical_eeg_cnn.py
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.hierarchical_eeg_cnn import HFCNNStyleEEG, hierarchical_loss  # noqa: E402

DATA_PATH = "outputs/all_subjects_preprocessed.pkl"
SPLITS_PATH = "outputs/groupkfold_splits.pkl"
OUT_DIR = Path("outputs/hierarchical_eeg_cnn")


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


def make_loader(X, y_flat, y_binary, y_smartphone, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X).float(),                 # (N, C, T)
        torch.from_numpy(y_flat).long(),             # original 4-class label
        torch.from_numpy(y_binary).long(),           # 0=video, 1=smartphone
        torch.from_numpy(y_smartphone).long(),       # smartphone subtype, dummy 0 for video
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def standardize(X_train, *others):
    """Per-channel z-score from train stats. X shape: (N, C, T)."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-7
    out = [(X_train - mean) / std]
    out += [(X - mean) / std for X in others]
    return out


def class_weights(y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_hierarchical_targets(y_flat, video_idx, flat_to_smartphone_idx):
    """Create binary and smartphone-subclass targets from original flat labels."""
    y_binary = (y_flat != video_idx).astype(np.int64)
    y_smartphone = np.zeros_like(y_flat, dtype=np.int64)

    for flat_idx, smartphone_idx in flat_to_smartphone_idx.items():
        y_smartphone[y_flat == flat_idx] = smartphone_idx

    return y_binary, y_smartphone


@torch.no_grad()
def evaluate(model, loader, device, video_idx, smartphone_idx_to_flat_idx):
    model.eval()
    preds, trues = [], []
    smartphone_idx_to_flat_idx = smartphone_idx_to_flat_idx.to(device)

    for xb, y_flat, _, _ in loader:
        xb = xb.to(device)
        binary_logits, smartphone_logits = model(xb)

        binary_pred = binary_logits.argmax(dim=1)
        smartphone_pred = smartphone_logits.argmax(dim=1)

        final_pred = torch.full_like(binary_pred, fill_value=video_idx)
        mask = binary_pred == 1
        final_pred[mask] = smartphone_idx_to_flat_idx[smartphone_pred[mask]]

        preds.append(final_pred.cpu().numpy())
        trues.append(y_flat.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return trues, preds


def metrics(trues, preds, n_classes):
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "macro_f1": float(f1_score(trues, preds, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": f1_score(trues, preds, average=None, labels=labels, zero_division=0).tolist(),
        "confusion": confusion_matrix(trues, preds, labels=labels).tolist(),
    }


def train_one(
    model,
    train_loader,
    val_loader,
    stage1_class_weights,
    stage2_class_weights,
    device,
    args,
    video_idx,
    smartphone_idx_to_flat_idx,
    label="",
):
    """Train with early stopping on final 4-class validation macro-F1."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1, best_state, patience = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        running, running_bin, running_phone, n_batches = 0.0, 0.0, 0.0, 0

        for xb, _, y_binary, y_smartphone in train_loader:
            xb = xb.to(device)
            y_binary = y_binary.to(device)
            y_smartphone = y_smartphone.to(device)

            opt.zero_grad()
            stage1_logits, stage2_logits = model(xb)
            loss, loss_stage1, loss_stage2 = hierarchical_loss(
                stage1_logits,
                stage2_logits,
                y_binary,
                y_smartphone,
                stage1_weight=args.stage1_loss_weight,
                stage2_weight=args.stage2_loss_weight,
                stage1_class_weights=stage1_class_weights,
                stage2_class_weights=stage2_class_weights,
            )
            loss.backward()
            opt.step()

            running += loss.item()
            running_bin += loss_stage1.item()
            running_phone += loss_stage2.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        train_bin = running_bin / max(n_batches, 1)
        train_phone = running_phone / max(n_batches, 1)
        pct = 100 * (epoch + 1) / args.epochs

        if val_loader is None:
            print(
                f"  [{label}] epoch {epoch+1:3d}/{args.epochs} ({pct:5.1f}%)  "
                f"train_loss={train_loss:.4f}  bin={train_bin:.4f}  phone={train_phone:.4f}",
                flush=True,
            )
            continue

        trues, preds = evaluate(model, val_loader, device, video_idx, smartphone_idx_to_flat_idx)
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        improved = f1 > best_f1

        if improved:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        print(
            f"  [{label}] epoch {epoch+1:3d}/{args.epochs} ({pct:5.1f}%)  "
            f"train_loss={train_loss:.4f}  bin={train_bin:.4f}  phone={train_phone:.4f}  "
            f"val_f1={f1:.4f}  best={best_f1:.4f}{' *' if improved else ''}  "
            f"patience={patience}/{args.patience}",
            flush=True,
        )

        if patience >= args.patience:
            print(f"  [{label}] early stop at epoch {epoch+1} (best val_f1={best_f1:.4f})", flush=True)
            break

    return best_state, best_f1


def build_model(args, n_channels, n_samples, n_smartphone_classes, device):
    return HFCNNStyleEEG(
        n_channels=n_channels,
        n_samples=n_samples,
        n_smartphone_classes=n_smartphone_classes,
        dropout=args.dropout,
    ).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--stage1_loss_weight", type=float, default=0.5)
    p.add_argument("--stage2_loss_weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    data = pickle.load(open(DATA_PATH, "rb"))
    splits = pickle.load(open(SPLITS_PATH, "rb"))

    X = data["X"].astype(np.float32) * 1e6  # volts -> microvolts
    classes = list(splits["classes"])
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if "video" not in class_to_idx:
        raise ValueError(f"Expected class 'video' in classes, got: {classes}")

    video_idx = class_to_idx["video"]
    smartphone_classes = [c for c in classes if c != "video"]
    n_smartphone_classes = len(smartphone_classes)

    flat_to_smartphone_idx = {class_to_idx[c]: i for i, c in enumerate(smartphone_classes)}
    smartphone_idx_to_flat_idx = torch.tensor(
        [class_to_idx[c] for c in smartphone_classes],
        dtype=torch.long,
    )

    y_flat = np.array([class_to_idx[c] for c in np.array(data["y"]).astype(str)], dtype=np.int64)
    y_binary, y_smartphone = make_hierarchical_targets(y_flat, video_idx, flat_to_smartphone_idx)

    n_channels, n_samples = X.shape[1], X.shape[2]
    print(f"X {X.shape}, classes {classes}")
    print(f"Stage 1: video_idx={video_idx} vs smartphone")
    print(f"Stage 2 smartphone classes: {smartphone_classes}")

    # ---- Cross-validation over precomputed folds ----
    n_folds = len(splits["folds"])
    fold_results = []

    for fi, fold in enumerate(splits["folds"]):
        print(f"\n=== Fold {fi+1}/{n_folds} ===", flush=True)
        tr, va = fold["train_idx"], fold["val_idx"]

        Xtr, Xva = standardize(X[tr], X[va])
        ytr_flat, yva_flat = y_flat[tr], y_flat[va]
        ytr_bin, yva_bin = y_binary[tr], y_binary[va]
        ytr_phone, yva_phone = y_smartphone[tr], y_smartphone[va]

        model = build_model(args, n_channels, n_samples, n_smartphone_classes, device)
        stage1_class_weights = class_weights(ytr_bin, 2, device)
        stage2_class_weights = class_weights(ytr_phone[ytr_bin == 1], n_smartphone_classes, device)

        train_loader = make_loader(Xtr, ytr_flat, ytr_bin, ytr_phone, args.batch_size, True)
        val_loader = make_loader(Xva, yva_flat, yva_bin, yva_phone, args.batch_size, False)

        best_state, _ = train_one(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            stage1_class_weights=stage1_class_weights,
            stage2_class_weights=stage2_class_weights,
            device=device,
            args=args,
            video_idx=video_idx,
            smartphone_idx_to_flat_idx=smartphone_idx_to_flat_idx,
            label=f"fold {fi+1}/{n_folds}",
        )

        if best_state is not None:
            model.load_state_dict(best_state)

        trues, preds = evaluate(model, val_loader, device, video_idx, smartphone_idx_to_flat_idx)
        m = metrics(trues, preds, n_classes)
        m["fold"] = fold["fold"]
        fold_results.append(m)

        print(
            f"Fold {fold['fold']}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
            f"perClassF1={[round(f, 3) for f in m['per_class_f1']]}",
            flush=True,
        )

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    print(
        f"\nCV: acc {np.mean(accs):.3f}+/-{np.std(accs):.3f} | "
        f"macroF1 {np.mean(f1s):.3f}+/-{np.std(f1s):.3f}",
        flush=True,
    )

    # ---- Final model: train on train/val pool, evaluate on held-out test subjects ----
    tv = splits["train_val_idx"]
    te = splits["test_idx"]

    # Stratified early-stopping slice inside the train/val pool.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
    fit_rel, es_rel = next(splitter.split(np.zeros(len(tv)), y_flat[tv]))
    fit_idx, es_idx = tv[fit_rel], tv[es_rel]

    Xfit, Xes, Xte = standardize(X[fit_idx], X[es_idx], X[te])
    yfit_flat, yes_flat, yte_flat = y_flat[fit_idx], y_flat[es_idx], y_flat[te]
    yfit_bin, yes_bin, yte_bin = y_binary[fit_idx], y_binary[es_idx], y_binary[te]
    yfit_phone, yes_phone, yte_phone = y_smartphone[fit_idx], y_smartphone[es_idx], y_smartphone[te]

    print("\n=== Final model (train on pool, eval on test) ===", flush=True)
    final = build_model(args, n_channels, n_samples, n_smartphone_classes, device)
    stage1_class_weights = class_weights(yfit_bin, 2, device)
    stage2_class_weights = class_weights(yfit_phone[yfit_bin == 1], n_smartphone_classes, device)

    best_state, _ = train_one(
        model=final,
        train_loader=make_loader(Xfit, yfit_flat, yfit_bin, yfit_phone, args.batch_size, True),
        val_loader=make_loader(Xes, yes_flat, yes_bin, yes_phone, args.batch_size, False),
        stage1_class_weights=stage1_class_weights,
        stage2_class_weights=stage2_class_weights,
        device=device,
        args=args,
        video_idx=video_idx,
        smartphone_idx_to_flat_idx=smartphone_idx_to_flat_idx,
        label="final",
    )

    if best_state is not None:
        final.load_state_dict(best_state)

    test_loader = make_loader(Xte, yte_flat, yte_bin, yte_phone, args.batch_size, False)
    trues, preds = evaluate(final, test_loader, device, video_idx, smartphone_idx_to_flat_idx)
    test_m = metrics(trues, preds, n_classes)

    print(
        f"\nTEST: acc={test_m['accuracy']:.3f} macroF1={test_m['macro_f1']:.3f} "
        f"perClassF1={[round(f, 3) for f in test_m['per_class_f1']]}",
        flush=True,
    )
    print("Test confusion (rows=true, cols=pred):")
    for c, row in zip(classes, test_m["confusion"]):
        print(f"  {c:28s} {row}")

    results = {
        "classes": classes,
        "hierarchy": {
            "stage_1": {"0": "video", "1": "smartphone"},
            "stage_2_smartphone_classes": smartphone_classes,
            "video_idx": int(video_idx),
        },
        "args": vars(args),
        "cv_folds": fold_results,
        "cv_summary": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        },
        "test": test_m,
    }

    json.dump(results, open(OUT_DIR / "results.json", "w"), indent=2)
    torch.save(final.state_dict(), OUT_DIR / "final_model.pt")
    print(f"\nSaved -> {OUT_DIR}/results.json, {OUT_DIR}/final_model.pt")


if __name__ == "__main__":
    main()
