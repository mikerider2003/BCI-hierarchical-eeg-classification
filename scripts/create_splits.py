import pickle
import numpy as np

from collections import Counter
from sklearn.model_selection import GroupKFold


# Config
RANDOM_SEED = 42
N_SPLITS = 5
READING_CLASS = "smartphone/reading"

DATA_PATH = "outputs/all_subjects_preprocessed.pkl"
OUT_PATH = "outputs/groupkfold_splits.pkl"

# Load preprocessed data
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = np.array(data["y"]).astype(str)
groups = np.array(data["groups"]).astype(str)
classes = data["classes"]

subjects = np.array(sorted(set(groups)))

print("X shape:", X.shape)
print("Classes:", classes)
print("Subjects:", subjects)
print("Number of subjects:", len(subjects))

# One smartphone class per subject - Find each subject’s smartphone class
subject_to_smartphone_class = {}

for subject in subjects:
    subject_labels    = y[groups == subject]
    smartphone_labels = sorted(
        set(label for label in subject_labels if label != "video")
    )
    if len(smartphone_labels) != 1:
        raise ValueError(
            f"{subject} has {len(smartphone_labels)} smartphone labels: {smartphone_labels}"
        )
    subject_to_smartphone_class[subject] = smartphone_labels[0]

subject_classes = np.array([subject_to_smartphone_class[s] for s in subjects])

print("\nSubjects per smartphone class:")
for label, count in Counter(subject_classes).items():
    print(f"  {label:28s}: {count}")

# Fixed test set: 1 subject per smartphone class
rng = np.random.default_rng(RANDOM_SEED)

test_subjects = []
for smartphone_class in sorted(set(subject_classes)):
    subjects_in_class = subjects[subject_classes == smartphone_class]
    selected = rng.choice(subjects_in_class, size=1, replace=False)
    test_subjects.extend(selected.tolist())

test_subjects = np.array(sorted(test_subjects))
train_val_subjects = np.array(
    sorted([s for s in subjects if s not in test_subjects])
)

test_idx = np.where(np.isin(groups, test_subjects))[0]
train_val_idx = np.where(np.isin(groups, train_val_subjects))[0]

print("\nFixed test subjects:", test_subjects)
print("Train/val subjects: ", train_val_subjects)

print("\nFixed test set class counts:")
for label in classes:
    print(f"  {label:28s}: {(y[test_idx] == label).sum()}")

print("\nTrain/val pool class counts:")
for label in classes:
    print(f"  {label:28s}: {(y[train_val_idx] == label).sum()}")

# HP tuning pool
# Reading subjects are explicitly excluded from the GroupKFold pool so they can never land in a val fold. 

reading_train_subjects = np.array([
    s for s in train_val_subjects
    if subject_to_smartphone_class[s] == READING_CLASS
])
hp_subjects = np.array([
    s for s in train_val_subjects
    if subject_to_smartphone_class[s] != READING_CLASS
])

hp_idx            = np.where(np.isin(groups, hp_subjects))[0]
reading_train_idx = np.where(np.isin(groups, reading_train_subjects))[0]

X_hp = X[hp_idx]
y_hp = y[hp_idx]
groups_hp = groups[hp_idx]

print(f"\nReading subjects (train-only, never in val): {reading_train_subjects}")
print(f"HP tuning subjects ({len(hp_subjects)}): {hp_subjects}")

# GroupKFold
gkf = GroupKFold(n_splits=N_SPLITS)
folds = []

print(f"\n{N_SPLITS}-Fold GroupKFold splits")

for fold, (train_rel, val_rel) in enumerate(
    gkf.split(X_hp, y_hp, groups_hp)
):
    # Map relative indices back to absolute positions in full X/y/groups
    train_abs = hp_idx[train_rel]
    val_abs = hp_idx[val_rel]

    # Re-add reading subject to training — guaranteed never in val
    train_abs = np.concatenate([train_abs, reading_train_idx])

    train_subjects_fold = sorted(set(groups[train_abs]))
    val_subjects_fold = sorted(set(groups[val_abs]))

    print(f"\nFold {fold + 1}")
    print(f"  Train subjects ({len(train_subjects_fold)}): {train_subjects_fold}")
    print(f"  Val   subjects ({len(val_subjects_fold)}):   {val_subjects_fold}")

    print("  Train class counts:")
    for label in classes:
        print(f"    {label:28s}: {(y[train_abs] == label).sum()}")

    print("  Val class counts:")
    for label in classes:
        print(f"    {label:28s}: {(y[val_abs] == label).sum()}")

    folds.append({
        "fold": fold,
        "train_idx": train_abs,
        "val_idx": val_abs,
    })

# checks
print("\n Checks")

for fold_data in folds:
    train_abs = fold_data["train_idx"]
    val_abs = fold_data["val_idx"]
    fold = fold_data["fold"]

    train_subs = set(groups[train_abs])
    val_subs = set(groups[val_abs])
    test_subs = set(groups[test_idx])

    assert train_subs.isdisjoint(val_subs), \
        f"Fold {fold+1}: subject overlap between train and val!"
    assert train_subs.isdisjoint(test_subs), \
        f"Fold {fold+1}: subject overlap between train and test!"
    assert val_subs.isdisjoint(test_subs), \
        f"Fold {fold+1}: subject overlap between val and test!"

    reading_in_val = any(
        subject_to_smartphone_class.get(s) == READING_CLASS
        for s in val_subs
    )
    assert not reading_in_val, \
        f"Fold {fold+1}: reading subject leaked into val!"

print("All checks passed.")

# Save
output = {
    "folds": folds,
    "test_idx": test_idx,
    "train_val_idx": train_val_idx,
    "test_subjects": test_subjects,
    "train_val_subjects": train_val_subjects,
    "reading_train_subjects": reading_train_subjects,
    "subject_to_smartphone_class": subject_to_smartphone_class,
    "classes": classes,
    "random_seed": RANDOM_SEED,
    "n_splits": N_SPLITS,
}

with open(OUT_PATH, "wb") as f:
    pickle.dump(output, f)

print(f"\nSaved splits to {OUT_PATH}")
print(f"  Folds:            {len(folds)}")
print(f"  Test epochs:      {len(test_idx)}")
print(f"  Train/val epochs: {len(train_val_idx)}")