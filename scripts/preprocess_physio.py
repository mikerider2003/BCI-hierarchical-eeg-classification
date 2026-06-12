"""Separate preprocessing pipeline for the GSR + PPG auxiliary channels.

The goal is per-epoch physiological data that is aligned 1:1 with the EEG epochs
already in ``data/ds007537/all_subjects_preprocessed.pkl``. If the two arrays
share the same row order, the precomputed cross-subject splits
(``outputs/groupkfold_splits.pkl``) stay valid and EEG/physio can be fused
trial-by-trial.

GSR and PPG are physiological aux channels (type MISC) recorded inside the same
BrainVision file as the EEG, on the same clock, so they need no resynchronisation
-- only modality-appropriate filtering:
    GSR : low-pass 5 Hz, keeping the tonic skin-conductance level (no high-pass).
    PPG : band-pass 0.5-8 Hz, the pulse-waveform band.

Alignment problem and strategy
------------------------------
The EEG pipeline (see Preprocessing.ipynb) drops epochs with AutoReject, and the
surviving-epoch indices were never stored. Every step is seeded
(``random_state=97``, extended-infomax ICA, AutoReject), so we re-run the EEG
pipeline per subject purely to recover ``reject_log.bad_epochs``, then apply that
mask to GSR/PPG epochs built on the identical 2 s windows. Each subject is then
validated against the EEG pkl (epoch count + exact label order); the script
refuses to write a subject that does not match.

Outputs
-------
    data/ds007537/<sub>_physio.pkl         per subject
    data/ds007537/all_subjects_physio.pkl  combined, row-aligned with the EEG pkl

Run
---
    python scripts/preprocess_physio.py                  # all subjects + combined
    python scripts/preprocess_physio.py --subjects sub-13   # subset (testing)
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import mne
from mne.preprocessing import ICA
from pyprep import NoisyChannels
from autoreject import AutoReject

mne.set_log_level("ERROR")

DATA_ROOT = Path("data/ds007537")
EEG_PKL = DATA_ROOT / "all_subjects_preprocessed.pkl"
COMBINED_OUT = DATA_ROOT / "all_subjects_physio.pkl"
PHYSIO_CH = ["GSR", "PPG"]
APP_TYPE_NAMES = {"S": "short_videos", "G": "gaming", "R": "reading"}
EPOCH_LEN = 2.0
SEED = 97


def label_order_hash(labels) -> str:
    return hashlib.md5("|".join(labels).encode()).hexdigest()[:8]


def eeg_ground_truth(eeg_pkl=EEG_PKL):
    """Per-subject epoch count + label-order hash from the existing EEG pkl."""
    d = pickle.load(open(eeg_pkl, "rb"))
    g, y = d["groups"], d["y"]
    order = list(dict.fromkeys(g.tolist()))
    truth = {}
    for s in order:
        labels = y[g == s].tolist()
        truth[s] = {"n": len(labels), "hash": label_order_hash(labels), "labels": labels}
    return order, truth


def block_spans(eeg_dir, subject):
    events_df = pd.read_csv(eeg_dir / f"{subject}_task-phoneuse_events.tsv", sep="\t")
    stim = (
        events_df[events_df["trial_type"].str.startswith("Stimulus", na=False)]
        .drop_duplicates(subset=["onset", "trial_type", "value", "sample"])
        .reset_index(drop=True)
    )

    def span(onset_code, offset_code):
        return (
            float(stim.loc[stim["value"] == onset_code, "onset"].min()),
            float(stim.loc[stim["value"] == offset_code, "onset"].max()),
        )

    return span(12, 13), span(22, 23)  # (smartphone), (video)


def fixed_length_block_epochs(raw, label, start, end, event_id):
    """Non-overlapping EPOCH_LEN epochs over one cropped block, relabelled."""
    block = raw.copy().crop(tmin=start, tmax=min(end, raw.times[-1]))
    ep = mne.make_fixed_length_epochs(block, duration=EPOCH_LEN, preload=True)
    ep.events[:, 2] = event_id[label]
    ep.event_id = {label: event_id[label]}
    return ep


def eeg_reject_mask(subject):
    """Re-run the seeded EEG pipeline; return (keep_mask, surviving_labels, spans, event_id).

    ``keep_mask`` is a boolean over the pre-AutoReject epochs (video block then
    smartphone block, in that order). This mirrors Preprocessing.ipynb exactly so
    the mask reproduces the stored EEG epochs.
    """
    eeg_dir = DATA_ROOT / subject / "eeg"
    raw = mne.io.read_raw_brainvision(eeg_dir / f"{subject}_task-phoneuse_eeg.vhdr", preload=True)

    participants = pd.read_csv(DATA_ROOT / "participants.tsv", sep="\t")
    row = participants.loc[participants["participant_id"] == subject].iloc[0]
    smartphone_class = APP_TYPE_NAMES[row["app_type"]]

    elec = pd.read_csv(eeg_dir / f"{subject}_space-CapTrak_electrodes.tsv", sep="\t")
    ch_pos = {r["name"]: np.array([r["x"], r["y"], r["z"]], float) for _, r in elec.iterrows()}
    with open(eeg_dir / f"{subject}_space-CapTrak_coordsystem.json") as f:
        fids = json.load(f)["AnatomicalLandmarkCoordinates"]
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos, nasion=fids["NAS"], lpa=fids["LPA"], rpa=fids["RPA"], coord_frame="head"
    )
    raw.set_montage(montage, on_missing="warn")
    raw.info["line_freq"] = 50.0

    (smart_start, smart_end), (video_start, video_end) = block_spans(eeg_dir, subject)

    raw_eeg = raw.copy().pick("eeg")
    raw_eeg.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin")
    psd = raw_eeg.compute_psd(fmax=80)
    freqs, power = psd.freqs, psd.get_data().mean(axis=0)
    band = lambda lo, hi: power[(freqs >= lo) & (freqs <= hi)].mean()
    if band(49, 51) / band(43, 47) > 2.0:
        raw_eeg.notch_filter(freqs=[50, 100], fir_design="firwin")

    nc = NoisyChannels(raw_eeg, random_state=SEED)
    nc.find_all_bads()
    bads = nc.get_bads()
    raw_eeg.info["bads"] = bads
    if bads:
        raw_eeg.interpolate_bads(reset_bads=True)
    raw_eeg.set_eeg_reference("average", projection=False)

    if raw_eeg.info["sfreq"] > 250:
        raw_eeg.resample(250)

    event_id = {"video": 1, f"smartphone/{smartphone_class}": 2}
    epochs = mne.concatenate_epochs([
        fixed_length_block_epochs(raw_eeg, "video", video_start, video_end, event_id),
        fixed_length_block_epochs(raw_eeg, f"smartphone/{smartphone_class}", smart_start, smart_end, event_id),
    ])

    ica = ICA(n_components=0.99, method="infomax", fit_params=dict(extended=True),
              max_iter="auto", random_state=SEED)
    ica.fit(epochs, decim=3)
    eog_idx, _ = ica.find_bads_eog(epochs, ch_name=["Fp1", "Fp2"])
    mus_idx, _ = ica.find_bads_muscle(epochs)
    ica.exclude = sorted(set(eog_idx + mus_idx))
    epochs_ica = epochs.copy()
    ica.apply(epochs_ica)

    ar = AutoReject(random_state=SEED, n_jobs=-1, verbose=False)
    epochs_clean, reject_log = ar.fit_transform(epochs_ica, return_log=True)

    keep_mask = ~np.asarray(reject_log.bad_epochs, dtype=bool)
    code_to_label = {c: n for n, c in event_id.items()}
    surviving_labels = [code_to_label[c] for c in epochs_clean.events[:, 2]]

    spans = {"smart": (smart_start, smart_end), "video": (video_start, video_end)}
    return keep_mask, surviving_labels, spans, event_id, len(epochs)


def physio_epochs(subject, spans, event_id, n_pre_expected):
    """Filtered GSR/PPG epochs on the identical 2 s windows (pre-AutoReject order)."""
    eeg_dir = DATA_ROOT / subject / "eeg"
    raw = mne.io.read_raw_brainvision(eeg_dir / f"{subject}_task-phoneuse_eeg.vhdr", preload=True)
    raw_phys = raw.copy().pick(PHYSIO_CH)

    # Modality-appropriate filtering (filter MISC channels explicitly by name).
    raw_phys.filter(l_freq=None, h_freq=5.0, picks=["GSR"], fir_design="firwin")
    raw_phys.filter(l_freq=0.5, h_freq=8.0, picks=["PPG"], fir_design="firwin")
    if raw_phys.info["sfreq"] > 250:
        raw_phys.resample(250)

    smart_label = next(k for k in event_id if k.startswith("smartphone"))
    epochs = mne.concatenate_epochs([
        fixed_length_block_epochs(raw_phys, "video", *spans["video"], event_id),
        fixed_length_block_epochs(raw_phys, smart_label, *spans["smart"], event_id),
    ])
    if len(epochs) != n_pre_expected:
        raise RuntimeError(
            f"{subject}: physio pre-AR epoch count {len(epochs)} != EEG pre-AR {n_pre_expected}"
        )
    return epochs  # order: video block then smartphone block


def process_subject(subject, truth):
    keep_mask, surviving_labels, spans, event_id, n_pre = eeg_reject_mask(subject)
    epochs_phys = physio_epochs(subject, spans, event_id, n_pre)

    data = epochs_phys.get_data()[keep_mask].astype(np.float32)  # (n, 2, 500)

    # ---- validate against the stored EEG epochs ----
    t = truth.get(subject)
    if t is None:
        raise RuntimeError(f"{subject}: not present in EEG pkl (skip)")
    if data.shape[0] != t["n"]:
        raise RuntimeError(f"{subject}: physio n={data.shape[0]} != EEG n={t['n']}")
    if label_order_hash(surviving_labels) != t["hash"]:
        raise RuntimeError(f"{subject}: label order differs from EEG pkl")

    return {
        "subject": subject,
        "data": data,                       # (n_epochs, 2, n_samples) float32
        "labels": surviving_labels,         # condition label per epoch (matches EEG)
        "ch_names": list(PHYSIO_CH),
        "sfreq": float(epochs_phys.info["sfreq"]),
        "epoch_length_s": EPOCH_LEN,
        "filters": {"GSR": "lowpass 5 Hz", "PPG": "bandpass 0.5-8 Hz"},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="*", default=None,
                   help="subset of subjects (e.g. sub-13). Default: all in EEG pkl. "
                        "Combined pkl is only written for a full run.")
    args = p.parse_args()

    order, truth = eeg_ground_truth()
    targets = args.subjects if args.subjects else order
    full_run = args.subjects is None

    results, failed = {}, {}
    for s in targets:
        print(f"=== {s} ===", flush=True)
        try:
            res = process_subject(s, truth)
            with open(DATA_ROOT / f"{s}_physio.pkl", "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            results[s] = res
            print(f"  ok: {res['data'].shape}  (validated vs EEG pkl)", flush=True)
        except Exception as e:
            failed[s] = repr(e)
            print(f"  FAILED: {e!r}", flush=True)

    print(f"\nDone: {len(results)} ok, {len(failed)} failed.")
    for s, err in failed.items():
        print(f"  {s}: {err}")

    if full_run and not failed and set(results) == set(order):
        # Assemble in the EEG pkl's subject order so rows match exactly.
        X = np.concatenate([results[s]["data"] for s in order], axis=0)
        y = np.array([lbl for s in order for lbl in results[s]["labels"]])
        groups = np.array([s for s in order for _ in results[s]["labels"]])
        combined = {
            "X": X,                          # (n_epochs, 2, n_samples) float32, [GSR, PPG]
            "y": y,
            "groups": groups,
            "ch_names": list(PHYSIO_CH),
            "sfreq": float(next(iter(results.values()))["sfreq"]),
            "classes": sorted(set(y.tolist())),
        }
        # Final guard: row-for-row alignment with the EEG pkl.
        eeg = pickle.load(open(EEG_PKL, "rb"))
        assert np.array_equal(groups, eeg["groups"]), "groups order != EEG pkl"
        assert np.array_equal(y, eeg["y"]), "labels order != EEG pkl"
        with open(COMBINED_OUT, "wb") as f:
            pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved combined -> {COMBINED_OUT}  X={X.shape} (aligned with EEG pkl)")
    elif full_run:
        print("\nCombined pkl NOT written (some subjects failed/missing). Fix and re-run.")


if __name__ == "__main__":
    main()
