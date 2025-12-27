#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
main_deap_pipeline.py (FINAL)

Purpose
-------
Compute a forehead EEG differential-entropy feature (Hₑ) from the DEAP dataset and export
trial-level CSVs that are *analysis-ready* and consistent across artifact-handling variants.

Key choices (fixed in code; report in Methods)
----------------------------------------------
- Dataset: DEAP subject files `s01.dat` ... `s32.dat` (pickle)
- EEG sampling rate: 128 Hz
- Baseline: first 3 s (384 samples) are used for baseline correction and then discarded
- Stimulus segment analyzed: remaining 60 s (7680 samples)
- Windowing: 4 s windows (512 samples) with 50% overlap (step = 2 s = 256 samples)
  => 29 windows per trial
- Differential entropy (Gaussian approximation):
    h = 0.5 * ln(2*pi*e*sigma^2)
- Forehead channels: Fp1 and Fp2 (DEAP standard ordering)
- Channel combination for Hₑ:
    --combine_mode avg_signal: average Fp1/Fp2 time series first, then compute h
    --combine_mode avg_entropy: compute h per-channel then average entropies
- Artifact flagging:
    A window is flagged if VAR(avg_signal_window) >= --artifact_var.
    (Artifact flagging is intentionally defined on avg_signal variance to match the
     threshold calibration script `calibrate_artifact_threshold_fp1fp2.py`.)

Outputs
-------
In --out_dir, the script writes 3 CSV files (same columns):
- deap_entropy_primary.csv
- deap_entropy_artifact_windows_removed.csv
- deap_entropy_artifact_excluded.csv

Columns (all outputs)
---------------------
participant, trial, H_e, valence, arousal, n_windows, n_flagged, flagged_fraction, any_flagged

Notes
-----
- `n_windows` is the total number of windows (typically 29).
- In WINDOWS_REMOVED, Hₑ is computed using only clean windows; the number of clean windows is
  `n_windows - n_flagged`. Trials are dropped if clean windows < --min_clean_windows.
- In EXCLUDED, trials with any flagged window are removed entirely (so remaining rows have n_flagged=0).

Example (Windows CMD)
---------------------
cd /d C:\Users\DELL\Desktop\Uni\Amir\deap_entropy_null
.venv\Scripts\activate

python main_deap_pipeline.py ^
  --data_dir data ^
  --out_dir results_v3_fp12_p995 ^
  --artifact_var 26478.34 ^
  --label_min 1 --label_max 9 ^
  --min_clean_windows 20 ^
  --combine_mode avg_signal
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import detrend

# -------------------------------
# Fixed DEAP constants
# -------------------------------
FS = 128  # Hz
BASELINE_SEC = 3
BASELINE_SAMPLES = BASELINE_SEC * FS  # 384
STIM_SEC = 60
STIM_SAMPLES = STIM_SEC * FS  # 7680

WIN_SEC = 4
WIN_SAMPLES = WIN_SEC * FS  # 512
STEP_SAMPLES = WIN_SAMPLES // 2  # 50% overlap => 256
N_WINDOWS_EXPECTED = 1 + (STIM_SAMPLES - WIN_SAMPLES) // STEP_SAMPLES  # 29


DEFAULT_CHANNELS = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3","O1","Oz","Pz",
    "Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz","C4","T8","CP6","CP2","P4","P8","PO4","O2"
]


def load_deap_dat(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def find_channel_index(ch_names: List[str], target: str) -> int:
    # tolerant match: case-insensitive and strip
    t = target.strip().lower()
    for i, nm in enumerate(ch_names):
        if str(nm).strip().lower() == t:
            return i
    raise ValueError(f"Channel '{target}' not found in channel_names.")


def baseline_correct_drop(x: np.ndarray) -> np.ndarray:
    """Subtract baseline mean (first 3s) and return post-baseline 60s segment."""
    if x.shape[-1] < BASELINE_SAMPLES + STIM_SAMPLES:
        raise ValueError(f"Signal too short: need >= {BASELINE_SAMPLES + STIM_SAMPLES} samples, got {x.shape[-1]}")
    base = x[..., :BASELINE_SAMPLES].mean(axis=-1, keepdims=True)
    y = x - base
    return y[..., BASELINE_SAMPLES:BASELINE_SAMPLES + STIM_SAMPLES]


def sliding_windows_1d(x: np.ndarray) -> np.ndarray:
    """Return array of shape (n_windows, WIN_SAMPLES)."""
    n = x.shape[-1]
    if n != STIM_SAMPLES:
        raise ValueError(f"Expected STIM_SAMPLES={STIM_SAMPLES}, got {n}")
    wins = []
    for start in range(0, n - WIN_SAMPLES + 1, STEP_SAMPLES):
        wins.append(x[start:start + WIN_SAMPLES])
    W = np.stack(wins, axis=0)
    if W.shape[0] != N_WINDOWS_EXPECTED:
        # Not fatal, but warn via exception message
        raise ValueError(f"Unexpected window count: got {W.shape[0]}, expected {N_WINDOWS_EXPECTED}")
    return W


def gaussian_de_from_var(var: np.ndarray) -> np.ndarray:
    """h = 0.5 * ln(2*pi*e*var)."""
    eps = 1e-12
    return 0.5 * np.log(2.0 * np.pi * np.e * (var + eps))


def compute_trial_features(
    eeg_trial: np.ndarray,
    ch_fp1: int,
    ch_fp2: int,
    do_detrend: bool,
    combine_mode: str,
    artifact_var: Optional[float],
) -> Tuple[float, int, int, float, int]:
    """
    Returns:
      H_e_primary, n_windows, n_flagged, flagged_fraction, any_flagged
    """
    fp1 = eeg_trial[ch_fp1, :].astype(float)
    fp2 = eeg_trial[ch_fp2, :].astype(float)

    fp1 = baseline_correct_drop(fp1)
    fp2 = baseline_correct_drop(fp2)

    if do_detrend:
        fp1 = detrend(fp1, type="linear")
        fp2 = detrend(fp2, type="linear")

    # windows of the avg_signal (used both for artifact detection and for avg_signal H_e)
    avg_signal = 0.5 * (fp1 + fp2)
    W_sig = sliding_windows_1d(avg_signal)  # (29, 512)
    var_sig = W_sig.var(axis=1, ddof=0)

    if artifact_var is None:
        flagged = np.zeros(W_sig.shape[0], dtype=bool)
    else:
        flagged = var_sig >= float(artifact_var)

    n_windows = int(W_sig.shape[0])
    n_flagged = int(flagged.sum())
    flagged_fraction = float(n_flagged / n_windows) if n_windows > 0 else float("nan")
    any_flagged = int(n_flagged > 0)

    # compute window entropies per chosen combine_mode
    if combine_mode == "avg_signal":
        h_w = gaussian_de_from_var(var_sig)
    elif combine_mode == "avg_entropy":
        W_fp1 = sliding_windows_1d(fp1)
        W_fp2 = sliding_windows_1d(fp2)
        h1 = gaussian_de_from_var(W_fp1.var(axis=1, ddof=0))
        h2 = gaussian_de_from_var(W_fp2.var(axis=1, ddof=0))
        h_w = 0.5 * (h1 + h2)
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")

    H_e_primary = float(h_w.mean())
    return H_e_primary, n_windows, n_flagged, flagged_fraction, any_flagged


def parse_exclude_list(s: str) -> set:
    s = (s or "").strip()
    if not s:
        return set()
    items = [x.strip() for x in s.split(",") if x.strip()]
    return set(items)


def label_rescale_auto(L: np.ndarray) -> np.ndarray:
    """
    If labels look like [0,1] floats, rescale to [1,9]. Otherwise keep unchanged.
    """
    if np.nanmin(L) >= -1e-9 and np.nanmax(L) <= 1.0 + 1e-9:
        return 1.0 + 8.0 * L
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing DEAP s01.dat ...")
    ap.add_argument("--out_dir", required=True, help="Output folder")

    ap.add_argument("--artifact_var", type=float, default=None, help="Variance threshold for artifact windows (on avg_signal windows)")
    ap.add_argument("--combine_mode", choices=["avg_signal", "avg_entropy"], default="avg_signal")
    ap.add_argument("--detrend", action="store_true", help="Linear detrend after baseline correction")

    ap.add_argument("--exclude_participants", default="", help="Comma-separated IDs (e.g., s11,s22)")
    ap.add_argument("--max_flagged_fraction_trial", type=float, default=None,
                    help="If set, drop any trial with flagged_fraction > this value (applies to all outputs).")

    ap.add_argument("--min_clean_windows", type=int, default=10,
                    help="Minimum clean windows required for WINDOWS_REMOVED output (clean = n_windows - n_flagged).")

    ap.add_argument("--label_min", type=float, default=None, help="Drop trials where valence/arousal < label_min")
    ap.add_argument("--label_max", type=float, default=None, help="Drop trials where valence/arousal > label_max")
    ap.add_argument("--label_rescale", choices=["none", "auto"], default="none",
                    help="Optional rescale for labels (auto rescales [0,1] -> [1,9]).")

    ap.add_argument("--fp1_idx", type=int, default=None, help="Override Fp1 channel index (0-based)")
    ap.add_argument("--fp2_idx", type=int, default=None, help="Override Fp2 channel index (0-based)")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = parse_exclude_list(args.exclude_participants)

    subject_files = sorted([p for p in data_dir.glob("s*.dat") if p.is_file()])
    if not subject_files:
        raise FileNotFoundError(f"No s*.dat files found in: {data_dir}")

    rows_primary = []
    rows_win_removed = []
    rows_excluded = []

    total_windows = 0
    total_flagged = 0
    trials_any_flag = 0
    trials_total = 0

    for f in subject_files:
        pid = f.stem  # "s01"
        if pid in exclude:
            continue

        d = load_deap_dat(f)
        X = np.asarray(d.get("data", None))
        L = np.asarray(d.get("labels", None))
        if X is None or L is None:
            raise ValueError(f"{f.name} missing 'data' or 'labels'.")

        # Expect X: (40, 40, 8064) in DEAP (trials, channels, samples). We only need first 32 EEG channels.
        if X.ndim != 3:
            raise ValueError(f"{f.name}: expected X.ndim==3, got {X.ndim}")
        if L.ndim != 2 or L.shape[0] != X.shape[0]:
            raise ValueError(f"{f.name}: labels shape mismatch with data. data trials={X.shape[0]}, labels={L.shape}")

        ch_names = d.get("channel_names", None)
        if ch_names is None:
            ch_names = DEFAULT_CHANNELS
        ch_names = list(ch_names)

        # Resolve Fp1/Fp2 indices
        if args.fp1_idx is not None and args.fp2_idx is not None:
            ch_fp1, ch_fp2 = int(args.fp1_idx), int(args.fp2_idx)
        else:
            ch_fp1 = find_channel_index(ch_names, "Fp1")
            ch_fp2 = find_channel_index(ch_names, "Fp2")

        # Keep only EEG channels (first 32) if needed
        # But some pipelines already store EEG first; we safely index channels we need.
        for t in range(X.shape[0]):
            eeg_trial = X[t, :32, :]
            labs = L[t].astype(float)

            if args.label_rescale == "auto":
                labs = label_rescale_auto(labs)

            valence = float(labs[0])
            arousal = float(labs[1])

            # label filtering
            if args.label_min is not None:
                if (valence < args.label_min) or (arousal < args.label_min):
                    continue
            if args.label_max is not None:
                if (valence > args.label_max) or (arousal > args.label_max):
                    continue

            H_e_primary, n_windows, n_flagged, flagged_fraction, any_flagged = compute_trial_features(
                eeg_trial=eeg_trial,
                ch_fp1=ch_fp1,
                ch_fp2=ch_fp2,
                do_detrend=bool(args.detrend),
                combine_mode=str(args.combine_mode),
                artifact_var=args.artifact_var,
            )

            # trial-level artifact filter (applies to all outputs)
            if args.max_flagged_fraction_trial is not None and flagged_fraction > float(args.max_flagged_fraction_trial):
                continue

            trials_total += 1
            total_windows += n_windows
            total_flagged += n_flagged
            trials_any_flag += int(any_flagged > 0)

            base_row = {
                "participant": pid,
                "trial": int(t),
                "H_e": float(H_e_primary),  # for PRIMARY; may be replaced below for WIN_REMOVED
                "valence": valence,
                "arousal": arousal,
                "n_windows": int(n_windows),
                "n_flagged": int(n_flagged),
                "flagged_fraction": float(flagged_fraction),
                "any_flagged": int(any_flagged),
            }

            # PRIMARY
            rows_primary.append(dict(base_row))

            # EXCLUDED: drop trials with any flagged window
            if any_flagged == 0:
                rows_excluded.append(dict(base_row))

            # WINDOWS_REMOVED: recompute H_e on clean windows
            if args.artifact_var is None or n_flagged == 0:
                # same as primary
                rows_win_removed.append(dict(base_row))
            else:
                clean_windows = n_windows - n_flagged
                if clean_windows < int(args.min_clean_windows):
                    continue

                # recompute clean-window H_e (must recompute windows + entropies)
                fp1 = eeg_trial[ch_fp1, :].astype(float)
                fp2 = eeg_trial[ch_fp2, :].astype(float)
                fp1 = baseline_correct_drop(fp1)
                fp2 = baseline_correct_drop(fp2)
                if args.detrend:
                    fp1 = detrend(fp1, type="linear")
                    fp2 = detrend(fp2, type="linear")

                avg_signal = 0.5 * (fp1 + fp2)
                W_sig = sliding_windows_1d(avg_signal)
                var_sig = W_sig.var(axis=1, ddof=0)
                flagged = var_sig >= float(args.artifact_var)
                keep = ~flagged

                if args.combine_mode == "avg_signal":
                    h_w = gaussian_de_from_var(var_sig)
                else:
                    W_fp1 = sliding_windows_1d(fp1)
                    W_fp2 = sliding_windows_1d(fp2)
                    h1 = gaussian_de_from_var(W_fp1.var(axis=1, ddof=0))
                    h2 = gaussian_de_from_var(W_fp2.var(axis=1, ddof=0))
                    h_w = 0.5 * (h1 + h2)

                H_clean = float(h_w[keep].mean()) if keep.any() else float("nan")
                r = dict(base_row)
                r["H_e"] = H_clean
                rows_win_removed.append(r)

    # Save CSVs
    df_primary = pd.DataFrame(rows_primary)
    df_excluded = pd.DataFrame(rows_excluded)
    df_win_removed = pd.DataFrame(rows_win_removed)

    p1 = out_dir / "deap_entropy_primary.csv"
    p2 = out_dir / "deap_entropy_artifact_excluded.csv"
    p3 = out_dir / "deap_entropy_artifact_windows_removed.csv"

    df_primary.to_csv(p1, index=False)
    df_excluded.to_csv(p2, index=False)
    df_win_removed.to_csv(p3, index=False)

    # Print artifact summary on PRIMARY rows (after label filters / exclusions by fraction threshold)
    rate = (total_flagged / total_windows) if total_windows > 0 else float("nan")
    trial_any_rate = (trials_any_flag / trials_total) if trials_total > 0 else float("nan")
    print(f"Saved: {p1}")
    print(f"Artifact summary (PRIMARY rows used): total_windows={total_windows}, total_flagged={total_flagged}, rate={rate:.6f}")
    print(f"Trials with any flagged window: {trial_any_rate:.4f}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Rows primary: {df_primary.shape[0]}")


if __name__ == "__main__":
    main()
