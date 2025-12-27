#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_artifact_threshold_fp1fp2.py (FINAL)

Computes empirical variance thresholds for artifact flagging on DEAP Fp1/Fp2 windows.

Definition (matches main_deap_pipeline.py)
-----------------------------------------
- baseline correct using first 3s; then discard baseline and keep 60s stimulus segment
- compute avg_signal = 0.5*(Fp1 + Fp2)
- windowing: 4s windows with 50% overlap
- compute window variance VAR(avg_signal_window)
- print quantiles: p99 (≈1%), p99.5 (≈0.5%), p99.9 (≈0.1%)

Usage (Windows CMD)
-------------------
python calibrate_artifact_threshold_fp1fp2.py --data_dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import numpy as np

FS = 128
BASELINE_SEC = 3
BASELINE_SAMPLES = BASELINE_SEC * FS
STIM_SEC = 60
STIM_SAMPLES = STIM_SEC * FS

WIN_SEC = 4
WIN_SAMPLES = WIN_SEC * FS
STEP_SAMPLES = WIN_SAMPLES // 2

DEFAULT_FP1_IDX = 0
DEFAULT_FP2_IDX = 16


def load_deap(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def baseline_correct_drop(x: np.ndarray) -> np.ndarray:
    base = x[:BASELINE_SAMPLES].mean()
    y = x - base
    return y[BASELINE_SAMPLES:BASELINE_SAMPLES + STIM_SAMPLES]


def window_vars(x: np.ndarray) -> np.ndarray:
    vars_ = []
    for start in range(0, len(x) - WIN_SAMPLES + 1, STEP_SAMPLES):
        w = x[start:start + WIN_SAMPLES]
        vars_.append(np.var(w, ddof=0))
    return np.asarray(vars_, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with s01.dat ... s32.dat")
    ap.add_argument("--fp1_idx", type=int, default=DEFAULT_FP1_IDX, help="Fp1 index (0-based; DEAP default=0)")
    ap.add_argument("--fp2_idx", type=int, default=DEFAULT_FP2_IDX, help="Fp2 index (0-based; DEAP default=16)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted([p for p in data_dir.glob("s*.dat") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No s*.dat found in {data_dir}")

    all_vars = []
    for f in files:
        d = load_deap(f)
        X = np.asarray(d.get("data", None))
        if X is None or X.ndim != 3:
            raise ValueError(f"{f.name}: missing or invalid 'data' array")
        for t in range(X.shape[0]):
            eeg = X[t, :32, :]
            fp1 = baseline_correct_drop(eeg[args.fp1_idx].astype(float))
            fp2 = baseline_correct_drop(eeg[args.fp2_idx].astype(float))
            avg = 0.5 * (fp1 + fp2)
            all_vars.append(window_vars(avg))

    vars_all = np.concatenate(all_vars, axis=0)
    p99 = float(np.quantile(vars_all, 0.99))
    p995 = float(np.quantile(vars_all, 0.995))
    p999 = float(np.quantile(vars_all, 0.999))

    print("Computed on Fp1/Fp2 combined (avg_signal), baseline-corrected, 4s windows, 50% overlap")
    print(f"Windows total: {vars_all.size}")
    print(f"--artifact_var {p995:.2f}   (p99.5: flags ~0.5% highest-variance windows)")
    print(f"--artifact_var {p99:.2f}   (p99.0: flags ~1.0%)")
    print(f"--artifact_var {p999:.2f}   (p99.9: flags ~0.1%)")


if __name__ == "__main__":
    main()
