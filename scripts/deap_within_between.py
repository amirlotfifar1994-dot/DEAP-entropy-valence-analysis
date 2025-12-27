#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deap_within_between.py (FINAL)

Disaggregates H_e ~ (valence/arousal) into:
- Naive trial-level Pearson (descriptive only)
- Between-participant association using participant means
- Within-participant association using centered predictor with cluster-robust SEs
- Joint model including both within + between terms (cluster-robust SEs)

This is the recommended analysis to avoid Simpson's paradox in repeated-measures DEAP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


FILES = [
    ("PRIMARY", "deap_entropy_primary.csv"),
    ("WIN_REMOVED", "deap_entropy_artifact_windows_removed.csv"),
    ("EXCLUDED", "deap_entropy_artifact_excluded.csv"),
]


def cluster_robust_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    res = sm.OLS(y, X).fit()
    res_cl = res.get_robustcov_results(cov_type="cluster", groups=groups)
    return res, res_cl


def report(df: pd.DataFrame, pred: str):
    df = df.dropna(subset=["H_e", pred, "participant"])
    y = df["H_e"].astype(float).to_numpy()
    x = df[pred].astype(float).to_numpy()
    g = df["participant"].astype(str).to_numpy()

    # naive Pearson (trial-level)
    r, pr = pearsonr(y, x)

    # between: participant means
    means = df.groupby("participant")[["H_e", pred]].mean(numeric_only=True).reset_index()
    rb, prb = pearsonr(means["H_e"].astype(float), means[pred].astype(float))

    # within: center x by participant mean
    df2 = df.copy()
    df2["x_mean"] = df2.groupby("participant")[pred].transform("mean")
    df2["x_within"] = df2[pred] - df2["x_mean"]
    df2["x_between"] = df2["x_mean"]

    # within-only model
    Xw = sm.add_constant(df2["x_within"].astype(float))
    resw, resw_cl = cluster_robust_ols(df2["H_e"].astype(float), Xw, df2["participant"].astype(str))
    within_beta = float(resw_cl.params[1])
    within_p = float(resw_cl.pvalues[1])

    # joint model
    Xj = sm.add_constant(df2[["x_within", "x_between"]].astype(float))
    resj, resj_cl = cluster_robust_ols(df2["H_e"].astype(float), Xj, df2["participant"].astype(str))
    joint_within_beta = float(resj_cl.params[1])
    joint_within_p = float(resj_cl.pvalues[1])
    joint_between_beta = float(resj_cl.params[2])
    joint_between_p = float(resj_cl.pvalues[2])

    return {
        "pearson_r": float(r),
        "pearson_p": float(pr),
        "n_trials": int(df.shape[0]),

        "between_r": float(rb),
        "between_p": float(prb),
        "n_participants": int(means.shape[0]),

        "within_beta": within_beta,
        "within_p": within_p,

        "joint_within_beta": joint_within_beta,
        "joint_within_p": joint_within_p,
        "joint_between_beta": joint_between_beta,
        "joint_between_p": joint_between_p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing deap_entropy_*.csv files")
    args = ap.parse_args()
    d = Path(args.dir)

    for tag, fn in FILES:
        p = d / fn
        if not p.exists():
            continue
        df = pd.read_csv(p)
        print(f"\n[{tag}] {fn}: rows={df.shape[0]}, participants={df['participant'].nunique()}")

        for pred in ["valence", "arousal"]:
            o = report(df, pred)
            print(f"\n  Predictor: {pred}")
            print(f"    Naive Pearson (trial-level): r={o['pearson_r']:.4f}, p={o['pearson_p']:.6g}, n={o['n_trials']}")
            print(f"    Between participants (means): r={o['between_r']:.4f}, p={o['between_p']:.6g}, n_participants={o['n_participants']}")
            print(f"    Within (centered) OLS-CL: beta={o['within_beta']:.4f}, p={o['within_p']:.6g} (cluster=participant)")
            print(f"    Joint (within + between) OLS-CL:")
            print(f"      within beta={o['joint_within_beta']:.4f}, p={o['joint_within_p']:.6g}")
            print(f"      between beta={o['joint_between_beta']:.4f}, p={o['joint_between_p']:.6g}")


if __name__ == "__main__":
    main()
