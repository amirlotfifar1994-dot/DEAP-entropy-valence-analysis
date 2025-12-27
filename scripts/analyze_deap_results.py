#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_deap_results.py (FINAL)

Quick descriptive associations between H_e and labels.

Reports per file:
- Pearson correlation (trial-level)
- OLS regression (H_e ~ predictor)

This is *descriptive*; for inference with repeated measures, prefer:
- deap_within_between.py (recommended)
- analyze_deap_results_mixedlm.py (exploratory)
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


def ols_line(df: pd.DataFrame, pred: str):
    y = df["H_e"].astype(float)
    X = sm.add_constant(df[pred].astype(float))
    res = sm.OLS(y, X).fit()
    beta = float(res.params[pred])
    pval = float(res.pvalues[pred])
    r2 = float(res.rsquared)
    intercept = float(res.params["const"])
    return intercept, beta, pval, r2


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
            r, pr = pearsonr(df["H_e"].astype(float), df[pred].astype(float))
            ic, b, pv, r2 = ols_line(df, pred)
            print(f"H_e ~ {pred}: r={r:.4f}, p={pr:.4g}, n={df.shape[0]}")
            print(f"OLS      H_e ~ {pred}: beta={b:.4f}, p={pv:.4g}, R^2={r2:.4f}")
            # also print inverse line (label ~ H_e) for intuition
            # (purely descriptive; not used for inference)
            y2 = df[pred].astype(float)
            X2 = sm.add_constant(df["H_e"].astype(float))
            res2 = sm.OLS(y2, X2).fit()
            print(f"OLS {pred} ~ H_e: {pred}={res2.params['const']:.4f} + ({res2.params['H_e']:.4f})*H_e, R^2={res2.rsquared:.4f}, p={res2.pvalues['H_e']:.4g}, n={int(res2.nobs)}")


if __name__ == "__main__":
    main()
