#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_deap_results_mixedlm.py (FINAL)

Reports:
- Pearson r (trial-level, descriptive)
- OLS (naive)
- OLS with cluster-robust SEs clustered by participant (simple repeated-measures inference)
- Mixed-effects model with random intercept for participant (MixedLM) [exploratory]

Model:
  H_e ~ predictor + (1 | participant)

Important
---------
If within- and between-participant associations differ in sign (Simpson's paradox),
MixedLM (random intercept only) may not match the naive correlation. For disaggregation,
use `deap_within_between.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


FILES = [
    ("PRIMARY", "deap_entropy_primary.csv"),
    ("WIN_REMOVED", "deap_entropy_artifact_windows_removed.csv"),
    ("EXCLUDED", "deap_entropy_artifact_excluded.csv"),
]


def ols_report(df: pd.DataFrame, pred: str):
    y = df["H_e"].astype(float)
    X = sm.add_constant(df[pred].astype(float))
    res = sm.OLS(y, X).fit()
    return float(res.params[pred]), float(res.pvalues[pred]), float(res.rsquared), res


def ols_cluster_report(df: pd.DataFrame, pred: str, cluster_col: str = "participant"):
    y = df["H_e"].astype(float)
    X = sm.add_constant(df[pred].astype(float))
    res = sm.OLS(y, X).fit()
    res_cl = res.get_robustcov_results(cov_type="cluster", groups=df[cluster_col].astype(str))
    # res_cl.pvalues order matches params
    beta = float(res_cl.params[1])
    pval = float(res_cl.pvalues[1])
    return beta, pval, float(res.rsquared), res_cl


def mixedlm_report(df: pd.DataFrame, pred: str):
    endog = df["H_e"].astype(float)
    exog = sm.add_constant(df[pred].astype(float))
    groups = df["participant"].astype(str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MixedLM(endog=endog, exog=exog, groups=groups)
        res = model.fit(reml=False, method="lbfgs", disp=False)

    beta = float(res.params[pred])
    pval = float(res.pvalues[pred]) if hasattr(res, "pvalues") and pred in res.pvalues else float("nan")
    return beta, pval, int(res.nobs), int(df["participant"].nunique()), res


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
        df = df.dropna(subset=["H_e", "valence", "arousal", "participant"])
        print(f"\n[{tag}] {fn}: rows={df.shape[0]}, participants={df['participant'].nunique()}")

        for pred in ["valence", "arousal"]:
            r, pr = pearsonr(df["H_e"].astype(float), df[pred].astype(float))
            print(f"Pearson H_e ~ {pred}: r={r:.4f}, p={pr:.6g}, n={df.shape[0]}")

            b, pv, r2, _ = ols_report(df, pred)
            print(f"OLS      H_e ~ {pred}: beta={b:.4f}, p={pv:.6g}, R^2={r2:.4f}")

            bc, pvc, r2c, _ = ols_cluster_report(df, pred)
            print(f"OLS-CL   H_e ~ {pred}: beta={bc:.4f}, p={pvc:.6g}, R^2={r2c:.4f} (cluster=participant)")

            try:
                bm, pvm, nobs, ng, _ = mixedlm_report(df, pred)
                print(f"MixedLM  H_e ~ {pred}: beta={bm:.4f}, p={pvm:.6g}, n={nobs}, groups={ng} (1|participant)")
            except Exception as e:
                print(f"MixedLM  H_e ~ {pred}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
