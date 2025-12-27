#!/usr/bin/env python
"""
analyze_deap_results_mixedlm_v2.py

Reports:
- Pearson r (trial-level, for descriptives)
- OLS (naive)
- OLS with cluster-robust SEs clustered by participant (recommended if you want simple inference)
- Mixed-effects model with random intercept for participant (MixedLM)

Models:
H_e ~ valence  (+ intercept)
H_e ~ arousal  (+ intercept)

Usage:
python analyze_deap_results_mixedlm_v2.py --dir results_v3_fp12_p995
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

DATASETS = [
    ("PRIMARY", "deap_entropy_primary.csv"),
    ("WIN_REMOVED", "deap_entropy_artifact_windows_removed.csv"),
    ("EXCLUDED", "deap_entropy_artifact_excluded.csv"),
]

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    needed = {"participant", "H_e", "valence", "arousal"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {sorted(missing)}")
    return df

def pearson_report(df: pd.DataFrame, x: str, y: str):
    r, p = stats.pearsonr(df[x].to_numpy(), df[y].to_numpy())
    return float(r), float(p), int(df.shape[0])

def ols_report(df: pd.DataFrame, pred: str):
    y = df["H_e"].astype(float).to_numpy()
    X = sm.add_constant(df[pred].astype(float).to_numpy())
    m = sm.OLS(y, X).fit()
    beta = float(m.params[1])
    pval = float(m.pvalues[1])
    r2 = float(m.rsquared)
    return beta, pval, r2, m

def ols_cluster_report(df: pd.DataFrame, pred: str):
    """
    OLS with cluster-robust SEs clustered by participant.
    """
    y = df["H_e"].astype(float).to_numpy()
    X = sm.add_constant(df[pred].astype(float).to_numpy())
    groups = df["participant"].astype(str).to_numpy()
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    beta = float(m.params[1])
    pval = float(m.pvalues[1])
    r2 = float(m.rsquared)
    return beta, pval, r2, m

def mixedlm_report(df: pd.DataFrame, pred: str):
    """
    Random-intercept mixed model:
      H_e ~ pred + (1 | participant)
    Fit with ML (reml=False) for comparability.
    """
    endog = df["H_e"].astype(float)
    exog = sm.add_constant(df[pred].astype(float))
    groups = df["participant"].astype(str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MixedLM(endog=endog, exog=exog, groups=groups)
        res = model.fit(reml=False, method="lbfgs", disp=False)

    beta = float(res.params[pred])
    # MixedLM returns pvalues (Wald z test) in most versions
    pval = float(res.pvalues[pred]) if hasattr(res, "pvalues") and pred in res.pvalues else float("nan")
    return beta, pval, int(res.nobs), int(df["participant"].nunique()), res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Results directory containing the CSVs")
    args = ap.parse_args()

    base = args.dir

    for tag, fname in DATASETS:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            # Skip missing variants gracefully
            continue

        df = _safe_read_csv(path).dropna(subset=["participant", "H_e", "valence", "arousal"])
        n = int(df.shape[0])
        ng = int(df["participant"].nunique())
        print(f"\n[{tag}] {fname}: rows={n}, participants={ng}")

        for pred in ["valence", "arousal"]:
            # Pearson (descriptive)
            r, p, n2 = pearson_report(df, "H_e", pred)
            print(f"Pearson H_e ~ {pred}: r={r:.4f}, p={p:.4g}, n={n2}")

            # OLS naive
            b, pv, r2, _ = ols_report(df, pred)
            print(f"OLS      H_e ~ {pred}: beta={b:.4f}, p={pv:.4g}, R^2={r2:.4f}")

            # OLS cluster-robust
            bc, pvc, r2c, _ = ols_cluster_report(df, pred)
            print(f"OLS-CL   H_e ~ {pred}: beta={bc:.4f}, p={pvc:.4g}, R^2={r2c:.4f} (cluster=participant)")

            # MixedLM
            try:
                bm, pvm, nobs, ng2, _ = mixedlm_report(df, pred)
                print(f"MixedLM  H_e ~ {pred}: beta={bm:.4f}, p={pvm:.4g}, n={nobs}, groups={ng2} (1|participant)")
            except Exception as e:
                print(f"MixedLM  H_e ~ {pred}: FAILED ({type(e).__name__}: {e})")

if __name__ == "__main__":
    main()
