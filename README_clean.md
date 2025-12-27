# DEAP Entropy Project (Clean + Paper-Ready)

## Keep these files (final)
### 1) Data-processing + feature extraction
- `main_deap_pipeline_v3.py`  ✅ (main pipeline that generates the `deap_entropy_*.csv` outputs)

### 2) Artifact-threshold calibration
- `calibrate_artifact_threshold_fp1fp2.py` ✅ (computes p99/p99.5/p99.9 suggested thresholds)

### 3) (Optional) Analysis checks
- `analyze_deap_results.py` ✅ (quick Pearson + OLS summaries)
- `analyze_deap_results_mixedlm_v2.py` ✅ (Pearson + OLS + OLS-CL + MixedLM)
- `deap_within_between.py` ✅ (within–between decomposition; the *key* disaggregation)

### 4) Paper + figures (NEW)
- `build_manuscript_with_figures.py` ✅ (creates **all figures** and a **DOCX manuscript fully aligned with your CSV results**)

### 5) Reproducible environment
- `requirements_deap.txt` ✅ (or `requirements.txt` if you prefer one)

## You can delete / ignore (obsolete)
- any `*_v1.py`, `*_v2.py` older variants you no longer use
- old plotting scripts you replaced (unless you want them)
- `reliability_icc.R*` if you use the Python reliability already (optional)

## Minimal workflow (Windows CMD)
### A) go to your project folder
```bat
cd /d C:\Users\DELL\Desktop\Uni\Amir\deap_entropy_null
```

### B) (optional) create venv + install deps
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_deap.txt
```

### C) Calibrate artifact threshold (Fp1/Fp2 avg_signal)
```bat
python calibrate_artifact_threshold_fp1fp2.py --data_dir data
```
Pick a threshold (usually p99.5) and copy the number.

### D) Run the main pipeline (produces CSVs in the output folder)
```bat
python main_deap_pipeline_v3.py --data_dir data --out_dir results_v3_fp12_p995 --artifact_var 26478.34 --label_min 1 --label_max 9 --min_clean_windows 20 --combine_mode avg_signal
```

### E) Build the Word manuscript + figures (fully data-aligned)
```bat
python build_manuscript_with_figures.py --results_dir results_v3_fp12_p995
```

Outputs:
- `results_v3_fp12_p995\figures\fig*.png`
- `DEAP_entropy_manuscript_WITH_FIGURES.docx`
