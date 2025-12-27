# Frontal EEG Differential Entropy and Subjective Valence in DEAP

Analysis code for the manuscript: **"Frontal EEG Differential Entropy and Subjective Valence in the DEAP Dataset: Artifact-Calibrated Analysis with Within–Between Decomposition"**

**Authors:** Amirmohammad Lotfifar, Melika Dabbagh Mohammadi

**Preprint:** [Link to PsyArXiv preprint - will be added upon publication]

---

## Overview

This repository contains the complete analysis pipeline for investigating the relationship between frontal EEG differential entropy and subjective valence ratings using the DEAP dataset. The analysis emphasizes artifact calibration, within-between decomposition, and reliability assessment.

### Key Features
- Artifact-calibrated preprocessing with percentile-based thresholds
- Multiple statistical approaches (OLS, mixed-effects models, within-between decomposition)
- Comprehensive reliability analysis (split-half, temporal stability)
- Three artifact-handling variants for sensitivity analysis
- Reproducible figure generation

---

## Repository Structure

```
DEAP-entropy-valence-analysis/
├── scripts/           # Main analysis scripts
├── data/             # Output CSV files from pipeline
├── figures/          # Generated manuscript figures
├── requirements.txt  # Python dependencies
├── USAGE.txt        # Detailed usage instructions
└── README.md        # This file
```

---

## Requirements

### Dataset
This analysis uses the **DEAP dataset** (Database for Emotion Analysis using Physiological Signals):
- **Citation:** Koelstra et al. (2012). IEEE Transactions on Affective Computing, 3(1), 18-31
- **Download:** http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- Preprocessed EEG data (32 channels, 32 participants, 40 trials each)

### Python Dependencies
See `requirements.txt` for complete list. Main packages:
- Python 3.8+
- NumPy, Pandas, SciPy
- Statsmodels (for mixed-effects models)
- Matplotlib, Seaborn (for visualization)
- MNE (for EEG processing)

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Analysis Pipeline

### 1. Artifact Threshold Calibration
```bash
python scripts/calibrate_artifact_threshold_fp1fp2.py
```
**Purpose:** Calibrates the percentile-based variance threshold (p99.5) for artifact detection on Fp1/Fp2 channels.

**Output:** Threshold value and diagnostic plots

---

### 2. Main Feature Extraction Pipeline
```bash
python scripts/main_deap_pipeline.py
```
**Purpose:** Core pipeline for differential entropy computation and artifact handling.

**Steps:**
- Load DEAP dataset
- Baseline correction
- 4-second windowing with 50% overlap (29 windows per trial)
- Differential entropy computation per window
- Artifact flagging using calibrated threshold
- Trial-level aggregation across three variants:
  - PRIMARY: All windows included
  - WIN_REMOVED: Flagged windows removed
  - EXCLUDED: Trials with any flagged window excluded

**Output:** 
- `data/deap_entropy_primary.csv`
- `data/deap_entropy_artifact_windows_removed.csv`
- `data/deap_entropy_artifact_excluded.csv`

---

### 3. Statistical Analysis

#### 3a. Basic Analysis (Naive Correlations, OLS)
```bash
python scripts/analyze_deap_results.py
```
**Output:** Trial-level correlations, R² values, basic regression models

#### 3b. Mixed-Effects Models
```bash
python scripts/analyze_deap_results_mixedlm.py
python scripts/analyze_deap_results_mixedlm_v2.py
```
**Output:** Random-intercept models, cluster-robust standard errors

#### 3c. Within-Between Decomposition
```bash
python scripts/deap_within_between.py
```
**Purpose:** Separates within-participant from between-participant associations.

**Output:** Between-participant correlations, within-participant effects, joint models

---

### 4. Manuscript Figure Generation
```bash
python scripts/build_manuscript_with_figures.py
```
**Purpose:** Generates all figures for the manuscript with publication-ready formatting.

**Output:** Figures saved to `figures/` directory

---

## Key Results

### Primary Findings
- **Naive trial-level association:** r = −0.086, p = 0.004, R² = 0.007
- **Cluster-robust analysis:** p = 0.125 (non-significant when accounting for repeated measures)
- **Within-between decomposition:** Association primarily between participants (r = −0.285, p = 0.114) rather than within participants (β ≈ 0.006, p = 0.186)
- **Reliability:** Excellent (split-half ICC = 0.990, 95% CI [0.988, 0.991])

### Interpretation
Frontal differential entropy shows high measurement reliability but only a very small association with subjective valence. After accounting for repeated measures, evidence suggests the relationship reflects stable between-person differences rather than within-person emotional fluctuations.

---

## Usage

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/[your-username]/DEAP-entropy-valence-analysis.git
cd DEAP-entropy-valence-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download DEAP dataset (see link above)
# Place preprocessed data in appropriate directory

# 4. Run calibration
python scripts/calibrate_artifact_threshold_fp1fp2.py

# 5. Run main pipeline
python scripts/main_deap_pipeline.py

# 6. Run statistical analyses
python scripts/analyze_deap_results.py
python scripts/deap_within_between.py

# 7. Generate figures
python scripts/build_manuscript_with_figures.py
```

### Detailed Instructions
See `USAGE.txt` for comprehensive step-by-step instructions.

---

## Citation

If you use this code, please cite our preprint:

```bibtex
@article{lotfifar2024frontal,
  title={Frontal EEG Differential Entropy and Subjective Valence in the DEAP Dataset: Artifact-Calibrated Analysis with Within–Between Decomposition},
  author={Lotfifar, Amirmohammad and Mohammadi, Melika Dabbagh},
  journal={PsyArXiv},
  year={2024},
  doi={[DOI will be added upon publication]}
}
```

And the original DEAP dataset:

```bibtex
@article{koelstra2012deap,
  title={DEAP: A database for emotion analysis using physiological signals},
  author={Koelstra, Sander and Muhl, Christian and Soleymani, Mohammad and others},
  journal={IEEE Transactions on Affective Computing},
  volume={3},
  number={1},
  pages={18--31},
  year={2012},
  publisher={IEEE}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

**Corresponding Author:** Amirmohammad Lotfifar  
**Email:** amirlotfifar1994@gmail.com

---

## Acknowledgments

- DEAP dataset creators for making the data publicly available
- Center for Open Science for hosting preprints and code

---

## Notes

- This is a secondary analysis of the publicly available DEAP dataset
- All numerical results reported in the manuscript can be reproduced using these scripts
- For questions about specific implementation details, please open an issue on GitHub