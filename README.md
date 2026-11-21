# Sigma Seal — Verifiable, Calibrated Anomaly Detection for Collider Physics

## What this is (month 1 scope)
A baseline unsupervised anomaly detector for particle-collision events, trained
only on background (Standard-Model-like) data, evaluated on its ability to
flag injected signal events as anomalous. This is the foundation step before
adding calibrated uncertainty (month 2) and zk-proof verification (month 3+).

## Dataset: LHC Olympics 2020 R&D dataset — HIGH-LEVEL FEATURES FILE
- Source: https://zenodo.org/record/4536377 (official R&D dataset used by the
  LHC Olympics 2020 anomaly detection challenge)
- **Download only `events_anomalydetection_v2.features.h5` (74.3 MB)** —
  not the raw 2.8GB particle file. This file already contains the standard
  6-feature representation (jet masses + n-subjettiness) used in published
  LHCO baselines like CATHODE, pre-extracted. Confirmed columns:
  `pxj1, pyj1, pzj1, mj1, tau1j1, tau2j1, tau3j1, pxj2, pyj2, pzj2, mj2,
  tau1j2, tau2j2, tau3j2, label`
- Why this dataset: it's a real benchmark used in published anomaly-detection
  papers, and at 74.3 MB the full 1.1M-event file loads and trains on CPU in
  minutes — no need to subsample for month 1. Background vs. injected-signal
  labels exist for evaluation ONLY — the model never sees labels during
  training (that's the point: unsupervised anomaly detection).

## Setup
```bash
pip install -r requirements.txt
```
Download `events_anomalydetection_v2.features.h5` from the Zenodo page above.

## Plan for month 1
1. `src/data_loader.py` — load the features file directly with
   `pd.read_hdf`, compute tau21/tau32 ratios from the raw tau1/2/3 columns.
2. `src/model.py` — simple dense autoencoder trained on background-only jets.
3. `src/train.py` — training loop + reconstruction-error anomaly score.
4. Validation: check your trained model's anomaly score actually separates
   injected signal from background on the held-out labeled set — this is
   your "does the pipeline even work" sanity check before anything novel.

## Explicitly NOT in scope yet
- Calibrated uncertainty (conformal prediction) — month 2
- zk-proof verification — month 3+
- Real ATLAS/CMS Open Data — later, once this pipeline is proven

## Results log

- Baseline (6 hand-picked features, features.h5 file): AUC 0.7789, 10% FPR calibration guarantee holds
- Self-contained raw-particle features (26 features: jet mass/pT/multiplicity + energy rings, 20k events): AUC 0.8102
- Self-contained raw-particle features (same 26 features, 100k events): AUC 0.7968, 9.8% empirical FPR vs 10% target
  - **This 100k result is the one to cite** -- the 20k run's higher AUC was a smaller-sample fluctuation, not a real improvement
- Note: discovered and fixed a row-alignment bug between events_anomalydetection.h5 and events_anomalydetection_v2.features.h5 (correlation ~0 between the two files' events) -- switched to deriving all features from the raw file alone to avoid it
