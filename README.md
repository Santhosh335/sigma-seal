# Sigma Seal — Verifiable, Calibrated Anomaly Detection for Collider Physics

An unsupervised anomaly detector for LHC collision events that reports a
**calibrated confidence guarantee** instead of a bare score, and produces a
**cryptographic (zero-knowledge) proof** that a published result came from
the exact claimed model run on the exact claimed input — verifiable by
anyone, without access to the model weights or training data.

## What's actually in this repo

| File | What it does |
|---|---|
| `data_loader.py` | Loads the LHCO high-level features file (6-feature baseline) |
| `raw_features.py` | Self-contained feature extraction directly from raw particle data (26 features: jet mass/pT/multiplicity + energy rings), using a simplified jet-finding heuristic |
| `model.py` | The autoencoder architecture (shared by all training scripts) |
| `train.py` | Trains the baseline model, reports ROC-AUC |
| `calibrate_baseline.py` | Baseline model + conformal calibration (verified false-positive guarantee) |
| `calibrate_raw.py` | Same calibration pipeline, using the richer raw-particle features |
| `export_onnx.py` | Trains the baseline model and exports it to ONNX for the zk-proof pipeline |
| `generate_proof.py` | Full EZKL pipeline: settings → circuit → proof → verification, for one event |
| `batch_prove.py` | Runs proving + verification across multiple events, reports timing/size benchmarks |
| `check_data_alignment.py` | Diagnostic: confirms whether two data files share the same event order (caught a real bug — see Results log) |
| `check_onnx_compatibility.py` | Diagnostic: inspects an exported ONNX file's IR/opset version for EZKL compatibility |

## Dataset

[LHC Olympics 2020 R&D dataset](https://zenodo.org/record/4536377) — a real,
published benchmark for collider anomaly detection (used in CATHODE and
similar published methods). Two files are used:
- `events_anomalydetection_v2.features.h5` (74 MB) — pre-extracted jet mass
  and n-subjettiness features, used for the baseline.
- `events_anomalydetection.h5` (2.8 GB) — raw per-particle four-vectors
  (pT, eta, phi), used for the richer feature set. **Important:** this file
  is NOT row-aligned with the features file above (see Results log) — all
  raw-particle feature extraction here derives everything, including labels,
  from this file alone.

## Setup
```bash
pip install -r requirements.txt
```
Download both files above from the Zenodo link.

## Usage
```bash
# Baseline: 6-feature model + calibration
python calibrate_baseline.py "path\to\events_anomalydetection_v2.features.h5"

# Richer model: raw-particle features + calibration
python calibrate_raw.py "path\to\events_anomalydetection.h5" 100000

# zk-proof pipeline (single event)
python export_onnx.py "path\to\events_anomalydetection_v2.features.h5"
python generate_proof.py

# Batch proving benchmarks
python batch_prove.py "path\to\events_anomalydetection_v2.features.h5" 10
```

## Results log

- **Baseline** (6 hand-picked features from the official features file):
  AUC 0.7789, 10% false-positive calibration guarantee holds exactly.
- **Raw-particle features** (26 features: jet mass/pT/multiplicity + energy
  rings, derived entirely from the raw particle file), validated at 100,000
  events: **AUC 0.7968**, 9.8% empirical FPR vs. 10% target — this is the
  number to cite, not the smaller 20k-event run's 0.8102 (sample noise).
- **zk-proof pipeline**: verified working end-to-end on the trained model.
  Batch benchmark across 10 events: ~1.68s to prove, ~0.03s to verify,
  ~18.6KB proof size, 10/10 verified successfully.
- **Data integrity finding**: discovered the raw particle file and the v2
  features file are NOT row-aligned (correlation ≈ 0 between the two files'
  per-event momenta) — a real, documented issue with this dataset's history.
  Fixed by deriving all raw-particle features self-contained from one file.

## Known limitations (stated honestly, not glossed over)

- Jet-finding for the raw-particle features uses a simplified sequential-cone
  heuristic, not the official anti-kT algorithm (would need a fastjet/pyjet
  dependency this project avoids for portability).
- This is a verification + calibration method, **not a new-physics discovery**
  — the anomaly signal used here is a known, injected benchmark signal, not
  a genuine unseen physics result.
- zk-proof generation (~1.7s/event) is far too slow for real-time use
  (e.g. LHC trigger decisions, which need ~25 nanoseconds) — this pipeline
  is designed for verifying already-published offline analysis results,
  not live triggering.