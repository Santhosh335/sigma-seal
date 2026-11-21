"""
Stronger alignment check: compare total visible pT per event (computed
from raw particles) against the two jets' combined pT (from the features
file). If the files are truly row-aligned, these should correlate
strongly -- since the trigger selects events where nearly all visible
momentum is in the two leading jets. If misaligned, correlation ~0.
"""
import sys
import numpy as np
import pandas as pd

raw_path = sys.argv[1]
features_path = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 50000

print(f"Loading first {n} rows of each file ...")
raw_df = pd.read_hdf(raw_path)
raw_arr = raw_df.iloc[:n, :-1].values  # drop raw file's own label column
n_particles = raw_arr.shape[1] // 3
raw_arr = raw_arr.reshape(n, n_particles, 3)
total_pt_raw = raw_arr[:, :, 0].sum(axis=1)  # sum of all particle pT per event

feat_df = pd.read_hdf(features_path).iloc[:n]
jet_pt = (np.sqrt(feat_df["pxj1"]**2 + feat_df["pyj1"]**2) +
          np.sqrt(feat_df["pxj2"]**2 + feat_df["pyj2"]**2)).values

correlation = np.corrcoef(total_pt_raw, jet_pt)[0, 1]
print(f"\nCorrelation between raw-file total pT and features-file jet pT: {correlation:.4f}")
print("(close to 1.0 = aligned, close to 0 = NOT aligned/different event order)")

print(f"\nFirst 5 rows, side by side:")
for i in range(5):
    print(f"  row {i}: raw_total_pT={total_pt_raw[i]:.1f}  features_jet_pT={jet_pt[i]:.1f}")