"""
Check whether the raw particle file and the features file are actually
row-aligned (same event order), and find where signal events live in
the raw file's row ordering.
"""
import sys
import pandas as pd

raw_path = sys.argv[1]
features_path = sys.argv[2]

print("Loading raw file's own label column (last column)...")
raw_df = pd.read_hdf(raw_path)
raw_labels = raw_df.iloc[:, -1].values
print(f"Raw file: {len(raw_labels)} rows")

print("\nLoading features file's label column...")
feat_df = pd.read_hdf(features_path)
feat_labels = feat_df.iloc[:, -1].values
print(f"Features file: {len(feat_labels)} rows")

print(f"\nSame length? {len(raw_labels) == len(feat_labels)}")

n = min(len(raw_labels), len(feat_labels))
match = (raw_labels[:n] == feat_labels[:n]).mean()
print(f"Label match rate (first {n} rows, raw vs features): {match:.4f}")
print("(1.0 = perfectly aligned, ~0.9 or random = NOT aligned/shuffled differently)")

print(f"\nRaw file signal fraction, by chunk of the file:")
chunk = len(raw_labels) // 10
for i in range(10):
    start, end = i * chunk, (i + 1) * chunk
    frac = raw_labels[start:end].mean()
    print(f"  rows {start}-{end}: signal fraction = {frac:.4f}")