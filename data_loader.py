"""
Load the LHC Olympics R&D HIGH-LEVEL FEATURES file (events_anomalydetection_v2.features.h5).

Confirmed columns (from the official Zenodo record), in order:
  pxj1, pyj1, pzj1, mj1, tau1j1, tau2j1, tau3j1,
  pxj2, pyj2, pzj2, mj2, tau1j2, tau2j2, tau3j2,
  label   (1 = signal, 0 = background; included for evaluation only)

We don't need the raw 700-particle four-vectors at all for the baseline --
this file is already the standard 6-feature representation (jet masses +
n-subjettiness ratios) used in most published LHCO baselines, just with
the raw tau1/tau2/tau3 instead of pre-divided ratios. We compute the
ratios ourselves below.
"""
import pandas as pd

COLUMNS = [
    "pxj1", "pyj1", "pzj1", "mj1", "tau1j1", "tau2j1", "tau3j1",
    "pxj2", "pyj2", "pzj2", "mj2", "tau1j2", "tau2j2", "tau3j2",
    "label",
]


def load_features(h5_path: str) -> pd.DataFrame:
    """Load the features file. It's stored as a pandas DataFrame via
    pd.to_hdf, so pd.read_hdf works directly -- no manual h5py parsing
    needed, unlike the raw particle file."""
    df = pd.read_hdf(h5_path)
    df.columns = COLUMNS
    return df


def build_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the canonical 6 features used in published LHCO baselines:
    mj1, mj2, tau21j1, tau21j2, tau32j1, tau32j2.
    tau21 = tau2/tau1 (how 1-prong-like vs 2-prong-like a jet is)
    tau32 = tau3/tau2 (how 2-prong-like vs 3-prong-like a jet is)
    Small epsilon avoids division by zero on degenerate jets.
    """
    eps = 1e-8
    out = pd.DataFrame({
        "mj1": df["mj1"],
        "mj2": df["mj2"],
        "tau21j1": df["tau2j1"] / (df["tau1j1"] + eps),
        "tau21j2": df["tau2j2"] / (df["tau1j2"] + eps),
        "tau32j1": df["tau3j1"] / (df["tau2j1"] + eps),
        "tau32j2": df["tau3j2"] / (df["tau2j2"] + eps),
        "label": df["label"],
    })
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "events_anomalydetection_v2.features.h5"
    raw = load_features(path)
    feats = build_model_inputs(raw)
    print(feats.describe())
    print("Signal fraction:", feats["label"].mean())
