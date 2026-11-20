"""
Day 3-6: extract richer features from the RAW particle-level file
(events_anomalydetection.h5, 2.8GB), combined with the high-level features
file (mj1, mj2, tau ratios) we already use.
"""
import numpy as np
import pandas as pd

from data_loader import COLUMNS, load_features


def load_raw_particles(h5_path: str, n_events=None) -> np.ndarray:
    """The raw file is ALSO stored as a pandas DataFrame via pd.to_hdf,
    same as the features file -- use pd.read_hdf, not h5py directly."""
    try:
        df = pd.read_hdf(h5_path, start=0, stop=n_events) if n_events else pd.read_hdf(h5_path)
    except (TypeError, NotImplementedError):
        # fixed-format HDF5 doesn't support start/stop -- load fully, then slice
        df = pd.read_hdf(h5_path)
        if n_events:
            df = df.iloc[:n_events]
    arr = df.values
    n_cols = arr.shape[1]
    n_particles = (n_cols - 1) // 3
    particles = arr[:, :-1].reshape(len(arr), n_particles, 3).astype(np.float32)
    return particles


def compute_jet_axes(features_df: pd.DataFrame):
    def eta_phi(px, py, pz):
        p = np.sqrt(px**2 + py**2 + pz**2)
        p = np.where(p == 0, 1e-8, p)
        eta = np.arctanh(np.clip(pz / p, -0.999999, 0.999999))
        phi = np.arctan2(py, px)
        return eta, phi

    eta1, phi1 = eta_phi(features_df["pxj1"].values, features_df["pyj1"].values, features_df["pzj1"].values)
    eta2, phi2 = eta_phi(features_df["pxj2"].values, features_df["pyj2"].values, features_df["pzj2"].values)
    return eta1, phi1, eta2, phi2


def _dr_to(eta, phi, jet_eta, jet_phi):
    deta = eta - jet_eta[:, None]
    dphi = phi - jet_phi[:, None]
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    return np.sqrt(deta**2 + dphi**2)


def compute_energy_rings(raw_particles, jet_eta, jet_phi, other_eta, other_phi,
                          n_rings=10, max_dr=1.0):
    pt = raw_particles[:, :, 0]
    eta = raw_particles[:, :, 1]
    phi = raw_particles[:, :, 2]
    valid = pt > 0

    dr_this = _dr_to(eta, phi, jet_eta, jet_phi)
    dr_other = _dr_to(eta, phi, other_eta, other_phi)
    belongs_here = valid & (dr_this <= dr_other)

    total_pt = (pt * belongs_here).sum(axis=1, keepdims=True)
    total_pt = np.where(total_pt == 0, 1, total_pt)

    ring_edges = np.linspace(0, max_dr, n_rings + 1)
    rings = np.zeros((raw_particles.shape[0], n_rings))
    for i in range(n_rings):
        in_ring = belongs_here & (dr_this >= ring_edges[i]) & (dr_this < ring_edges[i + 1])
        rings[:, i] = (pt * in_ring).sum(axis=1) / total_pt[:, 0]
    return rings


def build_combined_features(raw_h5_path: str, features_h5_path: str, n_events=None) -> pd.DataFrame:
    features_df = load_features(features_h5_path)
    if n_events:
        features_df = features_df.iloc[:n_events]

    print(f"Loading raw particles for {len(features_df)} events ...")
    raw_particles = load_raw_particles(raw_h5_path, n_events=len(features_df))

    eta1, phi1, eta2, phi2 = compute_jet_axes(features_df)

    print("Computing energy rings for jet 1 ...")
    rings1 = compute_energy_rings(raw_particles, eta1, phi1, eta2, phi2)
    print("Computing energy rings for jet 2 ...")
    rings2 = compute_energy_rings(raw_particles, eta2, phi2, eta1, phi1)

    eps = 1e-8
    out = pd.DataFrame({
        "mj1": features_df["mj1"].values,
        "mj2": features_df["mj2"].values,
        "tau21j1": (features_df["tau2j1"] / (features_df["tau1j1"] + eps)).values,
        "tau21j2": (features_df["tau2j2"] / (features_df["tau1j2"] + eps)).values,
        "tau32j1": (features_df["tau3j1"] / (features_df["tau2j1"] + eps)).values,
        "tau32j2": (features_df["tau3j2"] / (features_df["tau2j2"] + eps)).values,
    })
    for i in range(10):
        out[f"ring_j1_{i}"] = rings1[:, i]
        out[f"ring_j2_{i}"] = rings2[:, i]
    out["label"] = features_df["label"].values
    return out