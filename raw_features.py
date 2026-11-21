"""
Self-contained feature extraction from the RAW particle file ONLY --
no dependency on the (misaligned) features file. Uses a simplified
sequential-cone jet-finding heuristic, NOT the official anti-kT
algorithm. This is an honest approximation, worth stating plainly in
any writeup: real jet clustering is more sophisticated, but this keeps
everything self-consistent (same file, same label column) and avoids
the alignment bug we found between events_anomalydetection.h5 and
events_anomalydetection_v2.features.h5.
"""
import numpy as np
import pandas as pd


def load_raw_events(h5_path: str, n_events=None):
    df = pd.read_hdf(h5_path)
    if n_events:
        df = df.iloc[:n_events]
    arr = df.values
    labels = arr[:, -1]
    n_particles = (arr.shape[1] - 1) // 3
    particles = arr[:, :-1].reshape(len(arr), n_particles, 3).astype(np.float32)
    return particles, labels


def find_two_jets(raw_particles, dr_cone=1.0):
    n_events, n_particles, _ = raw_particles.shape
    pt = raw_particles[:, :, 0]
    eta = raw_particles[:, :, 1]
    phi = raw_particles[:, :, 2]

    jet1_mask = np.zeros((n_events, n_particles), dtype=bool)
    jet2_mask = np.zeros((n_events, n_particles), dtype=bool)

    for e in range(n_events):
        valid = pt[e] > 0
        if not valid.any():
            continue
        seed1 = np.argmax(np.where(valid, pt[e], -1))
        deta = eta[e] - eta[e, seed1]
        dphi = np.mod(phi[e] - phi[e, seed1] + np.pi, 2 * np.pi) - np.pi
        dr = np.sqrt(deta**2 + dphi**2)
        in_jet1 = valid & (dr < dr_cone)
        jet1_mask[e] = in_jet1

        remaining = valid & ~in_jet1
        if not remaining.any():
            continue
        seed2 = np.argmax(np.where(remaining, pt[e], -1))
        deta2 = eta[e] - eta[e, seed2]
        dphi2 = np.mod(phi[e] - phi[e, seed2] + np.pi, 2 * np.pi) - np.pi
        dr2 = np.sqrt(deta2**2 + dphi2**2)
        in_jet2 = remaining & (dr2 < dr_cone)
        jet2_mask[e] = in_jet2

        if e % 5000 == 0 and e > 0:
            print(f"  jet-finding progress: {e}/{n_events} events")

    return jet1_mask, jet2_mask


def jet_kinematics(raw_particles, jet_mask):
    pt = raw_particles[:, :, 0]
    eta = raw_particles[:, :, 1]
    phi = raw_particles[:, :, 2]

    px = pt * np.cos(phi) * jet_mask
    py = pt * np.sin(phi) * jet_mask
    pz = pt * np.sinh(eta) * jet_mask
    E = pt * np.cosh(eta) * jet_mask

    sum_px, sum_py, sum_pz, sum_E = px.sum(1), py.sum(1), pz.sum(1), E.sum(1)
    mass_sq = np.clip(sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2, 0, None)
    mass = np.sqrt(mass_sq)
    jet_pt = np.sqrt(sum_px**2 + sum_py**2)
    multiplicity = jet_mask.sum(axis=1).astype(np.float32)
    return mass, jet_pt, multiplicity


def jet_axis(raw_particles, jet_mask):
    pt = raw_particles[:, :, 0]
    eta = raw_particles[:, :, 1]
    phi = raw_particles[:, :, 2]
    masked_pt = pt * jet_mask
    total = masked_pt.sum(axis=1)
    total = np.where(total == 0, 1, total)
    axis_eta = (masked_pt * eta).sum(axis=1) / total
    axis_phi = (masked_pt * phi).sum(axis=1) / total
    return axis_eta, axis_phi


def energy_rings(raw_particles, jet_mask, axis_eta, axis_phi, n_rings=10, max_dr=1.0):
    pt = raw_particles[:, :, 0]
    eta = raw_particles[:, :, 1]
    phi = raw_particles[:, :, 2]

    deta = eta - axis_eta[:, None]
    dphi = np.mod(phi - axis_phi[:, None] + np.pi, 2 * np.pi) - np.pi
    dr = np.sqrt(deta**2 + dphi**2)

    total_pt = (pt * jet_mask).sum(axis=1, keepdims=True)
    total_pt = np.where(total_pt == 0, 1, total_pt)

    ring_edges = np.linspace(0, max_dr, n_rings + 1)
    rings = np.zeros((raw_particles.shape[0], n_rings))
    for i in range(n_rings):
        in_ring = jet_mask & (dr >= ring_edges[i]) & (dr < ring_edges[i + 1])
        rings[:, i] = (pt * in_ring).sum(axis=1) / total_pt[:, 0]
    return rings


def build_selfcontained_features(h5_path: str, n_events=None) -> pd.DataFrame:
    print(f"Loading raw events ...")
    raw, labels = load_raw_events(h5_path, n_events=n_events)
    print(f"Loaded {len(raw)} events. Finding jets (this loops per-event, be patient) ...")
    jet1_mask, jet2_mask = find_two_jets(raw)

    print("Computing jet kinematics ...")
    m1, pt1, mult1 = jet_kinematics(raw, jet1_mask)
    m2, pt2, mult2 = jet_kinematics(raw, jet2_mask)

    print("Computing energy rings ...")
    axis1_eta, axis1_phi = jet_axis(raw, jet1_mask)
    axis2_eta, axis2_phi = jet_axis(raw, jet2_mask)
    rings1 = energy_rings(raw, jet1_mask, axis1_eta, axis1_phi)
    rings2 = energy_rings(raw, jet2_mask, axis2_eta, axis2_phi)

    out = pd.DataFrame({
        "mj1": m1, "ptj1": pt1, "multj1": mult1,
        "mj2": m2, "ptj2": pt2, "multj2": mult2,
    })
    for i in range(10):
        out[f"ring_j1_{i}"] = rings1[:, i]
        out[f"ring_j2_{i}"] = rings2[:, i]
    out["label"] = labels
    return out