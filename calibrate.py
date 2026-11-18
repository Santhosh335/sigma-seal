"""
Month 2: wrap the autoencoder's raw anomaly score in a calibrated confidence
using split conformal prediction.

The problem with train.py's raw reconstruction error: "0.4391" has no
inherent meaning -- you don't know if that's "mildly weird" or "extremely
weird" without external context. Conformal prediction fixes this by giving
a threshold with a GUARANTEED, verifiable false-positive rate on background:
e.g. "any event scoring above this threshold is flagged, and we guarantee
at most 10% of genuine background events will be falsely flagged this way."

This only works if we're honest about the data split:
  - train split: used to train the autoencoder (background only)
  - calibration split: UNSEEN by training, used only to pick the threshold
  - test split: held out from both, used to check the guarantee actually holds
Using the same data for more than one of these roles is the single most
common way conformal prediction claims end up being wrong in practice.
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_loader import build_model_inputs, load_features
from model import JetAutoencoder


def conformal_threshold(calib_scores: np.ndarray, alpha: float) -> float:
    """Standard split-conformal threshold: the (1-alpha) quantile of
    calibration scores, with the finite-sample correction that makes the
    false-positive-rate guarantee exact rather than approximate."""
    n = len(calib_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(calib_scores, q_level, method="higher")


def main(h5_path: str, alpha: float = 0.10, epochs: int = 30):
    print(f"Loading events from {h5_path} ...")
    raw = load_features(h5_path)
    feats = build_model_inputs(raw)

    background = feats[feats["label"] == 0].drop(columns="label").values
    signal = feats[feats["label"] == 1].drop(columns="label").values

    # Three-way split of background: train / calibration / test
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(background))
    n_train = int(0.6 * len(background))
    n_calib = int(0.2 * len(background))
    train_idx = idx[:n_train]
    calib_idx = idx[n_train:n_train + n_calib]
    test_idx = idx[n_train + n_calib:]

    scaler = StandardScaler().fit(background[train_idx])
    train_x = torch.tensor(scaler.transform(background[train_idx]), dtype=torch.float32)
    calib_x = torch.tensor(scaler.transform(background[calib_idx]), dtype=torch.float32)
    test_bg_x = torch.tensor(scaler.transform(background[test_idx]), dtype=torch.float32)
    sig_x = torch.tensor(scaler.transform(signal), dtype=torch.float32)

    model = JetAutoencoder(input_dim=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(train_x), batch_size=256, shuffle=True)

    print("Training on background-only training split ...")
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(train_x):.5f}")

    calib_scores = model.anomaly_score(calib_x).numpy()
    threshold = conformal_threshold(calib_scores, alpha)
    print(f"\nConformal threshold at alpha={alpha}: {threshold:.5f}")
    print("(any event scoring above this is flagged as anomalous)")

    # Check the guarantee: on a FRESH background test split never used for
    # training OR calibration, the false-positive rate should be <= alpha
    test_bg_scores = model.anomaly_score(test_bg_x).numpy()
    empirical_fpr = (test_bg_scores > threshold).mean()
    print(f"\nTarget max false-positive rate: {alpha:.3f}")
    print(f"Empirical false-positive rate on held-out background: {empirical_fpr:.3f}")
    if empirical_fpr <= alpha + 0.02:  # small tolerance for finite-sample noise
        print("-> Calibration guarantee HOLDS. Good.")
    else:
        print("-> WARNING: guarantee violated. Check for data leakage between splits.")

    # How many signal events get flagged, now with a calibrated meaning
    sig_scores = model.anomaly_score(sig_x).numpy()
    detection_rate = (sig_scores > threshold).mean()
    print(f"\nSignal events flagged as anomalous: {detection_rate:.3f}")

    auc = roc_auc_score(
        np.concatenate([np.zeros(len(test_bg_scores)), np.ones(len(sig_scores))]),
        np.concatenate([test_bg_scores, sig_scores]),
    )
    print(f"ROC-AUC (unchanged metric, for comparison to month-1 baseline): {auc:.4f}")


if __name__ == "__main__":
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "events_anomalydetection_v2.features.h5"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    main(h5_path, alpha=alpha)
