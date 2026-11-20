"""
Days 3-8: train + calibrate on the RICHER feature set (mass + tau ratios +
20 energy-ring features, 26 total) instead of the original 6-feature
baseline. Same model architecture, same conformal calibration method as
calibrate.py -- only the input representation changes, so this is a fair,
direct comparison against the original AUC 0.7789 result.

Loading the full 1.1M-event raw particle file can be slow/memory-heavy on
a laptop -- defaults to a subset. Increase n_events once this runs
successfully on the smaller subset.
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import JetAutoencoder
from raw_features import build_combined_features


def conformal_threshold(calib_scores: np.ndarray, alpha: float) -> float:
    n = len(calib_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(calib_scores, q_level, method="higher")


def main(raw_h5_path: str, features_h5_path: str, n_events: int = 100000,
         alpha: float = 0.10, epochs: int = 30):
    feats = build_combined_features(raw_h5_path, features_h5_path, n_events=n_events)

    background = feats[feats["label"] == 0].drop(columns="label").values
    signal = feats[feats["label"] == 1].drop(columns="label").values
    print(f"\nBackground events: {len(background)}, Signal events: {len(signal)}")
    print(f"Feature vector length: {background.shape[1]} (vs. 6 in the original baseline)")

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

    print("\nTraining on background-only training split ...")
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
    test_bg_scores = model.anomaly_score(test_bg_x).numpy()
    empirical_fpr = (test_bg_scores > threshold).mean()
    sig_scores = model.anomaly_score(sig_x).numpy()
    detection_rate = (sig_scores > threshold).mean()

    auc = roc_auc_score(
        np.concatenate([np.zeros(len(test_bg_scores)), np.ones(len(sig_scores))]),
        np.concatenate([test_bg_scores, sig_scores]),
    )

    print(f"\n=== Results with richer ({train_x.shape[1]}-feature) representation ===")
    print(f"Conformal threshold at alpha={alpha}: {threshold:.5f}")
    print(f"Target max false-positive rate: {alpha:.3f}")
    print(f"Empirical false-positive rate: {empirical_fpr:.3f}")
    print(f"Calibration guarantee holds: {empirical_fpr <= alpha + 0.02}")
    print(f"Signal events flagged as anomalous: {detection_rate:.3f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\n(Compare to original 6-feature baseline: AUC 0.7789, detection rate 0.440)")

    torch.save(model.state_dict(), "trained_model_v2.pt")
    np.save("scaler_mean_v2.npy", scaler.mean_)
    np.save("scaler_scale_v2.npy", scaler.scale_)
    print("\nSaved trained_model_v2.pt, scaler_mean_v2.npy, scaler_scale_v2.npy")


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "events_anomalydetection.h5"
    features_path = sys.argv[2] if len(sys.argv) > 2 else "events_anomalydetection_v2.features.h5"
    n_events = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    main(raw_path, features_path, n_events=n_events)