"""
Days 3-8 (corrected): train + calibrate using features derived ENTIRELY
from the raw particle file (self-contained, avoids the file-alignment
bug found between events_anomalydetection.h5 and the v2 features file).

Starts with a SMALL n_events (jet-finding loops per-event, so this is
slow) -- confirm correctness here before scaling up.
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import JetAutoencoder
from raw_features_v2 import build_selfcontained_features


def conformal_threshold(calib_scores: np.ndarray, alpha: float) -> float:
    n = len(calib_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(calib_scores, q_level, method="higher")


def main(h5_path: str, n_events: int = 20000, alpha: float = 0.10, epochs: int = 30):
    feats = build_selfcontained_features(h5_path, n_events=n_events)

    background = feats[feats["label"] == 0].drop(columns="label").values
    signal = feats[feats["label"] == 1].drop(columns="label").values
    print(f"\nBackground events: {len(background)}, Signal events: {len(signal)}")

    if len(signal) == 0:
        print("WARNING: zero signal events in this sample -- try a larger n_events.")
        return

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

    print("\nTraining ...")
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

    print(f"\n=== Results (self-contained {train_x.shape[1]}-feature raw-particle representation) ===")
    print(f"Target max false-positive rate: {alpha:.3f}")
    print(f"Empirical false-positive rate: {empirical_fpr:.3f}")
    print(f"Calibration guarantee holds: {empirical_fpr <= alpha + 0.02}")
    print(f"Signal events flagged as anomalous: {detection_rate:.3f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\n(Compare to original 6-feature baseline: AUC 0.7789)")


if __name__ == "__main__":
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "events_anomalydetection.h5"
    n_events = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    main(h5_path, n_events=n_events)