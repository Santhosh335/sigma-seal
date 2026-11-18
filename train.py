"""
Train the baseline autoencoder on background-only events, then evaluate
whether its reconstruction-error anomaly score separates held-out signal
from background. This is the sanity check the whole project depends on:
if this doesn't show real separation, nothing downstream (calibration,
zk-proofs) is worth building yet.
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_loader import build_model_inputs, load_features
from model import JetAutoencoder


def main(h5_path: str, n_events: int | None = None, epochs: int = 30):
    print(f"Loading events from {h5_path} ...")
    raw = load_features(h5_path)
    if n_events:
        raw = raw.iloc[:n_events]
    feats = build_model_inputs(raw)

    background = feats[feats["label"] == 0].drop(columns="label").values
    signal = feats[feats["label"] == 1].drop(columns="label").values

    scaler = StandardScaler().fit(background)
    bg_scaled = scaler.transform(background)
    sig_scaled = scaler.transform(signal)

    # Train/val split on background only — the model never sees signal
    n_train = int(0.8 * len(bg_scaled))
    train_x = torch.tensor(bg_scaled[:n_train], dtype=torch.float32)
    val_bg_x = torch.tensor(bg_scaled[n_train:], dtype=torch.float32)
    sig_x = torch.tensor(sig_scaled, dtype=torch.float32)

    model = JetAutoencoder(input_dim=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(train_x), batch_size=256, shuffle=True)

    print("Training on background-only events ...")
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

    # Evaluate: does reconstruction error separate held-out background from signal?
    bg_scores = model.anomaly_score(val_bg_x).numpy()
    sig_scores = model.anomaly_score(sig_x).numpy()

    y_true = np.concatenate([np.zeros(len(bg_scores)), np.ones(len(sig_scores))])
    y_score = np.concatenate([bg_scores, sig_scores])
    auc = roc_auc_score(y_true, y_score)

    print(f"\nHeld-out background mean score: {bg_scores.mean():.4f}")
    print(f"Signal mean score:              {sig_scores.mean():.4f}")
    print(f"ROC-AUC (signal vs background):  {auc:.4f}")
    print("(0.5 = no separation at all, 1.0 = perfect separation)")

    return model, auc


if __name__ == "__main__":
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "events_anomalydetection_v2.h5"
    main(h5_path)
