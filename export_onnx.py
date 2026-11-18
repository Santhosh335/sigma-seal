"""
Train the baseline autoencoder (same as calibrate.py) and export it to ONNX
for the EZKL zk-proof pipeline. Saves the trained weights too, so this step
doesn't need to be redone every time you re-run the proof pipeline.
"""
import sys

import numpy as np
import onnx
import torch
from onnx import version_converter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_loader import build_model_inputs, load_features
from model import JetAutoencoder


def main(h5_path: str, epochs: int = 30):
    print(f"Loading events from {h5_path} ...")
    raw = load_features(h5_path)
    feats = build_model_inputs(raw)
    background = feats[feats["label"] == 0].drop(columns="label").values

    scaler = StandardScaler().fit(background)
    bg_scaled = scaler.transform(background)
    train_x = torch.tensor(bg_scaled, dtype=torch.float32)

    model = JetAutoencoder(input_dim=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(train_x), batch_size=256, shuffle=True)

    print("Training ...")
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

    model.eval()
    torch.save(model.state_dict(), "trained_model.pt")
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)
    print("Saved trained_model.pt, scaler_mean.npy, scaler_scale.npy")

    dummy_input = torch.randn(1, train_x.shape[1], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        "sigma_seal_model.onnx",
        input_names=["input"],
        output_names=["reconstruction"],
        opset_version=13,
        dynamic_axes=None,
        dynamo=False,
    )
    print("Exported sigma_seal_model.onnx (pre-downgrade)")

    exported = onnx.load("sigma_seal_model.onnx")
    exported = version_converter.convert_version(exported, 13)
    exported.ir_version = 8
    onnx.save(exported, "sigma_seal_model.onnx")
    print("Downgraded ONNX file to opset 13 / IR version 8 for EZKL compatibility")

    sample = bg_scaled[0:1].astype(np.float32)
    np.save("sample_input.npy", sample)
    print("Saved sample_input.npy (one real background event, scaled)")


if __name__ == "__main__":
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "events_anomalydetection_v2.features.h5"
    main(h5_path)