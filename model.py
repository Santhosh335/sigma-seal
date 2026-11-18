"""
A deliberately simple dense autoencoder. Trained ONLY on background-labeled
events; anomaly score = reconstruction error. If background is reconstructed
well and signal poorly, the error itself is the anomaly signal — no labels
needed at inference time.
"""
import torch
import torch.nn as nn


class JetAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 4, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def anomaly_score(self, x):
        """Per-event reconstruction error (MSE), used as the anomaly score."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=1)
