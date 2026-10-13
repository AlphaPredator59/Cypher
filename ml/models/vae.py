"""
Variational Autoencoder (VAE) Architecture for CYPHER Anomaly Detection.
Trained on normalized benign network flows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)   # No sigmoid for StandardScaler data
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        logvar = torch.clamp(logvar, -10.0, 10.0)  # Stabilizer
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor, deterministic: bool = None):
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            deterministic: If True, uses the latent mean `mu` directly (no noise sampling).
                           If False, samples `z` via reparameterization trick.
                           If None (default), automatically selects deterministic inference
                           when in evaluation mode (`not self.training`), and stochastic sampling
                           during training mode (`self.training`).
        """
        mu, logvar = self.encode(x)
        use_deterministic = (not self.training) if deterministic is None else deterministic
        if use_deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class TotalLoss(nn.Module):
    def forward(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        # Sample-wise SSE then batch mean
        recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=1).mean()

        # Clamp logvar before exp to prevent overflow
        logvar = torch.clamp(logvar, -10.0, 10.0)

        # KL divergence per sample then batch mean
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        total = recon_loss + kld
        return total, recon_loss.detach(), kld.detach()
