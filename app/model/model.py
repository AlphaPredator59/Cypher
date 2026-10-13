"""
Backward-compatible adapter for VAE Model and TotalLoss.
Canonical implementation is consolidated in ml.models.vae.
"""
from ml.models.vae import Model, TotalLoss

__all__ = ["Model", "TotalLoss"]
