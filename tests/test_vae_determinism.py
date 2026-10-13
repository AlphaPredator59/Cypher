"""
Regression tests for deterministic VAE inference in CYPHER IDS.
Verifies that:
1. In evaluation mode (model.eval()), repeated inference on identical inputs produces identical scores and flags.
2. In training mode (model.train()), stochastic sampling via reparameterization remains active.
3. Explicit deterministic flags (deterministic=True/False) correctly override training state.
4. Input/output tensor shapes and pretrained model state_dict compatibility are preserved.
"""

import os
import pytest
import torch
import numpy as np
from ml.models.vae import Model
from ml.pipeline import VAEScorer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pth")


@pytest.fixture
def pretrained_vae():
    model = Model(input_dim=77, hidden_dim=256, latent_dim=16)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def test_vae_deterministic_eval_mode(pretrained_vae):
    """
    Verifies that calling forward() in eval mode produces strictly identical
    reconstructions, latent means, and log-variances on identical inputs.
    """
    pretrained_vae.eval()
    x = torch.randn(10, 77)

    with torch.no_grad():
        recon1, mu1, logvar1 = pretrained_vae(x)
        recon2, mu2, logvar2 = pretrained_vae(x)
        recon3, mu3, logvar3 = pretrained_vae(x)

    # All repeated runs must be mathematically identical
    assert torch.equal(recon1, recon2), "Reconstruction 1 and 2 differ in eval mode!"
    assert torch.equal(recon2, recon3), "Reconstruction 2 and 3 differ in eval mode!"
    assert torch.equal(mu1, mu2), "Latent mean differs between runs!"
    assert torch.equal(logvar1, logvar2), "Logvar differs between runs!"


def test_vae_scorer_zero_variance_on_identical_inputs():
    """
    Verifies that VAEScorer produces exactly 0.0 standard deviation in anomaly scores
    across 20 consecutive passes on the exact same input vector.
    """
    scorer = VAEScorer(input_dim=77, hidden_dim=256, latent_dim=16)
    np.random.seed(1337)
    sample_vector = np.random.randn(1, 77).astype(np.float32)

    scores = []
    flags = []
    top_indices = []

    for _ in range(20):
        result = scorer.score_scaled(sample_vector)
        scores.append(float(result["scores"][0]))
        flags.append(bool(result["flags"][0]))
        top_indices.append(tuple(result["top_idx"][0].tolist()))

    std_dev = np.std(scores)
    assert std_dev == 0.0, f"Expected 0.0 standard deviation in eval mode, got {std_dev}"
    assert len(set(flags)) == 1, "Anomaly flags fluctuated on identical inputs!"
    assert len(set(top_indices)) == 1, "Top-K feature indices fluctuated on identical inputs!"


def test_vae_stochastic_training_mode():
    """
    Verifies that when placed in training mode, the model samples latent noise
    so repeated passes on identical inputs produce distinct outputs.
    """
    model = Model(input_dim=77, hidden_dim=256, latent_dim=16)
    model.train()
    x = torch.randn(5, 77)

    recon1, _, _ = model(x)
    recon2, _, _ = model(x)

    # In train mode, stochastic sampling should produce differing reconstructions
    assert not torch.equal(recon1, recon2), "Train mode produced identical reconstructions (expected stochastic noise)"


def test_vae_explicit_deterministic_override(pretrained_vae):
    """
    Verifies that the deterministic argument explicitly overrides self.training.
    """
    # Force deterministic=True in train mode
    pretrained_vae.train()
    x = torch.randn(5, 77)
    with torch.no_grad():
        recon_det1, _, _ = pretrained_vae(x, deterministic=True)
        recon_det2, _, _ = pretrained_vae(x, deterministic=True)
    assert torch.equal(recon_det1, recon_det2), "deterministic=True should produce identical outputs in train mode"

    # Force deterministic=False in eval mode
    pretrained_vae.eval()
    with torch.no_grad():
        recon_stoch1, _, _ = pretrained_vae(x, deterministic=False)
        recon_stoch2, _, _ = pretrained_vae(x, deterministic=False)
    assert not torch.equal(recon_stoch1, recon_stoch2), "deterministic=False should produce stochastic outputs in eval mode"


def test_vae_tensor_shapes_and_device(pretrained_vae):
    """
    Verifies that input and output tensor shapes match architectural expectations.
    """
    batch_size = 8
    x = torch.zeros(batch_size, 77)

    with torch.no_grad():
        recon, mu, logvar = pretrained_vae(x)

    assert recon.shape == (batch_size, 77), f"Expected recon shape (8, 77), got {recon.shape}"
    assert mu.shape == (batch_size, 16), f"Expected mu shape (8, 16), got {mu.shape}"
    assert logvar.shape == (batch_size, 16), f"Expected logvar shape (8, 16), got {logvar.shape}"
