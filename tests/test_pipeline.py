import torch
from tensordict import TensorDict

from mimetic.covariance import make_residual_covariance
from mimetic.functional import (
    linear_output,
    logistic_output,
    observed_features,
    random_effects,
    tokens,
)


def test_task_pipeline_smoke() -> None:
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4

    data = TensorDict(
        {
            "id": torch.arange(num_samples).view(-1, 1, 1),
            "group": torch.zeros(num_samples, 1, 1, dtype=torch.long),
        },
        batch_size=[num_samples],
    )
    data = random_effects(data, hidden_dim=hidden_dim, latent_std=1.0, slope_std=0.05)
    covariance = make_residual_covariance(
        num_timepoints=num_timepoints, covariance_type="ar1", correlation=0.8
    )
    data = observed_features(
        data, num_timepoints=num_timepoints, observed_std=0.25, covariance=covariance
    )
    data = linear_output(data, weight=torch.randn(1, hidden_dim), prevalence=0.3)
    data = logistic_output(data)
    data = tokens(data, vocab_size=32)

    assert data["features"].shape == (num_samples, num_timepoints, hidden_dim)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)
    assert data["probability"].shape == (num_samples, 1, 1)
    assert data["label"].shape == (num_samples, 1, 1)
