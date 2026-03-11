import torch
from tensordict import TensorDict

from mimetic.covariance import AR1Covariance, make_residual_covariance
from mimetic.functional import (
    linear_predictor,
    logistic_output,
    observations,
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
    data = random_effects(data, hidden_dim=hidden_dim, stds=[1.0, 0.05])
    covariance = make_residual_covariance(
        num_timepoints, AR1Covariance(correlation=0.8)
    )
    data = observations(
        data, num_timepoints=num_timepoints, observed_std=0.25, covariance=covariance
    )
    data = linear_predictor(data, weight=torch.randn(1, hidden_dim), prevalence=0.3)
    data = logistic_output(data)
    data = tokens(data, vocab_size=32)

    assert data["y"].shape == (num_samples, num_timepoints, hidden_dim)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)
    assert data["probability"].shape == (num_samples, 1, 1)
    assert data["label"].shape == (num_samples, 1, 1)
