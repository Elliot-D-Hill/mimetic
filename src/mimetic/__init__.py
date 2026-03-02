from typing import Literal

import torch
from tensordict import TensorDict
from torch import Tensor

from .covariance import make_covariance
from .pipeline import add_latent_features, add_observed_features
from .tasks import (
    competing_risk_data,
    linear_data,
    logistic_data,
    mixture_cure_data,
    multi_event_data,
    multiclass_data,
    survival_data,
)


def make_parameters(
    parameters: int | list[float], scale: float, n_targets: int = 1
) -> Tensor:
    match parameters:
        case int():
            mean = torch.randn(n_targets, parameters)
        case list():
            if n_targets > 1:
                raise ValueError(
                    "Explicit parameter lists not supported with n_targets > 1. Use int."
                )
            mean = torch.tensor(parameters).unsqueeze(0)
    return mean * scale


def expand_constants(td: TensorDict, num_timepoints: int) -> TensorDict:
    """Broadcast tensors with singleton time dimension to num_timepoints."""
    for key, value in td.items():
        if value.dim() < 2 or value.size(1) != 1:
            continue
        expanded_shape = (value.size(0), num_timepoints, *value.shape[2:])
        td[key] = value.expand(*expanded_shape)
    return td


Tasks = Literal[
    "linear",
    "logistic",
    "multiclass",
    "survival",
    "mixture_cure",
    "competing_risk",
    "multi_event",
]


def simulate(
    task: Tasks,
    num_samples: int,
    num_timepoints: int,
    parameters: int | list[float],
    scale: float,
    latent_std: float,
    observed_std: float,
    # Task-specific (optional with defaults)
    n_targets: int = 1,
    prevalence: float = 0.5,
    gamma_shape: float = 1.0,
    gamma_rate: float = 1.0,
    vocab_size: int = 1000,
    concentration: float = 1.0,  # Dirichlet concentration for token sampling
    tte_boundaries: list[float] | None = None,
    # Covariance options
    covariance_type: Literal["isotropic", "ar1", "lkj"] = "isotropic",
    rho: float = 0.9,  # AR(1) autocorrelation
    eta: float = 1.0,  # LKJ concentration
) -> TensorDict:
    """Main simulation entry point with centralized component construction."""
    if n_targets > 1 and task in ("survival", "mixture_cure", "competing_risk", "multi_event"):
        raise ValueError(f"n_targets > 1 is not supported for task '{task}'")
    if task == "multiclass" and n_targets < 2:
        raise ValueError("multiclass task requires n_targets >= 2")
    weights = make_parameters(parameters, scale, n_targets=n_targets)
    covariance = make_covariance(
        num_timepoints=num_timepoints, covariance_type=covariance_type, rho=rho, eta=eta
    )
    data = TensorDict(
        {"id": torch.arange(num_samples).view(-1, 1, 1)},
        batch_size=num_samples,
    )
    data = add_latent_features(data, hidden_dim=weights.size(-1), latent_std=latent_std)
    data = add_observed_features(
        data,
        num_timepoints=num_timepoints,
        observed_std=observed_std,
        covariance=covariance,
    )
    # TODO should discretization happen after all data generation?
    boundaries = None
    if tte_boundaries is None:
        boundaries = torch.linspace(0, num_timepoints - 1, num_timepoints)
    else:
        boundaries = torch.tensor(
            tte_boundaries,
            device=data["features"].device,
            dtype=data["features"].dtype,
        )
    match task:
        case "linear":
            data = linear_data(data, weights, prevalence, vocab_size, concentration)
        case "logistic":
            data = logistic_data(data, weights, prevalence, vocab_size, concentration)
        case "multiclass":
            data = multiclass_data(data, weights, prevalence, vocab_size, concentration)
        case "survival":
            data = survival_data(
                data,
                weights,
                prevalence,
                gamma_shape,
                gamma_rate,
                vocab_size,
                concentration,
            )
        case "mixture_cure":
            data = mixture_cure_data(
                data,
                weights,
                prevalence,
                gamma_shape,
                gamma_rate,
                vocab_size,
                concentration,
            )
        case "competing_risk":
            data = competing_risk_data(data, vocab_size, boundaries)
        case "multi_event":
            data = multi_event_data(data, vocab_size, boundaries)
        case _:
            raise ValueError(
                f"Unknown task: {task}. Choose from "
                "'linear', 'logistic', 'multiclass', 'survival', 'mixture_cure', "
                "'competing_risk', or 'multi_event'."
            )
    if "label" in data.keys():
        if task == "multiclass":
            labels = data["label"].flatten()
            for c in range(n_targets):
                print(f"Class {c} proportion: {(labels == c).float().mean().item():.3f}")
        else:
            print("Label prevalence:", f"{data['label'].mean().item():.3f}")
    if "indicator" in data.keys():
        print("Indicator prevalence:", f"{data['indicator'].mean().item():.3f}")
    data = expand_constants(td=data, num_timepoints=num_timepoints)
    return data
