from typing import Literal

import torch
from tensordict import TensorDict

from .covariance import make_covariance
from .pipeline import add_observed_features, add_random_effects
from .tasks import (
    competing_risk_data,
    linear_data,
    logistic_data,
    mixture_cure_data,
    multi_event_data,
    multiclass_data,
    ordinal_data,
    survival_data,
)

Tasks = Literal[
    "linear",
    "logistic",
    "multiclass",
    "ordinal",
    "survival",
    "mixture_cure",
    "competing_risk",
    "multi_event",
]


def simulate(
    task: Tasks,
    num_samples: int,
    num_timepoints: int,
    num_parameters: int,
    scale: float,
    latent_std: float,
    observed_std: float,
    # Task-specific (optional with defaults)
    num_targets: int = 1,
    prevalence: float = 0.5,
    gamma_shape: float = 1.0,
    gamma_rate: float = 1.0,
    vocab_size: int = 1000,
    concentration: float = 1.0,  # Dirichlet concentration for token sampling
    tte_boundaries: list[float] | None = None,
    # Temporal covariance options
    covariance_type: Literal["isotropic", "ar1", "lkj"] = "isotropic",
    rho: float = 0.9,  # AR(1) autocorrelation
    eta: float = 1.0,  # LKJ concentration
    # Multilevel / hierarchy options
    num_groups: int | None = None,
    icc: float = 0.0,
    group_sizes: list[int] | None = None,
    slope_std: float = 0.0,
    random_effects_eta: float = torch.inf,
) -> TensorDict:
    """Main simulation entry point with centralized component construction."""
    if num_targets > 1 and task in (
        "survival",
        "mixture_cure",
        "competing_risk",
        "multi_event",
    ):
        raise ValueError(f"num_targets > 1 is not supported for task '{task}'")
    if task in ("multiclass", "ordinal") and num_targets < 2:
        raise ValueError(f"{task} task requires num_targets >= 2")
    if task == "ordinal":
        weights = torch.randn(1, num_parameters) * scale
    else:
        weights = torch.randn(num_targets, num_parameters) * scale
    covariance = make_covariance(
        num_timepoints=num_timepoints, covariance_type=covariance_type, rho=rho, eta=eta
    )
    data = TensorDict(
        {"id": torch.arange(num_samples).view(-1, 1, 1)}, batch_size=num_samples
    )
    # Group assignment (default: single group with all subjects)
    if group_sizes is not None:
        sizes = torch.tensor(group_sizes, dtype=torch.long)
        if num_groups is not None and len(group_sizes) != num_groups:
            raise ValueError(
                f"len(group_sizes)={len(group_sizes)} != num_groups={num_groups}"
            )
        if sizes.sum().item() != num_samples:
            raise ValueError(
                f"sum(group_sizes)={sizes.sum().item()} != num_samples={num_samples}"
            )
        num_groups = len(group_sizes)
    elif num_groups is not None:
        base_size = num_samples // num_groups
        remainder = num_samples % num_groups
        sizes = torch.full((num_groups,), base_size, dtype=torch.long)
        sizes[:remainder] += 1
    else:
        num_groups = 1
        sizes = torch.tensor([num_samples], dtype=torch.long)
    group_ids = torch.repeat_interleave(torch.arange(num_groups), sizes)
    data["group"] = group_ids.view(-1, 1, 1)  # [N, 1, 1]
    data = add_random_effects(
        data,
        hidden_dim=weights.size(-1),
        latent_std=latent_std,
        icc=icc,
        slope_std=slope_std,
        random_effects_eta=random_effects_eta,
    )
    data = add_observed_features(
        data,
        num_timepoints=num_timepoints,
        observed_std=observed_std,
        covariance=covariance,
    )
    boundaries = None
    if tte_boundaries is None:
        boundaries = torch.linspace(0, num_timepoints - 1, num_timepoints)
    else:
        boundaries = torch.tensor(
            tte_boundaries, device=data["features"].device, dtype=data["features"].dtype
        )
    match task:
        case "linear":
            data = linear_data(data, weights, prevalence, vocab_size, concentration)
        case "logistic":
            data = logistic_data(data, weights, prevalence, vocab_size, concentration)
        case "multiclass":
            data = multiclass_data(data, weights, prevalence, vocab_size, concentration)
        case "ordinal":
            data = ordinal_data(
                data, weights, prevalence, num_targets, vocab_size, concentration
            )
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
                "'linear', 'logistic', 'multiclass', 'ordinal', 'survival', "
                "'mixture_cure', 'competing_risk', or 'multi_event'."
            )
    if "label" in data.keys():
        if task in ("multiclass", "ordinal"):
            labels = data["label"].flatten()
            for c in range(num_targets):
                proportion = (labels == c).float().mean().item()
                print(f"Class {c} proportion: {proportion:.3f}")
        else:
            print("Label prevalence:", f"{data['label'].mean().item():.3f}")
    if "indicator" in data.keys():
        print("Indicator prevalence:", f"{data['indicator'].mean().item():.3f}")
    return data
