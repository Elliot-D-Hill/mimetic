from mimetic.covariance import AR1Covariance, residual_covariance
from mimetic.functional import logistic_output, observations, random_effects, tokens


def test_task_pipeline_smoke() -> None:
    num_samples = 16
    num_timepoints = 6
    num_features = 4

    covariance = residual_covariance(num_timepoints, AR1Covariance(correlation=0.8))
    data = observations(
        num_samples,
        num_timepoints=num_timepoints,
        num_features=num_features,
        std=0.25,
        covariance=covariance,
    )
    data = random_effects(data, stds=[1.0, 0.05])
    data = tokens(data, vocab_size=32)
    data = logistic_output(data, prevalence=0.3)

    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["probability"].shape == (num_samples, num_timepoints, 1)
    assert data["label"].shape == (num_samples, num_timepoints, 1)
    tok = data.get("tokens")
    assert tok is not None
    assert tok.shape == (num_samples, num_timepoints, 1)
