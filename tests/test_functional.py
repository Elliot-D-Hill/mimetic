from mimetic.functional import bernoulli, linear_predictor, random_effects, tokens


def test_task_pipeline_smoke() -> None:
    """Free-function pipeline: linear_predictor -> random_effects -> bernoulli -> tokens.

    Tests: end-to-end functional pipeline produces correct shapes for all keys.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4

    state = linear_predictor(num_samples, num_timepoints, num_features)
    state = random_effects(state, std=[1.0, 0.05])
    observed = bernoulli(state, prevalence=0.3)
    data = tokens(observed, vocab_size=32)

    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
    tok = data.get("tokens")
    assert tok is not None
    assert tok.shape == (num_samples, num_timepoints, 1)
