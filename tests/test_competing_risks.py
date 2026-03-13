import torch

from mimetic import CompetingRisksStep, Simulation


K = 3
N = 16
T = 6
P = 4


def _cr_state() -> CompetingRisksStep:
    """Helper: build a competing risks state with K=3."""
    torch.manual_seed(42)
    return Simulation(N, T, P).linear(K).competing_risks()


def test_competing_risks_shapes() -> None:
    """Smoke: .linear(K).competing_risks() produces correct shapes.

    Tests: failure_times [N, T, K], tokens [N, T, 1].
    """
    data = _cr_state().data
    assert data["failure_times"].shape == (N, T, K)
    assert data["tokens"].shape == (N, T, 1)


def test_risk_indicators_event_time_equals_minimum() -> None:
    """After risk_indicators, event_time equals the broadcast min failure time.

    Tests: event_time[:,:,k] == failure_times.min(dim=-1) for all k.
    """
    data = _cr_state().risk_indicators().data
    min_ft = data["failure_times"].min(dim=-1, keepdim=True).values  # [N, T, 1]
    assert torch.allclose(data["event_time"], min_ft.expand(-1, -1, K))


def test_competing_risks_tokens_index_minimum() -> None:
    """tokens holds the argmin risk index.

    Tests: tokens == failure_times.argmin(dim=-1).
    """
    data = _cr_state().data
    expected = data["failure_times"].argmin(dim=-1, keepdim=True)
    assert torch.equal(data["tokens"], expected)


def test_competing_risks_preserves_time() -> None:
    """time is unchanged after competing_risks (regression against old cumsum overwrite).

    Tests: time matches original observation schedule.
    """
    torch.manual_seed(42)
    sim = Simulation(N, T, P).linear(K)
    original_time = sim.data["time"].clone()
    cr_data = sim.competing_risks().data
    assert torch.equal(cr_data["time"], original_time)


def test_risk_indicators_one_hot() -> None:
    """Indicator sums to 1.0 along K and has shape [N, T, K].

    Tests: one-hot encoding of winning risk.
    """
    data = _cr_state().risk_indicators().data
    assert data["indicator"].shape == (N, T, K)
    assert torch.allclose(data["indicator"].sum(dim=-1), torch.ones(N, T))


def test_risk_indicators_event_time_broadcast() -> None:
    """All K columns of event_time are identical after risk_indicators.

    Tests: event_time [N, T, K] is broadcast from [N, T, 1].
    """
    data = _cr_state().risk_indicators().data
    assert data["event_time"].shape == (N, T, K)
    assert torch.equal(data["event_time"][..., 0], data["event_time"][..., 1])
    assert torch.equal(data["event_time"][..., 1], data["event_time"][..., 2])


def test_multi_event_shapes() -> None:
    """multi_event produces indicator and event_time with shape [N, T, K].

    Tests: output tensor shapes.
    """
    data = _cr_state().multi_event(horizon=10.0).data
    assert data["indicator"].shape == (N, T, K)
    assert data["event_time"].shape == (N, T, K)


def test_multi_event_horizon_clamp() -> None:
    """event_time.max() <= horizon after clamping.

    Tests: horizon ceiling is respected.
    """
    horizon = 5.0
    data = _cr_state().multi_event(horizon=horizon).data
    assert data["event_time"].max().item() <= horizon


def test_discretize_risk_shapes() -> None:
    """discrete_event_time has shape [N, T, K, J] where J = len(boundaries)-1.

    Tests: discretization output dimensionality.
    """
    boundaries = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
    J = len(boundaries) - 1
    data = _cr_state().risk_indicators().discretize(boundaries).data
    assert data["discrete_event_time"].shape == (N, T, K, J)


def test_competing_risks_full_chain() -> None:
    """End-to-end: .linear(K).competing_risks().risk_indicators().discretize(boundaries).data.

    Tests: complete chain produces all expected keys.
    """
    torch.manual_seed(42)
    boundaries = torch.tensor([0.0, 0.5, 1.0, 2.0])
    data = (
        Simulation(N, T, P)
        .linear(K)
        .competing_risks()
        .risk_indicators()
        .discretize(boundaries)
        .data
    )
    assert "failure_times" in data.keys()
    assert "indicator" in data.keys()
    assert "discrete_event_time" in data.keys()
    assert data["discrete_event_time"].shape == (N, T, K, 3)
