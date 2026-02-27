from pathlib import Path

from .simulate import SimulationConfig, simulate


def main():
    cfg = SimulationConfig(
        task="multi_event",
        num_samples=100,
        num_timepoints=10,
        parameters=2,
        scale=1.0,
        latent_std=1.0,
        observed_std=1.0,
        vocab_size=10,
        path=Path("./data/simulation"),
    )
    data = simulate(cfg)
    print(data)
    data.memmap(str(cfg.path))


if __name__ == "__main__":
    main()
