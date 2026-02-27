from . import simulate


def main():
    data = simulate(
        task="multi_event",
        num_samples=100,
        num_timepoints=10,
        parameters=2,
        scale=1.0,
        latent_std=1.0,
        observed_std=1.0,
        vocab_size=1000,
        covariance_type="ar1",
        rho=0.9,
    )
    print(data)
    path = "./data/simulation"
    data.memmap(path)


if __name__ == "__main__":
    main()
