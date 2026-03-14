import torch
import matplotlib.pyplot as plt
from mimetic import Simulation

torch.manual_seed(0)

data = Simulation(200, 8, 3).poisson().data
mu = data["mu"][:, :, 0]  # [N, T]
y = data["y"][:, :, 0]  # [N, T]

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].hist(mu.flatten(), bins=30, color="steelblue", edgecolor="white")
axes[0].set_xlabel("mu")
axes[0].set_title("Rate (log link)")
axes[1].hist(
    y.flatten(), bins=range(int(y.max()) + 2), color="tomato", edgecolor="white"
)
axes[1].set_xlabel("y")
axes[1].set_title("Count")
plt.tight_layout()
