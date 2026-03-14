import torch
import matplotlib.pyplot as plt
from mimetic import Simulation

torch.manual_seed(0)

data = Simulation(200, 8, 3).bernoulli(prevalence=0.3).data
mu = data["mu"][:, :, 0]  # [N, T]
y = data["y"][:, :, 0]  # [N, T]

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].hist(mu.flatten(), bins=30, color="steelblue", edgecolor="white")
axes[0].set_xlabel("mu")
axes[0].set_title("Probability (logit link)")
axes[1].bar([0, 1], [float((y == 0).sum()), float((y == 1).sum())], color="tomato")
axes[1].set_xlabel("y")
axes[1].set_xticks([0, 1])
axes[1].set_title("Binary outcome")
plt.tight_layout()
