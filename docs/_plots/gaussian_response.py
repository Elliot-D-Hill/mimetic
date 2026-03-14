import torch
import matplotlib.pyplot as plt
from mimetic import Simulation

torch.manual_seed(0)

data = Simulation(50, 8, 3).gaussian(std=0.5).data
mu = data["mu"]  # [N, T, 1]
y = data["y"]  # [N, T, 1]
time = data["time"][0, :, 0]  # [T]

fig, ax = plt.subplots(figsize=(6, 3))
for i in range(mu.shape[0]):
    ax.plot(time, mu[i, :, 0], alpha=0.15, color="steelblue")
    ax.scatter(time, y[i, :, 0], alpha=0.15, color="tomato", s=10)
ax.plot([], [], color="steelblue", label="mu (identity link)")
ax.scatter([], [], color="tomato", s=10, label="y (observed)")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Value")
ax.set_title("Gaussian response")
ax.legend()
plt.tight_layout()
