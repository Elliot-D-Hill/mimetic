import torch
import matplotlib.pyplot as plt
from mimetic import Simulation

torch.manual_seed(0)

data = Simulation(1, 8, 3).linear(4).competing_risks().data
failure_times = data["failure_times"][0]  # [T, K]

fig, ax = plt.subplots(figsize=(6, 3.5))
im = ax.imshow(failure_times, cmap="YlOrRd", aspect="auto")
ax.set_xlabel("Risk")
ax.set_ylabel("Timepoint")
ax.set_xticks(range(failure_times.shape[1]))
ax.set_yticks(range(failure_times.shape[0]))
ax.set_title("Weibull failure times [T, K]")
for t in range(failure_times.shape[0]):
    for k in range(failure_times.shape[1]):
        ax.text(
            k, t, f"{failure_times[t, k]:.2f}", ha="center", va="center", fontsize=8
        )
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
