import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

failure_times = torch.tensor(
    [
        [0.62, 0.07, 1.47],
        [0.40, 0.41, 0.23],
        [0.06, 0.69, 0.13],
        [0.60, 0.57, 0.85],
        [0.76, 0.23, 0.86],
    ]
)

fig, ax = plt.subplots(figsize=(6, 3.5))
im = ax.imshow(failure_times.numpy(), cmap="YlOrRd", aspect="auto")
ax.set_xlabel("Risk")
ax.set_ylabel("Timepoint")
ax.set_xticks(range(3))
ax.set_yticks(range(5))
ax.set_title("Weibull failure times [T, K]")
for t in range(5):
    for k in range(3):
        ax.text(
            k, t, f"{failure_times[t, k]:.2f}", ha="center", va="center", fontsize=9
        )
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
