import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

event_time = torch.tensor(
    [
        [8.0, 1.0, 3.0],
        [8.0, 1.0, 2.0],
        [8.0, 2.0, 1.0],
        [8.0, 1.0, 8.0],
        [8.0, 8.0, 8.0],
    ]
)
T, K = event_time.shape
time = np.arange(T)
width = 0.25
colors = ["#4c72b0", "#dd8452", "#55a868"]

fig, ax = plt.subplots(figsize=(6, 3.5))
for k in range(K):
    offset = (k - 1) * width
    ax.bar(
        time + offset,
        event_time[:, k].numpy(),
        width,
        label=f"Risk {k}",
        color=colors[k],
    )
ax.axhline(8.0, color="gray", linestyle="--", linewidth=0.8, label="Horizon")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Time-to-event")
ax.set_title("Multi-event TTE per risk")
ax.legend(fontsize=8)
plt.tight_layout()
