import torch
import matplotlib.pyplot as plt
from simulacra import Simulation

torch.manual_seed(0)

data = Simulation(1, 8, 3).linear(4).competing_risks().multi_event(horizon=10.0).data
event_time = data["event_time"][0]  # [T, K]
T, K = event_time.shape
time = torch.arange(T)
width = 0.2
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

fig, ax = plt.subplots(figsize=(6, 3.5))
for k in range(K):
    offset = (k - (K - 1) / 2) * width
    ax.bar(time + offset, event_time[:, k], width, label=f"Risk {k}", color=colors[k])
ax.axhline(10.0, color="gray", linestyle="--", linewidth=0.8, label="Horizon")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Time-to-event")
ax.set_title("Multi-event TTE per risk")
ax.legend(fontsize=8)
plt.tight_layout()
