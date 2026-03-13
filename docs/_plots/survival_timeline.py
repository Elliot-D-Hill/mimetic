import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

event_time = 0.58
censor_time = 3.19
observed_time = min(event_time, censor_time)
time = torch.arange(5, dtype=torch.float32)

fig, ax = plt.subplots(figsize=(7, 2.5))

ax.hlines(0.5, 0, 4, linewidth=3, color="lightgray", zorder=1)
ax.axvspan(0, observed_time, alpha=0.15, color="steelblue", label="Observed period")
ax.plot(
    event_time,
    0.5,
    marker="v",
    markersize=12,
    color="tomato",
    zorder=5,
    label=f"Event (t={event_time:.2f})",
)
ax.plot(
    censor_time,
    0.5,
    marker="^",
    markersize=12,
    color="gray",
    zorder=5,
    label=f"Censor (t={censor_time:.2f})",
)

for t in time.numpy():
    ax.plot(t, 0.5, marker="|", markersize=10, color="black", zorder=4)

ax.set_xlabel("Time")
ax.set_yticks([])
ax.set_xlim(-0.3, 4.5)
ax.set_title("Survival timeline (indicator = 1, event observed)")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
