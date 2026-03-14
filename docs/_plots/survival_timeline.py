import torch
import matplotlib.pyplot as plt
from simulacra import Simulation

torch.manual_seed(0)

data = (
    Simulation(20, 8, 3)
    .gaussian(std=0.5)
    .event_time()
    .censor_time()
    .survival_indicators()
    .data
)
event_time = data["event_time"][:, 0, 0]  # [N]
censor_time = data["censor_time"][:, 0, 0]  # [N]
observed_time = data["observed_time"][:, 0, 0]  # [N]
indicator = data["indicator"][:, 0, 0]  # [N]

order = observed_time.argsort(descending=True)

fig, ax = plt.subplots(figsize=(6, 4))
for rank, i in enumerate(order):
    t_obs = float(observed_time[i])
    event = bool(indicator[i])
    color = "tomato" if event else "gray"
    ax.barh(rank, t_obs, height=0.7, color=color, alpha=0.7)
    marker = "v" if event else "|"
    ax.plot(t_obs, rank, marker=marker, color="black", markersize=5)

ax.barh([], [], color="tomato", alpha=0.7, label="Event")
ax.barh([], [], color="gray", alpha=0.7, label="Censored")
ax.set_xlabel("Time")
ax.set_ylabel("Subject")
ax.set_title("Survival timeline (20 subjects)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
