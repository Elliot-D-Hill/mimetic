import torch
import matplotlib.pyplot as plt
from simulacra import Simulation

torch.manual_seed(0)

sim = Simulation(20, 8, 3).random_effects(std=[0.5, 0.3])
eta = sim.state["eta"]  # [N, T, 1]
time = sim.state["time"][0, :, 0]  # [T]
cmap = plt.colormaps["tab10"]

fig, ax = plt.subplots(figsize=(6, 3))
for i in range(eta.shape[0]):
    ax.plot(time, eta[i, :, 0], color=cmap(i % 10), alpha=0.7)
ax.set_xlabel("Timepoint")
ax.set_ylabel("eta")
ax.set_title("Subject-specific trajectories with random effects")
plt.tight_layout()
