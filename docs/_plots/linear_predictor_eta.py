import torch
import matplotlib.pyplot as plt
from simulacra import Simulation

torch.manual_seed(0)

sim = Simulation(50, 8, 3)
eta = sim.state["eta"]  # [N, T, 1]
time = sim.state["time"][0, :, 0]  # [T]

fig, ax = plt.subplots(figsize=(6, 3))
for i in range(eta.shape[0]):
    ax.plot(time, eta[i, :, 0], alpha=0.3, color="steelblue")
ax.set_xlabel("Timepoint")
ax.set_ylabel("eta")
ax.set_title("Linear predictor across 50 subjects")
plt.tight_layout()
