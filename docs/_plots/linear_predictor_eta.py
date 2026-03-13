import torch
import matplotlib.pyplot as plt
from mimetic import Simulation

torch.manual_seed(0)

sim = Simulation(1, 5, 2)
eta = sim.state["eta"].squeeze()
time = sim.state["time"].squeeze()

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(time.numpy(), eta.numpy(), marker="o", linewidth=2)
ax.set_xlabel("Timepoint")
ax.set_ylabel("eta")
ax.set_title("Linear predictor over time")
plt.tight_layout()
