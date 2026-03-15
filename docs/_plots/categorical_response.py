import torch
import matplotlib.pyplot as plt
from simulacra import Simulation

torch.manual_seed(0)

data = Simulation(200, 8, 3).linear(4).categorical().data
mu = data["mu"]
y = data["y"]
assert isinstance(mu, torch.Tensor) and isinstance(y, torch.Tensor)
y = y[:, :, 0]  # [N, T]

K = mu.shape[2]
mean_probs = mu.reshape(-1, mu.shape[2]).mean(0)  # [K]
counts = torch.bincount(y.long().flatten(), minlength=K).float()
counts = counts / counts.sum()

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
axes[0].bar(range(K), mean_probs, color=colors)
axes[0].set_xlabel("Class")
axes[0].set_title("Mean probability (softmax)")
axes[1].bar(range(K), counts, color=colors)
axes[1].set_xlabel("Class")
axes[1].set_title("Observed frequency")
plt.tight_layout()
