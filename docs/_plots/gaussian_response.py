import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

mu = torch.tensor([-0.97, 1.40, 0.39, -0.09, 0.36])
y = torch.tensor([-1.40, 1.95, -0.14, -0.03, 0.07])
time = torch.arange(5)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(time.numpy(), mu.numpy(), marker="o", label="mu (identity link)")
ax.scatter(time.numpy(), y.numpy(), marker="x", s=80, zorder=5, label="y (observed)")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Value")
ax.set_title("Gaussian response")
ax.legend()
plt.tight_layout()
