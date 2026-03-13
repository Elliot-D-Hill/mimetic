import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

mu = torch.tensor([0.76, 3.27, 0.31, 0.54, 0.38])
y = torch.tensor([2.0, 4.0, 1.0, 0.0, 0.0])
time = torch.arange(5)

width = 0.35
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(time - width / 2, mu.numpy(), width, label="mu (log link)")
ax.bar(time + width / 2, y.numpy(), width, label="y (count)")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Value")
ax.set_title("Poisson response")
ax.legend()
plt.tight_layout()
