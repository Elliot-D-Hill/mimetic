import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

mu = torch.tensor([0.42, 0.21, 0.39, 0.83, 0.32])
y = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
time = torch.arange(5)

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(time.numpy(), mu.numpy(), color="steelblue", alpha=0.7, label="mu (probability)")
ax.scatter(
    time.numpy(), y.numpy(), marker="D", s=60, color="tomato", zorder=5, label="y (0/1)"
)
ax.set_xlabel("Timepoint")
ax.set_ylabel("Value")
ax.set_title("Bernoulli response (prevalence=0.3)")
ax.legend()
plt.tight_layout()
