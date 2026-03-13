import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

time = torch.arange(5, dtype=torch.float32)
t_centered = time - time.mean()
Z = torch.stack([torch.ones(5), t_centered], dim=1)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(time.numpy(), Z[:, 0].numpy(), marker="s", label="Intercept")
ax.plot(time.numpy(), Z[:, 1].numpy(), marker="^", label="Slope basis")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Z value")
ax.set_title("Vandermonde basis matrix Z")
ax.legend()
plt.tight_layout()
