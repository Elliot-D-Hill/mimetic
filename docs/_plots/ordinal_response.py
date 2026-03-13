import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

mu = torch.tensor(
    [
        [0.21, 0.45, 0.28, 0.07],
        [0.06, 0.26, 0.46, 0.23],
        [0.01, 0.07, 0.32, 0.59],
        [0.77, 0.19, 0.03, 0.01],
        [0.22, 0.46, 0.26, 0.06],
    ]
)
y = torch.tensor([1, 0, 3, 0, 2])
T, K = mu.shape
time = np.arange(T)

fig, ax = plt.subplots(figsize=(6, 3))
bottom = np.zeros(T)
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
for k in range(K):
    vals = mu[:, k].numpy()
    ax.bar(time, vals, bottom=bottom, label=f"Class {k}", color=colors[k])
    bottom += vals
for t in range(T):
    ax.annotate(f"y={y[t].item()}", (t, 1.02), ha="center", fontsize=9)
ax.set_xlabel("Timepoint")
ax.set_ylabel("Probability")
ax.set_title("Ordinal response (cumulative logit)")
ax.legend(loc="lower right")
plt.tight_layout()
