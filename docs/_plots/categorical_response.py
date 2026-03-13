import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

mu = torch.tensor(
    [
        [0.42, 0.06, 0.52],
        [0.06, 0.90, 0.04],
        [0.25, 0.53, 0.23],
        [0.35, 0.29, 0.36],
        [0.25, 0.51, 0.24],
    ]
)
y = torch.tensor([0, 1, 0, 0, 1])
T, K = mu.shape
time = np.arange(T)

fig, ax = plt.subplots(figsize=(6, 3))
bottom = np.zeros(T)
colors = ["#4c72b0", "#dd8452", "#55a868"]
for k in range(K):
    vals = mu[:, k].numpy()
    ax.bar(time, vals, bottom=bottom, label=f"Class {k}", color=colors[k])
    bottom += vals
for t in range(T):
    ax.annotate(f"y={y[t].item()}", (t, 1.02), ha="center", fontsize=9)
ax.set_xlabel("Timepoint")
ax.set_ylabel("Probability")
ax.set_title("Categorical response (softmax)")
ax.legend(loc="lower right")
plt.tight_layout()
