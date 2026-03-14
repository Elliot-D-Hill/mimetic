# Getting started

## Installation

```bash
uv add simulacra
```

## Minimal example

Generate a longitudinal Gaussian dataset for 100 subjects over 8
timepoints with 3 fixed-effect predictors:

```python
import torch
from simulacra import Simulation

torch.manual_seed(42)
data = (
    Simulation(100, 8, 3)
    .gaussian(std=0.5)
    .data
)
print(data["y"].shape)  # torch.Size([100, 8, 1])
```

Each method returns the next step in the pipeline, and `.data` exports
the accumulated tensors as a `TensorDict`. Chain methods to build the
dataset you need.
