# simulacra

Synthetic data generation for generalized linear mixed models.

**simulacra** builds longitudinal datasets one pipeline stage at a time:
design matrices, random effects, response distributions, survival
processes, and competing risks — all as pure tensor operations.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Response families
Gaussian, Poisson, Bernoulli, categorical, and ordinal responses
with the correct link functions.
:::

:::{grid-item-card} Random effects
Vandermonde basis with AR(1), isotropic, or LKJ covariance
structures for subject-level variation.
:::

:::{grid-item-card} Survival & competing risks
Event times, censoring, risk indicators, multi-event
time-to-event, and interval discretization.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

getting-started
tutorials/index
api
```
