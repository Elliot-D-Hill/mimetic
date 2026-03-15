"""Value object types for runtime parameter validation via beartype."""

from typing import Annotated

from beartype.vale import Is

UnitInterval = Annotated[float, Is[lambda x: 0.0 <= x <= 1.0]]
"""Probability or proportion in [0, 1]."""

Correlation = Annotated[float, Is[lambda x: 0.0 <= x < 1.0]]
"""Off-diagonal correlation in [0, 1); rho=1 gives singular covariance."""

PositiveFloat = Annotated[float, Is[lambda x: x > 0.0]]
"""Strictly positive real number."""

PositiveInt = Annotated[int, Is[lambda x: x >= 1]]
"""Positive integer (>= 1)."""
