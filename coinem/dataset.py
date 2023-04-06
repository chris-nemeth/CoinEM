from __future__ import annotations

from jaxtyping import Array, Float, jaxtyped
from typing import Optional
from simple_pytree import Pytree
from dataclasses import dataclass
from beartype import beartype


@beartype
@jaxtyped
@dataclass
class Dataset(Pytree):
    """Base class for datasets.

    Attributes:
        X (Optional[Float[Array, "N D"]]): Input data.
        y (Optional[Float[Array, "N Q"]]): Output data.
    """

    X: Optional[Float[Array, "N D"]] = None
    y: Optional[Float[Array, "N Q"]] = None

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        repr = (
            f"- Number of observations: {self.n}\n- Input dimension:"
            f" {self.in_dim}\n- Output dimension: {self.out_dim}"
        )
        return repr

    def is_supervised(self) -> bool:
        """Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        """Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    @property
    def n(self) -> int:
        """Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        """Dimension of the inputs, X."""
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        """Dimension of the outputs, y."""
        return self.y.shape[1]


__all__ = [
    "Dataset",
]
