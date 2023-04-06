from beartype import beartype
from dataclasses import dataclass
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple
from simple_pytree import Pytree, static_field
from functools import partial
from abc import abstractmethod
import jax.numpy as jnp

static_hidden_field = partial(static_field, init=False, repr=False)


@jaxtyped
@beartype
@dataclass
class ComputeDistances(Pytree):
    """Compute distances between points in a batch.

    Args:
        x (Float[Array, "N D"]): Input data.
    """

    x: Float[Array, "N D"]

    # dists attribute for caching distances!
    dists: Float[Array, "N N D"] = static_hidden_field()

    # square_dists attribute for caching sqaured-distances!
    square_dists: Float[Array, "N N"] = static_hidden_field()

    def __post_init__(self):
        # Matrix of entries [(x - y)].
        self.dists = self.x[:, None, :] - self.x[None, :, :]  # [N N D]

        # Matrix of entries [(x - y)^2].
        self.square_dists = jnp.sum(self.dists**2, axis=-1)  # [N N]


@jaxtyped
@beartype
@dataclass
class MedianRBF(Pytree):
    """RBF kernel with automatic bandwidth selection as the median of the pairwise distances."""

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute RBF kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        distances = ComputeDistances(x)
        h = jnp.sqrt(
            0.5
            * jnp.median(distances.square_dists)
            / jnp.log(distances.square_dists.shape[0] + 1.0)
        )

        K = jnp.exp(-0.5 * distances.square_dists / h**2)  # [N N]

        # ∇x Kxx = [∑_i exp(-0.5 * (x - y)^2 / h^2) * (x - y) / h^2]
        dK = jnp.sum(K[:, :, None] * distances.dists, axis=1) / h**2  # [N N D]

        return K, dK  # Kxx, ∇x Kxx


@jaxtyped
@beartype
@dataclass
class RBF(Pytree):
    """RBF kernel with static bandwidth selection.

    Args:
        h (Float[Array, "1"]): Bandwidth parameter if median_trick is False. Defaults to 1.0.
    """

    h: Float[Array, "1"] = jnp.array([1.0])

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute RBF kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        distances = ComputeDistances(x)

        K = jnp.exp(-0.5 * distances.square_dists / self.h**2)  # [N N]

        # ∇x Kxx = [∑_i exp(-0.5 * (x - y)^2 / h^2) * (x - y) / h^2]
        dK = jnp.sum(K[:, :, None] * distances.dists, axis=1) / self.h**2  # [N N D]

        return K, dK  # Kxx, ∇x Kxx


@jaxtyped
@beartype
@dataclass
class AutoKernel(Pytree):
    """Auto diff / vmap kernel."""

    @abstractmethod
    def __call__(
        self, xi: Float[Array, "D"], yi: Float[Array, "D"]
    ) -> Float[Array, "1"]:
        """Compute kernel between two points.

        Args:
            xi (Float[Array, "D"]): First point.
            yi (Float[Array, "D"]): Second point.

        Returns:
            Float[Array, "1"]: Kernel value.
        """
        raise NotImplementedError

    def K(self, x: Float[Array, "N D"]) -> Float[Array, "N N"]:
        """Compute kernel Gram matrix Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Float[Array, "N N"]: Kernel Gram matrix.
        """
        return vmap(lambda xi: vmap(lambda yi: self(xi, yi))(x))(x)

    def dK(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """Compute kernel gradient ∇x Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Float[Array, "N N D"]: Kernel gradient.
        """
        return jnp.sum(
            vmap(lambda yi: vmap(lambda xi: grad(self)(xi, yi))(x))(x), axis=0
        )

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute kernel Gram matrix Kxx and gradient ∇x Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix and gradient.
        """
        return self.K(x), self.dK(x)


@jaxtyped
@beartype
@dataclass
class AutoRBF(AutoKernel):
    """Auto diff / vmap RBF kernel.

    Args:
        h (Float[Array, "1"]): Bandwidth parameter if median_trick is False. Defaults to 1.0.
    """

    h: Float[Array, "1"] = jnp.array([1.0])

    def __call__(
        self, xi: Float[Array, "D"], yi: Float[Array, "D"]
    ) -> Float[Array, "1"]:
        """Compute kernel between two points.

        Args:
            xi (Float[Array, "D"]): First point.
            yi (Float[Array, "D"]): Second point.

        Returns:
            Float[Array, "1"]: Kernel value.
        """
        return jnp.exp(-0.5 * jnp.sum((xi - yi) ** 2) / self.h**2)
