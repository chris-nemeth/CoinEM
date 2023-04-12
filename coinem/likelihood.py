from beartype import beartype
from jaxtyping import Array, Float, jaxtyped
from simple_pytree import Pytree
from jax import grad, vmap

from abc import abstractmethod


@jaxtyped
@beartype
class AbstractLikelihood(Pytree):
    """Base class for gradients."""

    @abstractmethod
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """Compute gradient of the objective function at x.

        Args:
            x (Float[Array, "N D"]): Input data.

        Returns:
            Float[Array, "N D"]: Gradient of the objective function at x.
        """
        raise NotImplementedError

    def log_prob(self, x: Float[Array, "D"]) -> Float[Array, "1"]:
        raise NotImplementedError

    def score(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """Construct a score function from a log-probability function ∇x p(x)."""
        return vmap(lambda xi: grad(self.log_prob)(xi))(x)
