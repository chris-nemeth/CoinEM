from jaxtyping import Array, Float
from simple_pytree import Pytree
from jax import grad, vmap

from simple_pytree import Pytree
from dataclasses import dataclass
from jaxtyping import Float, Array
from jax import grad, vmap, jacobian
import jax.numpy as jnp

from coinem.dataset import Dataset

__all__ = ["AbstractModel"]


@dataclass
class AbstractModel(Pytree):
    """Base class for p(θ, x)."""

    def log_prob(
        self, latent: Float[Array, "D"], theta: Float[Array, "Q"], data: Dataset
    ) -> Float[Array, ""]:
        """Compute gradient of the objective function at x.

        Args:
            latent (Float[Array, "D"]): Latent variables of shape (D,).
            theta (Float[Array, "Q"]): Parameters of shape (Q,).

        Returns:
            Float[Array, ""]: log-probability of the data.
        """

        raise NotImplementedError

    def score_latent(
        self, latent: Float[Array, "D"], theta: Float[Array, "Q"], data: Dataset
    ) -> Float[Array, "D"]:
        """
        Compute gradient of the objective function at latent variables. FOR A SINGLE PARTICLE.

        Args:
            latent (Float[Array, "D"]): Latent variables of shape (D,).
            theta (Float[Array, "Q"]): Parameters of shape (Q,).

        Returns:
            Float[Array, "D"]: The gradient of the log-density for the latent variables of shape (D,).
        """

        return grad(self.log_prob, argnums=0)(latent, theta, data)

    def score_theta(
        self, latent: Float[Array, "D"], theta: Float[Array, "Q"], data: Dataset
    ) -> Float[Array, "D"]:
        """
        Compute gradient of the objective function at theta.

        Args:
            latent (Float[Array, "D"]): Latent variables of shape (D,).
            theta (Float[Array, "Q"]): Parameters of shape (Q,).

        Returns:
            Float[Array, "D"]: The gradient of the log-density for the latent variables of shape (D,).
        """

        return grad(self.log_prob, argnums=1)(latent, theta, data)

    def score_latent_particles(
        self,
        latent_particles: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Float[Array, "N D"]:
        """
        Compute gradient of the objective function at latent variables. MULTIPLE PARTICLES.

        Args:
            latent_particles (Float[Array, "N D"]): latent_particles variables of shape (N, D).
            theta (Float[Array, "Q"]): Parameters of shape (Q,).

        Returns:
            Float[Array, "N D"]: The gradient of the log-density for the latent latent_particles of shape (N, D).
        """

        return vmap(lambda particle: self.score_latent(particle, theta, data))(
            latent_particles
        )

    def average_score_theta(
        self,
        latent_particles: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Float[Array, "Q"]:
        """
        Compute gradient of the objective function at theta. MULTIPLE PARTICLES.

        Args:
            latent_particles (Float[Array, "N D"]): latent_particles variables of shape (N, D).
            theta (Float[Array, "Q"]): Parameters of shape (Q,).

        Returns:
            Float[Array, "D"]: The gradient of the log-density for theta averaged over the particle cloud of shape (Q,).
        """

        return jnp.mean(
            vmap(lambda particle: self.score_theta(particle, theta, data))(
                latent_particles
            ),
            axis=0,
        )

    def average_hessian_theta(
        self,
        latent_particles: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Float[Array, "Q Q"]:
        """
        Compute gradient of the objective function at theta. MULTIPLE PARTICLES.

        Args:
            latent_particles (Float[Array, "N D"]): latent_particles variables of shape (N, D).
            theta (Float[Array, "Q"]): Parameters of shape (Q,).

        Returns:
            Float[Array, "D"]: The gradient of the log-density for theta averaged over the particle cloud of shape (Q,Q).
        """

        return jnp.mean(
            vmap(
                lambda particle: jacobian(self.score_theta, argnums=1)(
                    particle, theta, data
                )
            )(latent_particles),
            axis=0,
        )  # <- Not checked.
