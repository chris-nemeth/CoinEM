from jaxtyping import Array, Float
from simple_pytree import Pytree, static_field
from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Tuple
from jax.random import KeyArray

import optax as ox
import jax.numpy as jnp
import jax.random as jr

from coinem.model import AbstractModel
from coinem.dataset import Dataset
from coinem.gradient_transforms import GradientTransformation, OptimiserState
from coinem.gradient_flow import stein_grad
from coinem.kernels import AbstractKernel, MedianRBF

from jax import lax
from jax import flatten_util
import jax.tree_util as jtu

import optax as ox


class ExpectationState(NamedTuple):
    optimiser_state: OptimiserState
    key: KeyArray


@dataclass
class AbstractExpectationStep(Pytree):
    """The E-step of the EM algorithm."""

    model: AbstractModel = static_field()

    @abstractmethod
    def init(self, *args, **kwargs) -> ExpectationState:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: ExpectationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "N D"], ExpectationState]:
        raise NotImplementedError


from jaxtyping import PyTree
from typing import Callable

from coinem.gradient_flow import ravel_pytree


@dataclass
class SteinExpectationStep(AbstractExpectationStep):
    """SVGD, E-step of the EM algorithm."""

    optimiser: GradientTransformation = static_field()
    kernel: AbstractKernel = MedianRBF()

    def init(self, params, key) -> ExpectationState:
        return ExpectationState(optimiser_state=self.optimiser.init(params), key=None)

    def update(
        self,
        expectation_state: ExpectationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "N D"], ExpectationState]:
        # Unpack expectation state
        latent_opt_state = expectation_state.optimiser_state

        # Find negative Stein gradient score of the latent particles (negative, since we are maximising, but optimisers minimise!)
        # latent_score = lambda x: self.model.score_latent_particles(
        #     x, theta=theta, data=data
        # )

        # Compute the score function:
        s = self.model.score_latent_particles(latent, theta=theta, data=data)  # ∇x p(x)

        # Flatten the particles and the score function:
        flat_particles, unravel_func = ravel_pytree(latent)
        flat_score, _ = ravel_pytree(s)

        num_particles = flat_particles.shape[0]

        # Compute the kernel and its gradient:
        K, dK = self.kernel.K_dK(flat_particles)  # Kxx, ∇x Kxx

        # Compute the Stein gradient Φ(x) = (Kxx ∇x p(x) + ∇x Kxx) / N:
        flat_stein = (jnp.matmul(K, flat_score) + dK) / num_particles
        negative_flat_stein_grad = -flat_stein
        negative_latent_grad = unravel_func(negative_flat_stein_grad)

        # Find update rule for theta
        latent_updates, latent_new_opt_state = self.optimiser.update(
            negative_latent_grad, latent_opt_state, latent
        )

        # Apply updates to theta
        latent_new = jtu.tree_map(lambda p, u: p + u, latent, latent_updates)

        # Update maximisation state
        maximisation_state_new = ExpectationState(
            optimiser_state=latent_new_opt_state, key=None
        )

        return latent_new, maximisation_state_new


@dataclass
class ParticleGradientExpectationStep(AbstractExpectationStep):
    """The E-step of the PGD algorithm."""

    step_size: float = static_field()

    def init(self, params, key) -> ExpectationState:
        return ExpectationState(optimiser_state=None, key=key)

    def update(
        self,
        expectation_state: ExpectationState,
        latent: Array,
        theta: Array,
        data: Dataset,
    ) -> Tuple[Array, ExpectationState]:
        # Unpack expectation state
        key = expectation_state.key

        # Split the PRNG key
        key, subkey = jr.split(key)

        # Update latent particles
        score_latent_particles = self.model.score_latent_particles(latent, theta, data)
        score_adjusted_latent = jtu.tree_map(
            lambda p, s: p + self.step_size * s, latent, score_latent_particles
        )

        ravel_score_adjusted_latent, unravel = flatten_util.ravel_pytree(
            score_adjusted_latent
        )

        ravel_latent_new = ravel_score_adjusted_latent + jnp.sqrt(
            2.0 * self.step_size
        ) * jr.normal(subkey, shape=ravel_score_adjusted_latent.shape)

        latent_new = unravel(ravel_latent_new)

        # Update expectation state
        expectation_state_new = ExpectationState(optimiser_state=None, key=key)

        return latent_new, expectation_state_new


@dataclass
class SoulExpectationStep(AbstractExpectationStep):
    """The E-step of the SOUL algorithm."""

    step_size: float = static_field()

    def init(self, params, key, *args, **kwargs) -> ExpectationState:
        return ExpectationState(optimiser_state=None, key=key)

    def update(
        self,
        expectation_state: ExpectationState,
        latent: Array,
        theta: Array,
        data: Dataset,
    ) -> Tuple[Array, ExpectationState]:
        # Unpack expectation state
        key = expectation_state.key

        # Split the PRNG key
        key, subkey = jr.split(key)

        # Update latent  via ULA chain

        def body_fun(carry, _):
            particle, key = carry

            key, subkey = jr.split(key)

            score_latent = self.model.score_latent(particle, theta, data)
            score_adjusted_latent = jtu.tree_map(
                lambda p, s: p + self.step_size * s, particle, score_latent
            )

            ravel_score_adjusted_latent, unravel = flatten_util.ravel_pytree(
                score_adjusted_latent
            )

            ravel_latent_new = ravel_score_adjusted_latent + jnp.sqrt(
                2.0 * self.step_size
            ) * jr.normal(subkey, shape=ravel_score_adjusted_latent.shape)

            new = unravel(ravel_latent_new)

            # new = (
            #     particle
            #     + self.step_size * self.model.score_latent(particle, theta, data)
            #     + jnp.sqrt(2.0 * self.step_size)
            #     * jr.normal(subkey, shape=particle.shape)
            # )

            return (new, key), new

        _, latent_new = lax.scan(
            body_fun,
            (jtu.tree_map(lambda l: l[-1], latent), subkey),
            jnp.arange(jtu.tree_leaves(latent)[0].shape[0]),
        )

        # Update expectation state
        expectation_state_new = ExpectationState(optimiser_state=None, key=key)

        return latent_new, expectation_state_new
