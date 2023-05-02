from jaxtyping import Array, Float
from simple_pytree import Pytree
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
import jax.tree_util as jtu

import optax as ox


class ExpectationState(NamedTuple):
    optimiser_state: OptimiserState
    key: KeyArray


@dataclass
class AbstractExpectationStep(Pytree):
    """The E-step of the EM algorithm."""

    model: AbstractModel

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


@dataclass
class SteinExpectationStep(AbstractExpectationStep):
    """SVGD, E-step of the EM algorithm."""

    optimiser: GradientTransformation
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
        latent_score = jtu.Partial(
            self.model.score_latent_particles, theta=theta, data=data
        )
        negative_latent_grad = -stein_grad(
            particles=latent, score=latent_score, kernel=self.kernel
        )

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

    step_size: float

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
        latent_new = (
            latent
            + self.step_size * self.model.score_latent_particles(latent, theta, data)
            + jnp.sqrt(2.0 * self.step_size) * jr.normal(subkey, shape=latent.shape)
        )

        # Update expectation state
        expectation_state_new = ExpectationState(optimiser_state=None, key=key)

        return latent_new, expectation_state_new


@dataclass
class SoulExpectationStep(AbstractExpectationStep):
    """The E-step of the SOUL algorithm."""

    step_size: float

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

            new = (
                particle
                + self.step_size * self.model.score_latent(particle, theta, data)
                + jnp.sqrt(2.0 * self.step_size)
                * jr.normal(subkey, shape=particle.shape)
            )

            return (new, key), new

        _, latent_new = lax.scan(
            body_fun, (latent[-1], subkey), jnp.arange(latent.shape[0])
        )

        # Update expectation state
        expectation_state_new = ExpectationState(optimiser_state=None, key=key)

        return latent_new, expectation_state_new
