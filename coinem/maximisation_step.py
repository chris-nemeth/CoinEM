from jaxtyping import Array, Float
from simple_pytree import Pytree, static_field
from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Tuple
from coinem.model import AbstractModel
from coinem.dataset import Dataset
from coinem.gradient_transforms import GradientTransformation, OptimiserState

import jax.tree_util as jtu

"""State for the maximisation step. """
AbstractMaximisationState = NamedTuple


@dataclass
class AbstractMaximisationStep(Pytree):
    """The M-step of the EM algorithm."""

    model: AbstractModel = static_field()

    def init(self, theta, key) -> AbstractMaximisationState:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: AbstractMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], AbstractMaximisationState]:
        raise NotImplementedError


class GradientMaximisationState(AbstractMaximisationState):
    optimiser_state: OptimiserState


@dataclass
class MaximisationStep(AbstractMaximisationStep):
    """The M-step of the EM algorithm."""

    optimiser: GradientTransformation = static_field()

    def init(self, params, key) -> GradientMaximisationState:
        return GradientMaximisationState(optimiser_state=self.optimiser.init(params))

    def update(
        self,
        maximisation_state: GradientMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], GradientMaximisationState]:
        # Unpack maximisation state
        theta_opt_state = maximisation_state.optimiser_state

        # Find negative average score of theta, since we are maximising, but optimisers minimise.
        average_score_theta = self.model.average_score_theta(latent, theta, data)

        negative_average_score_theta = jtu.tree_map(lambda x: -x, average_score_theta)

        # Find update rule for theta
        theta_updates, theta_new_opt_state = self.optimiser.update(
            negative_average_score_theta, theta_opt_state, theta
        )

        # Apply updates to theta
        theta_new = jtu.tree_map(lambda p, u: p + u, theta, theta_updates)

        # Update maximisation state
        maximisation_state_new = GradientMaximisationState(
            optimiser_state=theta_new_opt_state
        )

        return theta_new, maximisation_state_new


class MarginalMaximisationState(AbstractMaximisationState):
    pass


@dataclass
class MarginalStep(AbstractMaximisationStep):
    """The M-step of the EM algorithm, when the optimal theta is a function of the latent particles only."""

    def init(self, params, key) -> MarginalMaximisationState:
        return MarginalMaximisationState()

    def update(
        self,
        maximisation_state: MarginalMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], MarginalMaximisationState]:
        # Find update rule for theta as a function of the particle cloud only
        theta_new = self.model.optimal_theta(latent)

        return theta_new, maximisation_state
