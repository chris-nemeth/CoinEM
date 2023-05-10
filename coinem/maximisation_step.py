from jaxtyping import Array, Float
from simple_pytree import Pytree
from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Tuple
from coinem.model import AbstractModel
from coinem.dataset import Dataset
from coinem.gradient_transforms import GradientTransformation, OptimiserState

import jax.tree_util as jtu


class MaximisationState(NamedTuple):
    optimiser_state: OptimiserState


@dataclass
class AbstractMaximisationStep(Pytree):
    """The M-step of the EM algorithm."""

    model: AbstractModel
    optimiser: GradientTransformation

    def init(self, params, key) -> MaximisationState:
        return MaximisationState(optimiser_state=self.optimiser.init(params))

    @abstractmethod
    def update(
        self,
        state: MaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], MaximisationState]:
        raise NotImplementedError


@dataclass
class MaximisationStep(AbstractMaximisationStep):
    """The M-step of the EM algorithm."""

    model: AbstractModel
    optimiser: GradientTransformation

    def update(
        self,
        maximisation_state: MaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], MaximisationState]:
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
        maximisation_state_new = MaximisationState(optimiser_state=theta_new_opt_state)

        return theta_new, maximisation_state_new
