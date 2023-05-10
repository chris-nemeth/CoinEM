""" 🐒 Versions of the expectation maximisation algorithm. 🦓 """

import jax.random as jr

from jax.random import KeyArray
from jaxtyping import Array, Float
from typing import Tuple

from coinem.expectation_maximisation import expectation_maximisation
from coinem.model import AbstractModel
from coinem.maximisation_step import MaximisationStep
from coinem.expectation_step import (
    SteinExpectationStep,
    SoulExpectationStep,
    ParticleGradientExpectationStep,
)
from coinem.dataset import Dataset
from coinem.gradient_transforms import cocob, GradientTransformation

import optax as ox


def svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    latent_optimiser: GradientTransformation,
    theta_optimiser: GradientTransformation,
    num_steps: int,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """
    Perform the Stein variational gradient descent EM algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        latent_optimiser (GradientTransformation): The latent optimiser.
        theta_optimiser (GradientTransformation): The parameter optimiser.
        num_steps (int): The number of steps to perform, K.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=SteinExpectationStep(model=model, optimiser=latent_optimiser),
        maximisation_step=MaximisationStep(model=model, optimiser=theta_optimiser),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
    )


def coin_svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    alpha: float = 0.0,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the CoinEM algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    optimiser = cocob(alpha=alpha)

    return svgd(
        model=model,
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        latent_optimiser=optimiser,
        theta_optimiser=optimiser,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
    )


def adam_svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Adam SVGD algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    latent_optimiser = ox.adam(latent_step_size)
    theta_optimiser = ox.adam(theta_step_size)

    return svgd(
        model=model,
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        latent_optimiser=latent_optimiser,
        theta_optimiser=theta_optimiser,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
    )


def ada_svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Adam SVGD algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    latent_optimiser = ox.adagrad(latent_step_size)
    theta_optimiser = ox.adagrad(theta_step_size)

    return svgd(
        model=model,
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        latent_optimiser=latent_optimiser,
        theta_optimiser=theta_optimiser,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
    )


def soul(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
):
    """Perform the SoulEM algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=SoulExpectationStep(model=model, step_size=latent_step_size),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.sgd(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
    )


def pgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Particle Gradient Descent algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """
    return expectation_maximisation(
        expectation_step=ParticleGradientExpectationStep(
            model=model, step_size=latent_step_size
        ),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.sgd(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
    )
