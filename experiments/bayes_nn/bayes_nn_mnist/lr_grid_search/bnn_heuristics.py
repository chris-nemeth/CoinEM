import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd

from dataclasses import dataclass

from coinem.model import AbstractModel
from coinem.dataset import Dataset

from jaxtyping import Array, Float

import distrax as dx


@dataclass
class BayesNN(AbstractModel):
    """Base class for p(θ, x)."""

    def log_prob(
        self, latent: Float[Array, "D"], theta: Float[Array, "Q"], data: Dataset
    ) -> Float[Array, ""]:
        """Compute gradient of the objective function at x.

        Args:
            latent (Float[Array, "D"]): Input weights of shape (D,).
            theta (Float[Array, "Q"]): Parameters of shape (2,).

        Returns:
            Float[Array, ""]: log-probability of the data.
        """
        X = data.X  # [Batchsize, 28, 28]
        y = data.y  # [Batchsize, 1]

        # Compute prior:
        log_prior_w = (
            dx.Normal(0.0, jnp.sqrt(jnp.exp(2.0 * theta["alpha"].squeeze())))
            .log_prob(latent["w"].ravel())
            .sum()
        )
        log_prior_v = (
            dx.Normal(0.0, jnp.sqrt(jnp.exp(2.0 * theta["beta"].squeeze())))
            .log_prob(latent["v"].ravel())
            .sum()
        )

        def _log_nn(image):
            # Log of the network's output when evaluated at image with weights w, v.
            return jax.nn.log_softmax(
                jnp.dot(latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten())))
            )

        def _log_nn_vec(images):
            # _log_nn vectorized over particles.
            return jax.vmap(_log_nn)(images)

        def _log_likelihood(images, labels):
            # Log-likelihood for set of images and labels, vectorized over particles.
            return vmap(lambda v, l: v[l])(_log_nn_vec(images), labels).sum()

        # Compute log-probability.
        return (
            _log_likelihood(X, y) + log_prior_w + log_prior_v
        )  # / data.n # log p(y|x) + log p(x|theta)

    def optimal_theta(
        self, latent_particles: Float[Array, "N D *"]
    ) -> Float[Array, "Q *"]:
        """Optimal parameter for weight cloud w."""
        mom2 = (jax.vmap(_normsq)(latent_particles["w"])).mean()  # Second moment
        mom4 = (jax.vmap(_normsq)(latent_particles["v"])).mean()  # Fourth moment

        return {
            "alpha": jnp.log(mom2 / (latent_particles["w"][0].size)) / 2,
            "beta": jnp.log(mom4 / (latent_particles["v"][0].size)) / 2,
        }


def _normsq(x):
    # Squared Frobenius norm of x.
    v = x.reshape((x.size))
    return jnp.dot(v, v)


if __name__ == "__main__":
    from jax import vmap

    # @title Load, subsample, and normalize MNIST dataset.
    import numpy as np
    from coinem.dataset import Dataset

    # Load dataset:
    from keras.datasets import mnist

    (images, labels), _ = mnist.load_data()
    images = np.array(images).astype(float)
    labels = np.array(labels).astype(int)

    # Keep only datapoints with labels 4 and 9:
    indices = (labels == 4) | (labels == 9)
    labels = labels[indices]
    images = images[indices, :, :]

    # Relabel as 4 as 0 and 9 as 1:
    for n in range(labels.size):
        if labels[n] == 4:
            labels[n] = 0
        else:
            labels[n] = 1

    # Sub-sample 1000 images:
    from sklearn.model_selection import train_test_split

    images, _, labels, _ = train_test_split(
        images, labels, train_size=1000, random_state=0
    )

    # Normalize non-zero entries so that they have mean zero and unit standard
    # across the dataset:'''
    i = images.std(0) != 0
    images[:, i] = (images[:, i] - images[:, i].mean(0)) / images[:, i].std(0)

    # Number of steps, particles and replicates
    num_steps = 500
    num_particles = 10
    num_replicates = 10
    step_sizes = jnp.logspace(-9, 1, num=50)

    # Model
    model = BayesNN()

    # Rmse calculation
    def log_pointwise_predrictive_density(model, latent_particles, images, labels):
        """Returns LPPD for set of (test) images and labels."""

        def _latent_cal(latent):
            def _log_nn(image):
                # Log of the network's output when evaluated at image with weights w, v.
                return jax.nn.log_softmax(
                    jnp.dot(
                        latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten()))
                    )
                )

            def _log_nn_vec(images):
                # _log_nn vectorized over particles.
                return jax.vmap(_log_nn)(images)

            return _log_nn_vec(images)

        s = jax.vmap(_latent_cal)(latent_particles)
        return vmap(lambda s_: vmap(lambda v, l: v[l])(s_, labels))(s).mean()

    def _predict(model, latent_particles, images):
        """Returns LPPD for set of (test) images and labels."""

        def _latent_cal(latent):
            def _log_nn(image):
                # Log of the network's output when evaluated at image with weights w, v.
                return jax.nn.log_softmax(
                    jnp.dot(
                        latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten()))
                    )
                )

            def _log_nn_vec(images):
                # _log_nn vectorized over particles.
                return jax.vmap(_log_nn)(images)

            return _log_nn_vec(images)

        s = jax.vmap(_latent_cal)(latent_particles).mean(0)
        return jnp.argmax(s, axis=1)

    def test_error(model, latent_particles, images, labels):
        """Returns fraction of misclassified images in test set."""
        return jnp.abs(
            labels.squeeze() - _predict(model, latent_particles, images)
        ).mean()

    from time import perf_counter

    from coinem.zoo import (
        coin_svgd,
        pgd,
        soul,
        adam_svgd,
        ada_svgd,
        rms_pgd,
        rms_soul,
        ada_pgd,
        ada_soul,
        adam_pgd,
        adam_soul,
        rms_svgd,
    )
    from coinem.expectation_maximisation import (
        expectation_maximisation,
    )

    from coinem.maximisation_step import GradientMaximisationState, MaximisationStep
    from coinem.expectation_step import (
        ParticleGradientExpectationStep,
        SoulExpectationStep,
        SteinExpectationStep,
    )

    import pickle

    # Output
    results = {}

    # Store step sizes
    results["step_sizes"] = step_sizes

    # Store runtime
    results["runtime"] = {}
    results["runtime"]["pgd"] = []
    results["runtime"]["soul"] = []
    results["runtime"]["svgd"] = []

    # Store error
    results["error"] = {}
    results["error"]["pgd"] = []
    results["error"]["soul"] = []
    results["error"]["svgd"] = []

    # Store log predictive density
    results["lppd"] = {}
    results["lppd"]["pgd"] = []
    results["lppd"]["soul"] = []
    results["lppd"]["svgd"] = []

    for seed in range(num_replicates):
        key = jr.PRNGKey(seed)

        # Split data into 80/20 training and testing sets:
        itrain, itest, ltrain, ltest = train_test_split(
            images, labels, test_size=0.2, random_state=seed
        )

        train = Dataset(jnp.array(itrain), jnp.array(ltrain).reshape(-1, 1))
        test = Dataset(jnp.array(itest), jnp.array(ltest).reshape(-1, 1))

        lppd = vmap(
            lambda p: log_pointwise_predrictive_density(
                model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)
            )
        )
        error = vmap(
            lambda p: test_error(
                model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)
            )
        )

        # Initialise the model parameters
        key, subkey = jr.split(key)

        alpha = jnp.array(0.0)
        beta = jnp.array(0.0)

        # Initialize particle cloud by sampling prior:'
        w_init = jnp.exp(alpha) * jr.normal(
            subkey, (num_particles, 40, 28**2)
        )  # Input layer weights.
        v_init = jnp.exp(beta) * jr.normal(
            subkey, (num_particles, 2, 40)
        )  # Output layer weights.

        theta_init = {"alpha": alpha, "beta": beta}
        latent_init = {"w": w_init, "v": v_init}

        Dw = 40 * (28**2)
        Dv = 2 * 40

        from typing import Tuple
        from jaxtyping import Float, Array
        import jax.tree_util as jtu

        class HeuristicMaximisationState(MaximisationStep):
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
                average_score_theta = self.model.average_score_theta(
                    latent, theta, data
                )

                # Kuntz Heuristic: divide by dimension of parameter space for the latent component
                average_score_theta["alpha"] = average_score_theta["alpha"] / Dw
                average_score_theta["beta"] = average_score_theta["beta"] / Dv

                negative_average_score_theta = jtu.tree_map(
                    lambda x: -x, average_score_theta
                )

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

        from jax.random import KeyArray
        import optax as ox

        def heuristic_pgd(
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
                maximisation_step=HeuristicMaximisationState(
                    model=model, optimiser=ox.sgd(theta_step_size)
                ),
                data=data,
                latent_init=latent_init,
                theta_init=theta_init,
                num_steps=num_steps,
                batch_size=batch_size,
                key=key,
            )

        def heuristic_soul(
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
                expectation_step=SoulExpectationStep(
                    model=model, step_size=latent_step_size
                ),
                maximisation_step=HeuristicMaximisationState(
                    model=model, optimiser=ox.sgd(theta_step_size)
                ),
                data=data,
                latent_init=latent_init,
                theta_init=theta_init,
                num_steps=num_steps,
                batch_size=batch_size,
                key=key,
            )

        from coinem.kernels import AutoRBF

        def heuristic_svgd(
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
                expectation_step=SteinExpectationStep(
                    model=model,
                    optimiser=ox.adagrad(latent_step_size),
                    kernel=AutoRBF(),
                ),
                maximisation_step=HeuristicMaximisationState(
                    model=model, optimiser=ox.adagrad(theta_step_size)
                ),
                data=data,
                latent_init=latent_init,
                theta_init=theta_init,
                num_steps=num_steps,
                batch_size=batch_size,
                key=key,
            )

        for step_size in step_sizes:
            # Split keys for stochastic algorithms
            key_pgd, key_soul = jr.split(key)

            # Run other algorithms

            # PGD:
            pgd_start = perf_counter()
            latent_pgd, theta_pgd = heuristic_pgd(
                model,
                train,
                latent_init,
                theta_init,
                num_steps,
                theta_step_size=step_size,
                latent_step_size=step_size,
                key=key_pgd,
            )
            pgd_end = perf_counter()

            results["runtime"]["pgd"].append(pgd_end - pgd_start)
            results["error"]["pgd"].append(error(latent_pgd))
            results["lppd"]["pgd"].append(lppd(latent_pgd))

            # SOUL:
            soul_start = perf_counter()
            latent_soul, theta_soul = heuristic_soul(
                model,
                train,
                latent_init,
                theta_init,
                num_steps,
                theta_step_size=step_size,
                latent_step_size=step_size,
                key=key_soul,
            )
            soul_end = perf_counter()

            results["runtime"]["soul"].append(soul_end - soul_start)
            results["error"]["soul"].append(error(latent_soul))
            results["lppd"]["soul"].append(lppd(latent_soul))

            # SVGD:
            svgd_start = perf_counter()
            latent_svgd, theta_svgd = heuristic_svgd(
                model,
                train,
                latent_init,
                theta_init,
                num_steps,
                theta_step_size=step_size,
                latent_step_size=step_size,
                key=key_soul,
            )
            svgd_end = perf_counter()

            results["runtime"]["svgd"].append(svgd_end - svgd_start)
            results["error"]["svgd"].append(error(latent_svgd))
            results["lppd"]["svgd"].append(lppd(latent_svgd))

    with open(f"results/heuristics.pkl", "wb") as f:
        pickle.dump(results, f)
