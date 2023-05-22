import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import distrax as dx
from scipy.spatial.distance import pdist, squareform

from coinem.model import AbstractModel
from coinem.dataset import Dataset

# Pyplot for plots.
import matplotlib.pyplot as plt

import os

from jax import vmap

from jax.random import KeyArray
import optax as ox

from coinem.expectation_maximisation import (
    expectation_maximisation,
)

from coinem.maximisation_step import GradientMaximisationState, MaximisationStep
from coinem.expectation_step import (
    ParticleGradientExpectationStep,
    SoulExpectationStep,
    SteinExpectationStep,
)

from coinem.gradient_transforms import GradientTransformation
from coinem.zoo import svgd

from typing import Tuple
from jaxtyping import Float, Array
import jax.tree_util as jtu


@dataclass
class BayesNN(AbstractModel):
    """Base class for p(θ, x)."""

    def log_prob(self, latent: Float[Array, "D"], theta: Float[Array, "Q"], data: Dataset) -> Float[Array, ""]:
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
        log_prior_w = dx.Normal(0.0, jnp.exp(1.0 * theta["alpha"].squeeze())).log_prob(latent["w"].ravel()).sum()
        log_prior_v = dx.Normal(0.0, jnp.exp(1.0 * theta["beta"].squeeze())).log_prob(latent["v"].ravel()).sum()

        def _log_nn(image):
            # Log of the network's output when evaluated at image with weights w, v.
            return jax.nn.log_softmax(jnp.dot(latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten()))))

        def _log_nn_vec(images):
            # _log_nn vectorized over particles.
            return jax.vmap(_log_nn)(images)

        def _log_likelihood(images, labels):
            # Log-likelihood for set of images and labels, vectorized over particles.
            return vmap(lambda v, l: v[l])(_log_nn_vec(images), labels).sum()

        return (_log_likelihood(X, y) + log_prior_w + log_prior_v) / data.n  # log p(y|x) + log p(x|theta)

    def optimal_theta(self, latent_particles: Float[Array, "N D *"]) -> Float[Array, "Q *"]:
        """Optimal parameter for weight cloud w."""
        mom2 = (jax.vmap(_normsq)(latent_particles["w"])).mean()  # Second moment
        mom4 = (jax.vmap(_normsq)(latent_particles["v"])).mean()  # Fourth moment

        return {"alpha": jnp.log(mom2 / (latent_particles["w"][0].size)) / 2,
                "beta": jnp.log(mom4 / (latent_particles["v"][0].size)) / 2}


def _normsq(x):
    # Squared Frobenius norm of x.
    v = x.reshape((x.size))
    return jnp.dot(v, v)


def log_pointwise_predrictive_density(model, latent_particles, images, labels):
    """Returns LPPD for set of (test) images and labels."""

    def _latent_cal(latent):
        def _log_nn(image):
            # Log of the network's output when evaluated at image with weights w, v.
            return jax.nn.log_softmax(jnp.dot(latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten()))))

        def _log_nn_vec(images):
            # _log_nn vectorized over particles.
            return jax.vmap(_log_nn)(images)

        return _log_nn_vec(images)

    s = jax.vmap(_latent_cal)(latent_particles).mean(0)
    return vmap(lambda v, l: v[l])(s, labels).mean()


def _predict(model, latent_particles, images):
    """Returns LPPD for set of (test) images and labels."""

    def _latent_cal(latent):
        def _log_nn(image):
            # Log of the network's output when evaluated at image with weights w, v.
            return jax.nn.log_softmax(jnp.dot(latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten()))))

        def _log_nn_vec(images):
            # _log_nn vectorized over particles.
            return jax.vmap(_log_nn)(images)

        return _log_nn_vec(images)

    s = jax.vmap(_latent_cal)(latent_particles).mean(0)
    return jnp.argmax(s, axis=1)


def test_error(model, latent_particles, images, labels):
    """Returns fraction of misclassified images in test set."""
    return jnp.abs(labels.squeeze() - _predict(model, latent_particles, images)).mean()


import numpy as np

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
images, _, labels, _ = train_test_split(images, labels, train_size=1000,
                                        random_state=0)

# Normalize non-zero entries so that they have mean zero and unit standard
# across the dataset:'''
i = images.std(0) != 0
images[:, i] = (images[:, i] - images[:, i].mean(0))/images[:, i].std(0)


seeds = [1, 2, 3, 4, 5]

# note: we divide by number of data points (data.n) in the objective function to avoid overfloat issues
# thus, the effective step size is data.n * original LR
pgd_step_size_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
#pgd_heuristic_step_size_list = [5e-1, 1e0, 5e0, 1e1, 5e1]
pgd_heuristic_step_size_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
#svgd_step_size_list = [5e-2, 1e-1, 5e-1, 1e0, 5e0]
svgd_step_size_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
#svgd_heuristic_step_size_list = [5e-1, 1e0, 5e0, 1e1, 5e1]
svgd_heuristic_step_size_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
#soul_step_size_list = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
soul_step_size_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
#soul_heuristic_step_size_list = [5e-1, 1e0, 5e0, 1e1, 5e1]
soul_heuristic_step_size_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
coin_step_size_list = ["NA", "NA", "NA", "NA", "NA", "NA"]

step_size_dict_og = {
    "pgd": pgd_step_size_list,
    "pgd_heuristic": pgd_heuristic_step_size_list,
    "svgd": svgd_step_size_list,
    "svgd_heuristic": svgd_heuristic_step_size_list,
    "soul": soul_step_size_list,
    "soul_heuristic": soul_heuristic_step_size_list,
    "coin": coin_step_size_list,
}

pgd_step_size_list = [step * 800 for step in pgd_step_size_list]
pgd_heuristic_step_size_list = [step * 800 for step in pgd_heuristic_step_size_list]
svgd_step_size_list = [step * 800 for step in svgd_step_size_list]
svgd_heuristic_step_size_list = [step * 800 for step in svgd_heuristic_step_size_list]
soul_step_size_list = [step * 800 for step in soul_step_size_list]
soul_heuristic_step_size_list = [step * 800 for step in soul_heuristic_step_size_list]

step_size_dict = {
    "pgd": pgd_step_size_list,
    "pgd_heuristic": pgd_heuristic_step_size_list,
    "svgd": svgd_step_size_list,
    "svgd_heuristic": svgd_heuristic_step_size_list,
    "soul": soul_step_size_list,
    "soul_heuristic": soul_heuristic_step_size_list,
    "coin": coin_step_size_list,
}

N_vals = [2, 5, 10, 20, 50,100]

run_experiments = False
if run_experiments:
    for jj, seed in enumerate(seeds):

        print("Seed: " + str(jj+1) + "/" + str(len(seeds)))

        for ll, N in enumerate(N_vals):

            print("N: " + str(ll+1) + "/" + str(len(N_vals)))

            # Split data into 80/20 training and testing sets:
            itrain, itest, ltrain, ltest = train_test_split(images, labels, test_size=0.2,
                                                            random_state=seed)

            data = Dataset(jnp.array(itrain), jnp.array(ltrain).reshape(-1, 1))

            K = 500 # Number of steps.
            N = N  # Number of particles.

            key = jr.PRNGKey(seed)

            alpha = jnp.array(0.0)
            beta = jnp.array(0.0)

            # Initialize particle cloud by sampling prior:'
            w_init = jnp.exp(alpha) * jr.normal(key, (N, 40, 28**2))  # Input layer weights.
            v_init = jnp.exp(beta) * jr.normal(key, (N, 2, 40))  # Output layer weights.
            theta_init = {"alpha": alpha, "beta": beta}
            latent_init = {"w": w_init, "v": v_init}

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

            model = BayesNN()

            Dw = w_init[0, :, :].size
            Dv = v_init[0, :, :].size


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


            def heuristic_svgd(
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
                    expectation_step=SteinExpectationStep(
                        model=model, optimiser=latent_optimiser
                    ),
                    maximisation_step=HeuristicMaximisationState(
                        model=model, optimiser=theta_optimiser
                    ),
                    data=data,
                    latent_init=latent_init,
                    theta_init=theta_init,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    key=key,
                )

            def standard_heuristic_svgd(
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

                latent_optimiser = ox.sgd(latent_step_size)
                theta_optimiser = ox.sgd(theta_step_size)

                return heuristic_svgd(
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

            for kk in range(len(pgd_step_size_list)):

                print("LR: " + str(kk + 1) + "/" + str(len(pgd_step_size_list)))

                # # Run all methods for grid of step sizes
                results_dir = "new_experiments/results/bayes_nn/summary_results" + "/" + "N_" + str(N) + "/" + "T_" + str(
                    K) + "/" + "LR_" + str(kk)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                x_pgd, theta_pgd = ada_pgd(model, data, latent_init.copy(), theta_init.copy(), K,
                                           latent_step_size=pgd_step_size_list[kk],
                                           theta_step_size=pgd_step_size_list[kk])
                #np.save(results_dir + "/" + "x_pgd_" + str(seed), x_pgd)
                #np.save(results_dir + "/" + "theta_pgd_" + str(seed), theta_pgd)
                lpdd_pgd = vmap(lambda p: log_pointwise_predrictive_density(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_pgd)
                error_pgd = vmap(lambda p: test_error(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_pgd)
                np.save(results_dir + "/" + "lpdd_x_pgd_" + str(seed), lpdd_pgd)
                np.save(results_dir + "/" + "error_x_pgd_" + str(seed), error_pgd)
                del x_pgd
                del theta_pgd
                del lpdd_pgd
                del error_pgd

                x_pgd_heuristic, theta_pgd_heuristic = heuristic_pgd(model, data, latent_init, theta_init, K,
                                                                     theta_step_size=pgd_heuristic_step_size_list[kk],
                                                                     latent_step_size=pgd_heuristic_step_size_list[kk])
                #np.save(results_dir + "/" + "x_pgd_heuristic_" + str(seed), x_pgd_heuristic)
                #np.save(results_dir + "/" + "theta_pgd_heuristic_" + str(seed), theta_pgd_heuristic)
                lpdd_pgd_heuristic = vmap(lambda p: log_pointwise_predrictive_density(model, p, jnp.array(itest),jnp.array(ltest).reshape(-1, 1)))(x_pgd_heuristic)
                error_pgd_heuristic = vmap(lambda p: test_error(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_pgd_heuristic)
                np.save(results_dir + "/" + "lpdd_x_pgd_heuristic_" + str(seed), lpdd_pgd_heuristic)
                np.save(results_dir + "/" + "error_x_pgd_heuristic_" + str(seed), error_pgd_heuristic)
                del x_pgd_heuristic
                del theta_pgd_heuristic
                del lpdd_pgd_heuristic
                del error_pgd_heuristic

                #x_coin, theta_coin = coin_svgd(model, data, latent_init, theta_init, K, alpha=100.0)
                #np.save(results_dir + "/" + "x_coin_" + str(seed), x_coin)
                #np.save(results_dir + "/" + "theta_coin_" + str(seed), theta_coin)
                #lpdd_coin = vmap(lambda p: log_pointwise_predrictive_density(model, p, jnp.array(itest),jnp.array(ltest).reshape(-1, 1)))(x_coin)
                #error_coin = vmap(lambda p: test_error(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_coin)
                #np.save(results_dir + "/" + "lpdd_x_coin_" + str(seed), lpdd_coin)
                #np.save(results_dir + "/" + "error_x_coin_" + str(seed), error_coin)
                #del x_coin
                #del theta_coin
                #del lpdd_coin
                #del error_coin

                x_svgd, theta_svgd = ada_svgd(model, data, latent_init, theta_init, K,
                                              latent_step_size=svgd_step_size_list[kk],
                                              theta_step_size=svgd_step_size_list[kk])
                #np.save(results_dir + "/" + "x_svgd_" + str(seed), x_svgd)
                #np.save(results_dir + "/" + "theta_svgd_" + str(seed), theta_svgd)
                lpdd_svgd = vmap(lambda p: log_pointwise_predrictive_density(model, p, jnp.array(itest),jnp.array(ltest).reshape(-1, 1)))(x_svgd)
                error_svgd = vmap(lambda p: test_error(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_svgd)
                np.save(results_dir + "/" + "lpdd_x_svgd_" + str(seed), lpdd_svgd)
                np.save(results_dir + "/" + "error_x_svgd_" + str(seed), error_svgd)
                del x_svgd
                del theta_svgd
                del lpdd_svgd
                del error_svgd

                x_svgd_heuristic, theta_svgd_heuristic = standard_heuristic_svgd(model, data, latent_init, theta_init, K,
                                                                                 latent_step_size=svgd_heuristic_step_size_list[kk],
                                                                                 theta_step_size=svgd_heuristic_step_size_list[kk])
                #np.save(results_dir + "/" + "x_svgd_heuristic_" + str(seed), x_svgd_heuristic)
                #np.save(results_dir + "/" + "theta_svgd_heuristic_" + str(seed), theta_svgd_heuristic)
                lpdd_svgd_heuristic = vmap(lambda p: log_pointwise_predrictive_density(model, p, jnp.array(itest),jnp.array(ltest).reshape(-1, 1)))(x_svgd_heuristic)
                error_svgd_heuristic = vmap(lambda p: test_error(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_svgd_heuristic)
                np.save(results_dir + "/" + "lpdd_x_svgd_heuristic_" + str(seed), lpdd_svgd_heuristic)
                np.save(results_dir + "/" + "error_x_svgd_heuristic_" + str(seed), error_svgd_heuristic)
                del x_svgd_heuristic
                del theta_svgd_heuristic
                del lpdd_svgd_heuristic
                del error_svgd_heuristic

                x_soul, theta_soul = soul(model, data, latent_init.copy(), theta_init.copy(), K,
                                          latent_step_size=soul_step_size_list[kk],
                                          theta_step_size=soul_step_size_list[kk])
                #np.save(results_dir + "/" + "x_soul_" + str(seed), x_soul)
                #np.save(results_dir + "/" + "theta_soul_" + str(seed), theta_soul)
                lpdd_soul = vmap(lambda p: log_pointwise_predrictive_density(model, p, jnp.array(itest),jnp.array(ltest).reshape(-1, 1)))(x_soul)
                error_soul = vmap(lambda p: test_error(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_soul)
                np.save(results_dir + "/" + "lpdd_x_soul_" + str(seed), lpdd_soul)
                np.save(results_dir + "/" + "error_x_soul_" + str(seed), error_soul)
                del x_soul
                del theta_soul
                del lpdd_soul
                del error_soul

                x_soul_heuristic, theta_soul_heuristic = heuristic_soul(model, data, latent_init.copy(), theta_init.copy(), K,
                                                                        latent_step_size=soul_heuristic_step_size_list[kk],
                                                                        theta_step_size=soul_heuristic_step_size_list[kk])
                #np.save(results_dir + "/" + "x_soul_heuristic_" + str(seed), x_soul_heuristic)
                #np.save(results_dir + "/" + "theta_soul_heuristic_" + str(seed), theta_soul_heuristic)
                lpdd_soul_heuristic = vmap(lambda p: log_pointwise_predrictive_density(model, p, jnp.array(itest),jnp.array(ltest).reshape(-1, 1)))(x_soul_heuristic)
                error_soul_heuristic = vmap(lambda p: test_error(model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)))(x_soul_heuristic)
                np.save(results_dir + "/" + "lpdd_x_soul_heuristic_" + str(seed), lpdd_soul_heuristic)
                np.save(results_dir + "/" + "error_x_soul_heuristic_" + str(seed), error_soul_heuristic)
                del x_soul_heuristic
                del theta_soul_heuristic
                del lpdd_soul_heuristic
                del error_soul_heuristic


# load results and generate plots
generate_plots = True
if generate_plots:

    K = 500 # just needed for file names

    ###############################################
    # compare learning rates, fixed method, fixed N
    methods = ["pgd", "pgd_heuristic", "svgd", "svgd_heuristic", "soul", "soul_heuristic", "coin"]
    names = ["PGD", "PGD'", "SVGD EM", "SVGD EM'", "SOUL", "SOUL'", "Coin EM"]
    seeds = [1, 2, 3, 4, 5]

    fig_dir = "new_experiments/figures/bayes_nn/compare_LR/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for name, method in zip(names, methods):
        for ll, N in enumerate(N_vals):

            for kk in range(len(pgd_step_size_list)-1):
                errors = np.zeros((len(seeds), K))
                errors_coin = np.zeros((len(seeds), K))

                for jj, seed in enumerate(seeds):

                    results_dir = "new_experiments/results/bayes_nn/summary_results" + "/" + "N_" + str(N) + "/" + "T_" + str( K) + "/" + "LR_" + str(kk)
                    fname = results_dir + "/" + "error_x_" + method + "_" + str(seed) + ".npy"
                    errors[jj,:] = np.load(fname)

                    fname_coin = results_dir + "/" + "error_x_" + "coin" + "_" + str(seed) + ".npy"
                    errors_coin[jj, :] = np.load(fname_coin)

                if method == "coin":
                    if kk == 0:
                        plt.plot(range(K), np.mean(errors, axis=0), color="C" + str(kk))
                        plt.fill_between(range(K), np.mean(errors, axis=0) + np.std(errors, axis=0),
                                         np.mean(errors, axis=0) - np.std(errors, axis=0), alpha=0.2, color="C" + str(kk))

                        #plt.title(name + " (N = " + str(N) + ")")

                else:
                    if kk == 0:
                        plt.plot(range(K), np.mean(errors_coin, axis=0), label="Coin EM",
                                 color="C0")
                        plt.fill_between(range(K), np.mean(errors_coin, axis=0) + np.std(errors, axis=0),
                                         np.mean(errors_coin, axis=0) - np.std(errors_coin, axis=0), alpha=0.2, color="C0")

                    plt.plot(range(K), np.mean(errors, axis=0), label="LR = " + str(step_size_dict_og[method][kk]), color="C" + str(kk+1))
                    plt.fill_between(range(K), np.mean(errors, axis=0) + np.std(errors, axis=0),
                                     np.mean(errors, axis=0) - np.std(errors, axis=0), alpha=0.2, color="C" + str(kk+1))

                    #plt.title(name + " (N = " + str(N) + ")")

            if method != "coin":
                plt.legend(prop={'size':18})
            plt.ylim(0, 0.6)
            plt.xlabel("Iterations", fontsize=18)
            plt.ylabel("Error", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid(color='whitesmoke')
            fname = method + "_" + "N" + "_" + str(N)
            plt.savefig(fig_dir + fname + ".pdf", dpi=300, bbox_inches="tight")
            #plt.show()
            plt.close("all")
            # plt.show()

    ##############################################
    # compare N, fixed method, fixed learning rate
    methods = ["pgd", "pgd_heuristic", "svgd", "svgd_heuristic", "soul", "soul_heuristic", "coin"]

    fig_dir = "new_experiments/figures/bayes_nn/compare_N/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for name, method in zip(names, methods):

        if method == "pgd" or "pgd_heuristic" or "svgd" or "svgd_heuristic":
            step_list = pgd_step_size_list[1:]
        if method == "soul" or "soul_heuristic":
            step_list = pgd_step_size_list[:-1]

        for kk, step in enumerate(pgd_step_size_list[:-1]):

            for ll, N in enumerate(N_vals):

                errors = np.zeros((len(seeds), K))
                errors_coin = np.zeros((len(seeds), K))

                for jj, seed in enumerate(seeds):

                    results_dir = "new_experiments/results/bayes_nn/summary_results" + "/" + "N_" + str(N) + "/" + "T_" + str(K) + "/" + "LR_" + str(kk)
                    fname = results_dir + "/" + "error_x_" + method + "_" + str(seed) + ".npy"
                    errors[jj,:] = np.load(fname)
                    fname_coin = results_dir + "/" + "error_x_" + "coin" + "_" + str(seed) + ".npy"
                    errors_coin[jj, :] = np.load(fname_coin)

                plt.plot(range(K), np.mean(errors, axis=0), label="N = " + str(N), color="C" + str(ll))
                plt.fill_between(range(K), np.mean(errors, axis=0) + np.std(errors, axis=0),
                                 np.mean(errors, axis=0) - np.std(errors, axis=0), alpha=0.2, color="C" + str(ll))

                #plt.title(name + " (LR = " + str(step_size_dict[method][kk]) + ")")

            plt.ylim(0, 0.6)
            plt.xlabel("Iterations", fontsize=18)
            plt.ylabel("Error", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(prop={'size': 18}, loc='upper right')
            plt.grid(color='whitesmoke')
            fname = method + "_" + "LR" + "_" + str(step_size_dict_og[method][kk])
            plt.savefig(fig_dir + fname + ".pdf", dpi=300, bbox_inches="tight")
            plt.close("all")
            #plt.show()

    ##############################################
    # compare methods, fixed N, fixed learning rate
    methods = ["coin", "svgd", "pgd", "soul", "svgd_heuristic", "pgd_heuristic", "soul_heuristic"]
    names = ["Coin", "SVGD", "PGD", "SOUL", "SVGD'", "PGD'", "SOUL'"]

    fig_dir = "new_experiments/figures/bayes_nn/compare_methods/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # id of best learning rate for each method
    N_2_best = [0, 4, 4, 3, 4, 4, 4, 0] # 0 is a placeholder for coin
    N_5_best = [0, 4, 4, 3, 4, 4, 4, 0]
    N_10_best = [0, 4, 4, 3, 4, 4, 4, 0]
    N_20_best = [0, 4, 4, 3, 4, 4, 4, 0]
    N_50_best = [0, 4, 4, 3, 4, 4, 4, 0]
    N_100_best = [0, 4, 4, 3, 4, 4, 4, 0]
    N_best = np.array([N_2_best, N_5_best, N_10_best, N_20_best, N_50_best, N_100_best])

    color_dict = {
        "coin": "C0",
        "svgd": "C2",
        "pgd": "C1",
        "soul": "C3",
        "svgd_heuristic": "C4",
        "pgd_heuristic": "C5",
        "soul_heuristic": "C6",
    }

    for ll, N in enumerate(N_vals):

        for mm, (method,name) in enumerate(zip(methods, names)):

            errors = np.zeros((len(seeds), K))

            for jj, seed in enumerate(seeds):

                kk = N_best[ll, mm]

                results_dir = "new_experiments/results/bayes_nn/summary_results" + "/" + "N_" + str(N) + "/" + "T_" + str(K) + "/" + "LR_" + str(kk)
                fname = results_dir + "/" + "error_x_" + method + "_" + str(seed) + ".npy"
                errors[jj,:] = np.load(fname)

            plt.plot(range(K), np.mean(errors, axis=0), label=name, color=color_dict[method])
            plt.fill_between(range(K), np.mean(errors, axis=0) + np.std(errors, axis=0),
                             np.mean(errors, axis=0) - np.std(errors, axis=0), alpha=0.2, color=color_dict[method])

        plt.ylim(0, 0.6)
        plt.xlabel("Iterations", fontsize=18)
        plt.ylabel("Error", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(prop={'size': 18}, ncol=2)
        plt.grid(color='whitesmoke')
        fname = "N" + "_" + str(N)
        plt.savefig(fig_dir + fname + ".pdf", dpi=300, bbox_inches="tight")
        plt.close("all")