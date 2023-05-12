import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from dataclasses import dataclass
from coinem.model import AbstractModel
from coinem.dataset import Dataset
from jaxtyping import Float, Array, PyTree

from jax.scipy import optimize


@dataclass
class LogisticRegression(AbstractModel):
    """Base class for p(θ, x)."""

    num_datapoints: int

    def log_prob(
        self, latent: Float[Array, "D 1"], theta: Float[Array, "Q"], batch: Dataset
    ) -> Float[Array, ""]:
        """Compute gradient of the objective function at x.

        Args:
            latent (Float[Array, "D"]): Input weights of shape (D,).
            theta (Float[Array, "Q"]): Parameters of shape (Q,).

        Returns:
            Float[Array, ""]: log-probability of the data.
        """
        alpha = jnp.exp(latent["alpha"])
        beta = latent["beta"]
        a0 = jnp.exp(theta["a0"])
        b0 = jnp.exp(theta["b0"])

        # likelihood
        z = jnp.matmul(batch.X, beta)
        log_lik = tfd.Bernoulli(logits=z.squeeze()).log_prob(batch.y.squeeze()).sum()

        # Compute linear predictor.
        z = jnp.matmul(batch.X, beta)

        # Prior
        log_prior = (
            tfd.Normal(loc=0.0, scale=1.0 / jnp.sqrt(alpha))
            .log_prob(beta)
            .sum()
            .squeeze()
        )
        log_prior_alpha = tfd.Gamma(a0, rate=b0).log_prob(alpha).sum().squeeze()

        # Compute log-probability.
        return (
            log_lik * self.num_datapoints / batch.n + log_prior + log_prior_alpha
        ).squeeze()

    def optimal_theta(
        self, latent_particles: PyTree[Float[Array, "N D *"]]
    ) -> PyTree[Float[Array, "Q *"]]:
        samples = jnp.exp(latent_particles["alpha"])

        def gamma_pdf(log_a):
            a = jnp.exp(log_a)
            b = a / samples.mean()
            return -tfd.Gamma(a, b).log_prob(samples).sum()

        log_a_estimate, *_ = optimize.minimize(
            gamma_pdf, jnp.array([1.0]), method="BFGS"
        )
        log_b_estimte = log_a_estimate - jnp.log(samples.mean())

        return {"a0": log_a_estimate.squeeze(), "b0": log_b_estimte.squeeze()}


if __name__ == "__main__":
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
    from coinem.marginal_zoo import (
        marginal_coin_svgd,
        marginal_pgd,
        marginal_soul,
        marginal_ada_svgd,
        marginal_adam_svgd,
    )
    from coinem.uci import Covertype, Banknote, Wisconsin, Cleveland, Haberman
    from sklearn.metrics import roc_auc_score
    from jax import vmap
    import pickle
    import numpy as np

    from time import perf_counter

    datasets = [Covertype(), Wisconsin(), Banknote(), Cleveland(), Haberman()]

    # Number of steps, particles and replicates
    num_steps = 1000
    num_particles = 10
    num_replicates = 10
    step_sizes = jnp.logspace(-9, 1, num=50)

    for data in datasets:
        # Output
        results = {}

        # Store step sizes
        results["step_sizes"] = step_sizes

        # Store runtime
        results["runtime"] = {}
        results["runtime"]["coin_svgd"] = []
        results["runtime"]["adam_svgd"] = []
        results["runtime"]["ada_svgd"] = []
        results["runtime"]["rms_svgd"] = []
        results["runtime"]["marginal_adam_svgd"] = []
        results["runtime"]["marginal_ada_svgd"] = []
        results["runtime"]["marginal_coin_svgd"] = []

        results["runtime"]["pgd"] = []
        results["runtime"]["adam_pgd"] = []
        results["runtime"]["ada_pgd"] = []
        results["runtime"]["rms_pgd"] = []
        results["runtime"]["marginal_pgd"] = []

        results["runtime"]["soul"] = []
        results["runtime"]["adam_soul"] = []
        results["runtime"]["ada_soul"] = []
        results["runtime"]["rms_soul"] = []
        results["runtime"]["marginal_soul"] = []

        # Store AUC
        results["auc"] = {}
        results["auc"]["coin_svgd"] = []
        results["auc"]["adam_svgd"] = []
        results["auc"]["ada_svgd"] = []
        results["auc"]["rms_svgd"] = []
        results["auc"]["marginal_adam_svgd"] = []
        results["auc"]["marginal_ada_svgd"] = []
        results["auc"]["marginal_coin_svgd"] = []

        results["auc"]["pgd"] = []
        results["auc"]["adam_pgd"] = []
        results["auc"]["ada_pgd"] = []
        results["auc"]["rms_pgd"] = []
        results["auc"]["marginal_pgd"] = []

        results["auc"]["soul"] = []
        results["auc"]["adam_soul"] = []
        results["auc"]["ada_soul"] = []
        results["auc"]["rms_soul"] = []
        results["auc"]["marginal_soul"] = []

        # Store error
        results["error"] = {}
        results["error"]["coin_svgd"] = []
        results["error"]["adam_svgd"] = []
        results["error"]["ada_svgd"] = []
        results["error"]["rms_svgd"] = []
        results["error"]["marginal_adam_svgd"] = []
        results["error"]["marginal_ada_svgd"] = []
        results["error"]["marginal_coin_svgd"] = []

        results["error"]["pgd"] = []
        results["error"]["adam_pgd"] = []
        results["error"]["ada_pgd"] = []
        results["error"]["rms_pgd"] = []
        results["error"]["marginal_pgd"] = []

        results["error"]["soul"] = []
        results["error"]["adam_soul"] = []
        results["error"]["ada_soul"] = []
        results["error"]["rms_soul"] = []
        results["error"]["marginal_soul"] = []

        # Store log predictive density
        results["lppd"] = {}
        results["lppd"]["coin_svgd"] = []
        results["lppd"]["adam_svgd"] = []
        results["lppd"]["ada_svgd"] = []
        results["lppd"]["rms_svgd"] = []
        results["lppd"]["marginal_adam_svgd"] = []
        results["lppd"]["marginal_ada_svgd"] = []
        results["lppd"]["marginal_coin_svgd"] = []

        results["lppd"]["pgd"] = []
        results["lppd"]["adam_pgd"] = []
        results["lppd"]["ada_pgd"] = []
        results["lppd"]["rms_pgd"] = []
        results["lppd"]["marginal_pgd"] = []

        results["lppd"]["soul"] = []
        results["lppd"]["adam_soul"] = []
        results["lppd"]["ada_soul"] = []
        results["lppd"]["rms_soul"] = []
        results["lppd"]["marginal_soul"] = []

        # We use a batch size of 100 for datasets with more than 1000 datapoints
        if data.n > 1000:
            batch_size = 100

        else:
            batch_size = -1

        # Create the Bayesian logistic regression model
        model = LogisticRegression(num_datapoints=data.n)

        def predict_prob(test_inputs, latent):
            """Returns label maximizing the approximate posterior predictive
            distribution defined by the cloud X, vectorized over feature vectors f.
            """
            s = vmap(lambda x: tfd.Bernoulli(logits=jnp.matmul(test_inputs, x)).mean())(
                latent
            ).mean(0)
            return s

        for seed in range(num_replicates):
            key = jr.PRNGKey(seed)

            # Split dataset
            train, test = data.preprocess(test_size=0.3, random_state=seed)

            def auc(latent_particles):
                probs = predict_prob(test.X, latent_particles["beta"][-1])

                try:
                    score = roc_auc_score(test.y.squeeze(), probs.squeeze())

                except ValueError:
                    score = jnp.nan

                return score

            def error(latent_particles):
                probs = predict_prob(test.X, latent_particles["beta"][-1])

                return jnp.abs((test.y.squeeze() - (probs.squeeze() > 0.5))).mean()

            def lppd(latent_particles):
                return vmap(
                    lambda x: tfd.Bernoulli(logits=jnp.matmul(test.X, x)).log_prob(
                        test.y.squeeze()
                    )
                )(latent_particles["beta"][-1]).mean()

            # Initialise the model parameters
            key, subkey = jr.split(key)

            # Initialise the latent variables
            np.random.seed(42)
            dim = train.X.shape[-1]
            a0 = 1.0
            b0 = 0.01
            beta0 = np.zeros([num_particles, dim])
            alpha0 = np.random.gamma(a0, b0, num_particles).reshape(-1, 1)
            for i in range(num_particles):
                beta0[i, :] = np.random.normal(0, np.sqrt(1 / alpha0[i]), dim)

            # Create initial state:
            theta_init = {"a0": jnp.log(a0), "b0": jnp.log(b0)}
            latent_init = {"alpha": jnp.array(alpha0), "beta": jnp.array(beta0)}

            # COIN EM:
            coin_start = perf_counter()
            latent_coin, theta_coin = coin_svgd(
                model, train, latent_init, theta_init, num_steps, batch_size=batch_size
            )
            coin_end = perf_counter()

            results["runtime"]["coin_svgd"].append(coin_end - coin_start)
            results["auc"]["coin_svgd"].append(auc(latent_coin))
            results["error"]["coin_svgd"].append(error(latent_coin))
            results["lppd"]["coin_svgd"].append(lppd(latent_coin))

            # Coin marginal SVGD:
            coin_start = perf_counter()
            latent_coin, theta_coin = marginal_coin_svgd(
                model, train, latent_init, theta_init, num_steps, batch_size=batch_size
            )
            coin_end = perf_counter()

            results["runtime"]["marginal_coin_svgd"].append(coin_end - coin_start)
            results["auc"]["marginal_coin_svgd"].append(auc(latent_coin))
            results["error"]["marginal_coin_svgd"].append(error(latent_coin))
            results["lppd"]["marginal_coin_svgd"].append(lppd(latent_coin))

            for step_size in step_sizes:
                # Split keys for stochastic algorithms
                key_pgd, key_soul = jr.split(key)

                # Run other algorithms
                adam_start = perf_counter()
                latent_adam, theta_adam = adam_svgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                )
                adam_end = perf_counter()

                # SVGD
                results["runtime"]["adam_svgd"].append(adam_end - adam_start)
                results["auc"]["adam_svgd"].append(auc(latent_adam))
                results["error"]["adam_svgd"].append(error(latent_adam))
                results["lppd"]["adam_svgd"].append(lppd(latent_adam))

                ada_start = perf_counter()
                latent_ada, theta_ada = ada_svgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                )
                ada_end = perf_counter()

                results["runtime"]["ada_svgd"].append(ada_end - ada_start)
                results["auc"]["ada_svgd"].append(auc(latent_ada))
                results["error"]["ada_svgd"].append(error(latent_ada))
                results["lppd"]["ada_svgd"].append(lppd(latent_ada))

                rms_start = perf_counter()
                latent_rms, theta_rms = rms_svgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                )
                rms_end = perf_counter()

                results["runtime"]["rms_svgd"].append(rms_end - rms_start)
                results["auc"]["rms_svgd"].append(auc(latent_rms))
                results["error"]["rms_svgd"].append(error(latent_rms))
                results["lppd"]["rms_svgd"].append(lppd(latent_rms))

                adam_marginal_svgd_start = perf_counter()
                latent_adam_marginal, theta_adam_marginal = marginal_adam_svgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                )
                adam_marginal_svgd_end = perf_counter()

                results["runtime"]["marginal_adam_svgd"].append(
                    adam_marginal_svgd_end - adam_marginal_svgd_start
                )
                results["auc"]["marginal_adam_svgd"].append(auc(latent_adam_marginal))
                results["error"]["marginal_adam_svgd"].append(
                    error(latent_adam_marginal)
                )
                results["lppd"]["marginal_adam_svgd"].append(lppd(latent_adam_marginal))

                ada_marginal_svgd_start = perf_counter()
                latent_ada_marginal, theta_ada_marginal = marginal_ada_svgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                )
                ada_marginal_svgd_end = perf_counter()

                results["runtime"]["marginal_ada_svgd"].append(
                    ada_marginal_svgd_end - ada_marginal_svgd_start
                )
                results["auc"]["marginal_ada_svgd"].append(auc(latent_ada_marginal))
                results["error"]["marginal_ada_svgd"].append(error(latent_ada_marginal))
                results["lppd"]["marginal_ada_svgd"].append(lppd(latent_ada_marginal))

                # PGD:
                pgd_start = perf_counter()
                latent_pgd, theta_pgd = pgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_pgd,
                )
                pgd_end = perf_counter()

                results["runtime"]["pgd"].append(pgd_end - pgd_start)
                results["auc"]["pgd"].append(auc(latent_pgd))
                results["error"]["pgd"].append(error(latent_pgd))
                results["lppd"]["pgd"].append(lppd(latent_pgd))

                ada_pgd_start = perf_counter()
                latent_ada_pgd, theta_ada_pgd = ada_pgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_pgd,
                )
                ada_pgd_end = perf_counter()

                results["runtime"]["ada_pgd"].append(ada_pgd_end - ada_pgd_start)
                results["auc"]["ada_pgd"].append(auc(latent_ada_pgd))
                results["error"]["ada_pgd"].append(error(latent_ada_pgd))
                results["lppd"]["ada_pgd"].append(lppd(latent_ada_pgd))

                adam_pgd_start = perf_counter()
                latent_adam_pgd, theta_adam_pgd = adam_pgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_pgd,
                )
                adam_pgd_end = perf_counter()

                results["runtime"]["adam_pgd"].append(adam_pgd_end - adam_pgd_start)
                results["auc"]["adam_pgd"].append(auc(latent_adam_pgd))
                results["error"]["adam_pgd"].append(error(latent_adam_pgd))
                results["lppd"]["adam_pgd"].append(lppd(latent_adam_pgd))

                rms_pgd_start = perf_counter()
                latent_rms_pgd, theta_rms_pgd = rms_pgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_pgd,
                )
                rms_pgd_end = perf_counter()

                results["runtime"]["rms_pgd"].append(rms_pgd_end - rms_pgd_start)
                results["auc"]["rms_pgd"].append(auc(latent_rms_pgd))
                results["error"]["rms_pgd"].append(error(latent_rms_pgd))
                results["lppd"]["rms_pgd"].append(lppd(latent_rms_pgd))

                marginal_pgd_start = perf_counter()
                latent_marginal_pgd, theta_marignal_pgd = marginal_pgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_pgd,
                )
                marginal_pgd_end = perf_counter()

                results["runtime"]["marginal_pgd"].append(
                    marginal_pgd_end - marginal_pgd_start
                )
                results["auc"]["marginal_pgd"].append(auc(latent_marginal_pgd))
                results["error"]["marginal_pgd"].append(error(latent_marginal_pgd))
                results["lppd"]["marginal_pgd"].append(lppd(latent_marginal_pgd))

                # SOUL:
                soul_start = perf_counter()
                latent_soul, theta_soul = soul(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_soul,
                )
                soul_end = perf_counter()

                results["runtime"]["soul"].append(soul_end - soul_start)
                results["auc"]["soul"].append(auc(latent_soul))
                results["error"]["soul"].append(error(latent_soul))
                results["lppd"]["soul"].append(lppd(latent_soul))

                ada_soul_start = perf_counter()
                latent_ada_soul, theta_ada_soul = ada_soul(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_soul,
                )
                ada_soul_end = perf_counter()

                results["runtime"]["ada_soul"].append(ada_soul_end - ada_soul_start)
                results["auc"]["ada_soul"].append(auc(latent_ada_soul))
                results["error"]["ada_soul"].append(error(latent_ada_soul))
                results["lppd"]["ada_soul"].append(lppd(latent_ada_soul))

                rms_soul_start = perf_counter()
                latent_rms_soul, theta_rms_soul = rms_soul(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_soul,
                )
                rms_soul_end = perf_counter()

                results["runtime"]["rms_soul"].append(rms_soul_end - rms_soul_start)
                results["auc"]["rms_soul"].append(auc(latent_rms_soul))
                results["error"]["rms_soul"].append(error(latent_rms_soul))
                results["lppd"]["rms_soul"].append(lppd(latent_rms_soul))

                adam_soul_start = perf_counter()
                latent_adam_soul, theta_adam_soul = adam_soul(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_soul,
                )
                adam_soul_end = perf_counter()

                results["runtime"]["adam_soul"].append(adam_soul_end - adam_soul_start)
                results["auc"]["adam_soul"].append(auc(latent_adam_soul))
                results["error"]["adam_soul"].append(error(latent_adam_soul))
                results["lppd"]["adam_soul"].append(lppd(latent_adam_soul))

                adam_soul_start = perf_counter()
                latent_marginal_soul, theta_marignal_soul = marginal_soul(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_soul,
                )
                adam_soul_end = perf_counter()

                results["runtime"]["marginal_soul"].append(
                    adam_soul_end - adam_soul_start
                )
                results["auc"]["marginal_soul"].append(auc(latent_marginal_soul))
                results["error"]["marginal_soul"].append(error(latent_marginal_soul))
                results["lppd"]["marginal_soul"].append(lppd(latent_marginal_soul))

        with open(f"results/{data.name}.pkl", "wb") as f:
            pickle.dump(results, f)
