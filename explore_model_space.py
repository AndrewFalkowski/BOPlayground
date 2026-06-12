import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.generation.gen import gen_candidates_torch
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.synthetic import Branin
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch_community.acquisition.bayesian_active_learning import (
    qHyperparameterInformedPredictiveExploration,
)

torch.manual_seed(42)
torch.set_default_dtype(torch.double)


@dataclass(frozen=True)
class RunConfig:
    name: str
    n_initial_points: int
    n_batches: int
    batch_size: int
    n_rmse_points: int
    n_integration_points: int
    nuts_samples: int
    nuts_warmup: int
    nuts_thinning: int
    num_restarts: int
    raw_samples: int
    acq_opt_steps: int
    acq_opt_lr: float
    mc_samples: int
    beta_tuning_samples: int
    acq_batch_limit: int
    init_batch_limit: int


SMOKE_CONFIG = RunConfig(
    name="smoke",
    n_initial_points=5,
    n_batches=1,
    batch_size=2,
    n_rmse_points=64,
    n_integration_points=32,
    nuts_samples=32,
    nuts_warmup=32,
    nuts_thinning=1,
    num_restarts=2,
    raw_samples=32,
    acq_opt_steps=20,
    acq_opt_lr=0.05,
    mc_samples=16,
    beta_tuning_samples=8,
    acq_batch_limit=1,
    init_batch_limit=8,
)

PRODUCTION_CONFIG = RunConfig(
    name="production",
    n_initial_points=8,
    n_batches=15,
    batch_size=3,
    n_rmse_points=512,
    n_integration_points=256,
    nuts_samples=128,
    nuts_warmup=512,
    nuts_thinning=4,
    num_restarts=4,
    raw_samples=64,
    acq_opt_steps=80,
    acq_opt_lr=0.03,
    mc_samples=64,
    beta_tuning_samples=32,
    acq_batch_limit=1,
    init_batch_limit=4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore Branin with HIPE and a fully Bayesian GP."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke-test configuration instead of the default run.",
    )
    return parser.parse_args()


def build_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    unit_bounds: torch.Tensor,
    dim: int,
) -> FullyBayesianSingleTaskGP:
    return FullyBayesianSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=dim, bounds=unit_bounds),
        outcome_transform=Standardize(m=1),
    )


def main() -> None:
    args = parse_args()
    config = SMOKE_CONFIG if args.smoke else PRODUCTION_CONFIG

    problem = Branin(negate=True)
    native_bounds = problem.bounds

    # or manually specified, bounds are ordered as [lower, upper] for each dimension
    # native_bounds = torch.tensor([[-5., 0.], [10., 15.]])

    unit_bounds = normalize(native_bounds, bounds=native_bounds)

    train_x = torch.rand(config.n_initial_points, problem.dim)
    train_x_native = unnormalize(train_x, bounds=native_bounds)
    train_y = problem(train_x_native).unsqueeze(-1)

    # Fixed points used only to measure how well the model has learned the surface.
    rmse_x = draw_sobol_samples(
        bounds=unit_bounds, n=config.n_rmse_points, q=1
    ).squeeze(1)
    rmse_x_native = unnormalize(rmse_x, bounds=native_bounds)
    rmse_y_true = problem(rmse_x_native).unsqueeze(-1)

    # Points used by HIPE to measure predictive information over the space.
    integration_x = draw_sobol_samples(
        bounds=unit_bounds,
        n=config.n_integration_points,
        q=1,
    ).squeeze(1)

    print(
        f"Exploring {problem.__class__.__name__} with HIPE "
        f"using {config.name} settings in batches of {config.batch_size}"
    )

    rmse_trace = []

    for batch in range(1, config.n_batches + 1):
        model = build_model(train_x, train_y, unit_bounds, problem.dim)

        # fit model by sampling the GP hyperparameter posterior with NUTS
        fit_fully_bayesian_model_nuts(
            model,
            num_samples=config.nuts_samples,
            warmup_steps=config.nuts_warmup,
            thinning=config.nuts_thinning,
            disable_progbar=True,
        )

        # measure how close the model's mean prediction is to the true function
        posterior = model.posterior(rmse_x)
        rmse = torch.sqrt(
            torch.mean((posterior.mixture_mean - rmse_y_true) ** 2)
        ).item()
        rmse_trace.append(rmse)

        # specify HIPE acquisition function
        acquisition = qHyperparameterInformedPredictiveExploration(
            model=model,
            mc_points=integration_x,
            bounds=unit_bounds,
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([config.mc_samples]),
                seed=42 + batch,
            ),
            beta_tuning_samples=config.beta_tuning_samples,
        )

        # optimize acqf to get the next batch of selected candidate points
        #
        # HIPE conditions one fully Bayesian model per candidate batch, so
        # evaluating many raw samples or restarts in parallel can allocate very
        # large dense covariance tensors. Keep those batches explicitly bounded.
        candidates, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=unit_bounds,
            q=config.batch_size,
            num_restarts=config.num_restarts,
            raw_samples=config.raw_samples,
            gen_candidates=gen_candidates_torch,
            options={
                "maxiter": config.acq_opt_steps,
                "lr": config.acq_opt_lr,
                "batch_limit": config.acq_batch_limit,
                "init_batch_limit": config.init_batch_limit,
            },
        )
        candidates = candidates.detach()

        candidates_native = unnormalize(candidates, bounds=native_bounds)
        candidate_y = problem(candidates_native).unsqueeze(-1)

        train_x = torch.cat([train_x, candidates])
        train_y = torch.cat([train_y, candidate_y])

        print(
            f"batch {batch:02d} | "
            f"rmse={rmse: 9.4f} | "
            f"best_y={train_y.max().item(): 9.4f} | "
            f"n_points={len(train_y):02d} | "
        )

    # Fit once more after the final batch so the final RMSE includes all points.
    model = build_model(train_x, train_y, unit_bounds, problem.dim)
    fit_fully_bayesian_model_nuts(
        model,
        num_samples=config.nuts_samples,
        warmup_steps=config.nuts_warmup,
        thinning=config.nuts_thinning,
        disable_progbar=True,
    )
    posterior = model.posterior(rmse_x)
    rmse = torch.sqrt(torch.mean((posterior.mixture_mean - rmse_y_true) ** 2)).item()
    rmse_trace.append(rmse)

    print("\nResult")
    print(f"final rmse: {rmse: .4f}")
    print(f"number of observations: {len(train_y)}")

    rmse_steps = [
        config.n_initial_points + i * config.batch_size for i in range(len(rmse_trace))
    ]

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

    axs[0].plot(rmse_steps, rmse_trace, marker=".", color="dodgerblue", label="RMSE")
    axs[0].set_xlabel("number of observations")
    axs[0].set_ylabel("RMSE")
    axs[0].legend()

    axs[1].scatter(
        train_x[: config.n_initial_points, 0],
        train_x[: config.n_initial_points, 1],
        color="gray",
        label="initial",
    )
    axs[1].scatter(
        train_x[config.n_initial_points :, 0],
        train_x[config.n_initial_points :, 1],
        color="black",
        label="HIPE exploration",
    )
    axs[1].set_xlabel("x1 normalized")
    axs[1].set_ylabel("x2 normalized")
    axs[1].legend()

    fig.tight_layout()
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    plot_path = figures_dir / "model_space_error_reduction.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved exploration plot to {plot_path}")


if __name__ == "__main__":
    main()
