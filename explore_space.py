import matplotlib.pyplot as plt
import torch
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.fit import fit_gpytorch_mll
from botorch.generation.gen import gen_candidates_torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions.synthetic import Branin
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(42)
torch.set_default_dtype(torch.double)

N_INITIAL_POINTS = 5
N_BATCHES = 10
BATCH_SIZE = 3
N_RMSE_POINTS = 512
N_INTEGRATION_POINTS = 512  # more points gives a better NIPV estimate, but is slower
NUM_RESTARTS = 10  # more restarts makes acquisition optimization more robust
RAW_SAMPLES = 256  # more raw samples gives better optimizer starting points
ACQ_OPT_STEPS = 100  # more torch optimizer steps can improve candidates, but is slower
ACQ_OPT_LR = 0.05  # lower can be steadier; higher can move faster but overshoot


def main() -> None:

    problem = Branin(negate=True)
    native_bounds = problem.bounds

    # or manually specified, bounds are ordered as [lower, upper] for each dimension
    # native_bounds = torch.tensor([[-5., 0.], [10., 15.]])

    unit_bounds = normalize(native_bounds, bounds=native_bounds)

    train_x = torch.rand(N_INITIAL_POINTS, problem.dim)  # randomly sample unit points
    train_x_native = unnormalize(train_x, bounds=native_bounds)
    train_y = problem(train_x_native).unsqueeze(-1)

    # Fixed points used only to measure how well the model has learned the surface.
    rmse_x = draw_sobol_samples(bounds=unit_bounds, n=N_RMSE_POINTS, q=1).squeeze(1)
    rmse_x_native = unnormalize(rmse_x, bounds=native_bounds)
    rmse_y_true = problem(rmse_x_native).unsqueeze(-1)

    # Points used by NIPV to measure average posterior uncertainty over the space.
    integration_x = draw_sobol_samples(
        bounds=unit_bounds,
        n=N_INTEGRATION_POINTS,
        q=1,
    ).squeeze(1)

    print(f"Exploring {problem.__class__.__name__} in batches of {BATCH_SIZE}")

    rmse_trace = []

    for batch in range(1, N_BATCHES + 1):
        # specify model
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            input_transform=Normalize(d=problem.dim),  # built in normalization
            outcome_transform=Standardize(m=1),  # built in standardization
        )

        # fit model
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        # measure how close the model's mean prediction is to the true function
        posterior = model.posterior(rmse_x)
        rmse = torch.sqrt(torch.mean((posterior.mean - rmse_y_true) ** 2)).item()
        rmse_trace.append(rmse)

        # specify exploration acquisition function
        acquisition = qNegIntegratedPosteriorVariance(
            model=model,
            mc_points=integration_x,
        )

        # optimize acqf to get the next batch of selected candidate points
        candidates, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=unit_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            gen_candidates=gen_candidates_torch,
            options={
                "maxiter": ACQ_OPT_STEPS,
                "lr": ACQ_OPT_LR,
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
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=problem.dim),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    posterior = model.posterior(rmse_x)
    rmse = torch.sqrt(torch.mean((posterior.mean - rmse_y_true) ** 2)).item()
    rmse_trace.append(rmse)

    print("\nResult")
    print(f"final rmse: {rmse: .4f}")
    print(f"number of observations: {len(train_y)}")

    rmse_steps = [N_INITIAL_POINTS + i * BATCH_SIZE for i in range(len(rmse_trace))]

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

    axs[0].plot(rmse_steps, rmse_trace, marker=".", color="dodgerblue", label="RMSE")
    axs[0].set_xlabel("number of observations")
    axs[0].set_ylabel("RMSE")
    axs[0].legend()

    axs[1].scatter(
        train_x[:N_INITIAL_POINTS, 0],
        train_x[:N_INITIAL_POINTS, 1],
        color="gray",
        label="initial",
    )
    axs[1].scatter(
        train_x[N_INITIAL_POINTS:, 0],
        train_x[N_INITIAL_POINTS:, 1],
        color="black",
        label="exploration",
    )
    axs[1].set_xlabel("x1 normalized")
    axs[1].set_ylabel("x2 normalized")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig("error_reduction.png", dpi=150)
    print("\nSaved exploration plot to error_reduction.png")


if __name__ == "__main__":
    main()
