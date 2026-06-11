import matplotlib.pyplot as plt
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions.synthetic import Branin
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(42)
torch.set_default_dtype(torch.double)

N_INITIAL_POINTS = 3
N_BO_STEPS = 15
NUM_RESTARTS = 5  # more restarts makes acquisition optimization more robust
RAW_SAMPLES = 64  # more raw samples gives better optimizer starting points


def main() -> None:

    problem = Branin(negate=True)
    native_bounds = problem.bounds

    # or manually specified, bounds are ordered as [lower, upper] for each dimension
    # native_bounds = torch.tensor([[-5., 0.], [10., 15.]])

    unit_bounds = normalize(native_bounds, bounds=native_bounds)

    train_x = torch.rand(N_INITIAL_POINTS, problem.dim)  # randomly sample unit points
    train_x_native = unnormalize(train_x, bounds=native_bounds)
    train_y = problem(train_x_native).unsqueeze(-1)

    print(f"Optimizing {problem.__class__.__name__} in {problem.dim} dimensions")

    for step in range(1, N_BO_STEPS + 1):
        # specify model
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            input_transform=Normalize(d=problem.dim),  # built in normalization
            outcome_transform=Standardize(m=1),  # built in standardization
        )

        # fit model
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        # specify acquisition function
        acquisition = LogExpectedImprovement(model=model, best_f=train_y.max())

        # optimize acqf to get next selected candidate point
        candidate, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=unit_bounds,
            q=1,  # num candidate points
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        candidate_native = unnormalize(candidate, bounds=native_bounds)
        candidate_y = problem(candidate_native).unsqueeze(-1)

        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, candidate_y])

        best_idx = train_y.argmax()
        best_x_unit = train_x[best_idx]
        best_x_native = unnormalize(best_x_unit, bounds=native_bounds)
        best_y = train_y[best_idx].item()

        print(
            f"step {step:02d} | "
            f"candidate_y={candidate_y.item(): 9.4f} | "
            f"best_y={best_y: 9.4f} | "
        )

    true_optimizers = problem.optimizers
    true_value = problem(true_optimizers).max().item()
    best_x_native = unnormalize(train_x[train_y.argmax()], bounds=native_bounds)

    print("\nResult:")
    print(f"best observed x: {best_x_native.tolist()}")
    print(f"best observed y: {train_y.max().item(): .4f}")
    print(f"known optimum x values: {true_optimizers.tolist()}")
    print(f"known optimum y: {true_value: .4f}")
    print(
        f"best x normalized: {normalize(best_x_native, bounds=native_bounds).tolist()}"
    )

    observed_y_trace = train_y.squeeze(-1)
    best_y_trace = torch.cummax(observed_y_trace, dim=0).values
    steps = range(1, len(observed_y_trace) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(steps, observed_y_trace, color="black", label="observed y")
    ax.plot(steps, best_y_trace, color="dodgerblue", label="best y")
    ax.axvline(
        N_INITIAL_POINTS + 0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="start BO",
    )
    ax.axhline(
        true_value,
        color="dodgerblue",
        linestyle=":",
        linewidth=1,
        label="target",
    )
    ax.set_xlabel("experiment number")
    ax.set_ylabel("objective value")
    ax.legend()
    fig.tight_layout()
    fig.savefig("trace_plot.png", dpi=150)
    print("\nSaved trace plot to trace_plot.png")


if __name__ == "__main__":
    main()
