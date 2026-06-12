import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import torch
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.generation.gen import gen_candidates_torch
from botorch.models import SingleTaskGP
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
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    n_initial_points: int
    n_batches: int
    batch_size: int
    n_eval_points: int
    nipv_n_integration_points: int
    hipe_n_integration_points: int
    num_restarts: int
    raw_samples: int
    acq_opt_steps: int
    acq_opt_lr: float
    acq_batch_limit: int
    init_batch_limit: int
    crps_samples: int
    hipe_nuts_samples: int
    hipe_nuts_warmup: int
    hipe_nuts_thinning: int
    hipe_mc_samples: int
    hipe_beta_tuning_samples: int


SMOKE_CONFIG = BenchmarkConfig(
    name="smoke",
    n_initial_points=5,
    n_batches=1,
    batch_size=2,
    n_eval_points=64,
    nipv_n_integration_points=32,
    hipe_n_integration_points=32,
    num_restarts=2,
    raw_samples=32,
    acq_opt_steps=20,
    acq_opt_lr=0.05,
    acq_batch_limit=1,
    init_batch_limit=8,
    crps_samples=32,
    hipe_nuts_samples=32,
    hipe_nuts_warmup=32,
    hipe_nuts_thinning=1,
    hipe_mc_samples=16,
    hipe_beta_tuning_samples=8,
)

PRODUCTION_CONFIG = BenchmarkConfig(
    name="production",
    n_initial_points=8,
    n_batches=15,
    batch_size=3,
    n_eval_points=512,
    nipv_n_integration_points=128,
    hipe_n_integration_points=256,
    num_restarts=4,
    raw_samples=64,
    acq_opt_steps=80,
    acq_opt_lr=0.03,
    acq_batch_limit=1,
    init_batch_limit=4,
    crps_samples=64,
    hipe_nuts_samples=128,
    hipe_nuts_warmup=512,
    hipe_nuts_thinning=4,
    hipe_mc_samples=64,
    hipe_beta_tuning_samples=32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Sobol, NIPV + SingleTaskGP, and HIPE + fully Bayesian GP."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke-test benchmark instead of the default benchmark.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds to benchmark. Each seed uses shared starting data.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the benchmark plot from an existing CSV without rerunning.",
    )
    return parser.parse_args()


def build_single_task_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    return SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    )


def build_fully_bayesian_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    unit_bounds: torch.Tensor,
) -> FullyBayesianSingleTaskGP:
    return FullyBayesianSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=train_x.shape[-1], bounds=unit_bounds),
        outcome_transform=Standardize(m=1),
    )


def empirical_crps(samples: torch.Tensor, y_true: torch.Tensor) -> float:
    samples = samples.squeeze(-1).reshape(-1, y_true.shape[0])
    y_true = y_true.squeeze(-1)
    sorted_samples = samples.sort(dim=0).values
    n_samples = samples.shape[0]
    coeffs = (
        2 * torch.arange(n_samples, dtype=samples.dtype, device=samples.device)
        - n_samples
        + 1
    ).unsqueeze(-1)
    mean_abs_error = (samples - y_true).abs().mean(dim=0)
    half_pairwise_abs = (coeffs * sorted_samples).sum(dim=0) / n_samples**2
    return (mean_abs_error - half_pairwise_abs).mean().item()


def evaluate_model(
    model,
    eval_x: torch.Tensor,
    eval_y_true: torch.Tensor,
    sample_shape: torch.Size,
    sample_seed: int,
) -> tuple[float, float]:
    posterior = model.posterior(eval_x)
    mean = getattr(posterior, "mixture_mean", posterior.mean)
    if mean.ndim > eval_y_true.ndim:
        mean = mean.mean(dim=0)
    rmse = torch.sqrt(torch.mean((mean - eval_y_true) ** 2)).item()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sample_seed)
        samples = posterior.rsample(sample_shape)
    crps = empirical_crps(samples, eval_y_true)
    return rmse, crps


def optimize_nipv_candidates(
    model: SingleTaskGP,
    unit_bounds: torch.Tensor,
    integration_x: torch.Tensor,
    config: BenchmarkConfig,
) -> torch.Tensor:
    acquisition = qNegIntegratedPosteriorVariance(model=model, mc_points=integration_x)
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
    return candidates.detach()


def optimize_hipe_candidates(
    model: FullyBayesianSingleTaskGP,
    unit_bounds: torch.Tensor,
    integration_x: torch.Tensor,
    config: BenchmarkConfig,
    seed: int,
) -> torch.Tensor:
    acquisition = qHyperparameterInformedPredictiveExploration(
        model=model,
        mc_points=integration_x,
        bounds=unit_bounds,
        sampler=SobolQMCNormalSampler(
            sample_shape=torch.Size([config.hipe_mc_samples]),
            seed=seed,
        ),
        beta_tuning_samples=config.hipe_beta_tuning_samples,
    )
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
    return candidates.detach()


def draw_sobol_candidates(
    unit_bounds: torch.Tensor,
    config: BenchmarkConfig,
    seed: int,
) -> torch.Tensor:
    return draw_sobol_samples(
        bounds=unit_bounds,
        n=config.batch_size,
        q=1,
        seed=seed,
    ).squeeze(1)


def run_stack(
    stack: str,
    seed: int,
    config: BenchmarkConfig,
    problem: Branin,
    unit_bounds: torch.Tensor,
    native_bounds: torch.Tensor,
    initial_x: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y_true: torch.Tensor,
    nipv_integration_x: torch.Tensor,
    hipe_integration_x: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    train_x = initial_x.clone()
    train_y = problem(unnormalize(train_x, bounds=native_bounds)).unsqueeze(-1)
    records = []
    started_at = time.perf_counter()

    for batch in range(config.n_batches + 1):
        if stack in {"nipv", "sobol"}:
            model = build_single_task_model(train_x, train_y)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        elif stack == "hipe":
            model = build_fully_bayesian_model(train_x, train_y, unit_bounds)
            fit_fully_bayesian_model_nuts(
                model,
                num_samples=config.hipe_nuts_samples,
                warmup_steps=config.hipe_nuts_warmup,
                thinning=config.hipe_nuts_thinning,
                disable_progbar=True,
                seed=seed + batch,
            )
        else:
            raise ValueError(f"Unknown stack: {stack}")

        rmse, crps = evaluate_model(
            model=model,
            eval_x=eval_x,
            eval_y_true=eval_y_true,
            sample_shape=torch.Size([config.crps_samples]),
            sample_seed=seed + batch,
        )
        records.append(
            {
                "seed": seed,
                "stack": stack,
                "batch": batch,
                "n_observations": len(train_y),
                "rmse": rmse,
                "crps": crps,
                "best_y": train_y.max().item(),
                "elapsed_sec": time.perf_counter() - started_at,
            }
        )

        if batch == config.n_batches:
            break

        if stack == "nipv":
            candidates = optimize_nipv_candidates(
                model=model,
                unit_bounds=unit_bounds,
                integration_x=nipv_integration_x,
                config=config,
            )
        elif stack == "hipe":
            candidates = optimize_hipe_candidates(
                model=model,
                unit_bounds=unit_bounds,
                integration_x=hipe_integration_x,
                config=config,
                seed=seed + 1000 + batch,
            )
        else:
            candidates = draw_sobol_candidates(
                unit_bounds=unit_bounds,
                config=config,
                seed=seed + 2000 + batch,
            )

        candidate_y = problem(unnormalize(candidates, bounds=native_bounds)).unsqueeze(
            -1
        )
        train_x = torch.cat([train_x, candidates])
        train_y = torch.cat([train_y, candidate_y])

        latest = records[-1]
        print(
            f"{stack:4s} seed={seed} batch={batch:02d} | "
            f"rmse={latest['rmse']:9.4f} | crps={latest['crps']:9.4f} | "
            f"best_y={latest['best_y']:9.4f}"
        )

    return records


def write_csv(records: list[dict[str, float | int | str]], path: str) -> None:
    fieldnames = [
        "seed",
        "stack",
        "batch",
        "n_observations",
        "rmse",
        "crps",
        "best_y",
        "elapsed_sec",
    ]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_csv(path: str) -> list[dict[str, float | int | str]]:
    records = []
    with open(path) as csvfile:
        for record in csv.DictReader(csvfile):
            records.append(
                {
                    "seed": int(record["seed"]),
                    "stack": record["stack"],
                    "batch": int(record["batch"]),
                    "n_observations": int(record["n_observations"]),
                    "rmse": float(record["rmse"]),
                    "crps": float(record["crps"]),
                    "best_y": float(record["best_y"]),
                    "elapsed_sec": float(record["elapsed_sec"]),
                }
            )
    return records


def smart_upper_ylim(records: list[dict[str, float | int | str]], metric: str) -> float:
    values = torch.tensor([float(record[metric]) for record in records])
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return 1.0
    q95 = finite_values.quantile(0.95).item()
    q98 = finite_values.quantile(0.98).item()
    max_value = finite_values.max().item()
    upper = max(q98, q95 * 1.15)
    if max_value > upper * 1.5:
        return upper * 1.05
    return max_value * 1.05


def plot_results(records: list[dict[str, float | int | str]], path: str) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
    stacks = ["sobol", "nipv", "hipe"]
    colors = {"sobol": "gray", "nipv": "dodgerblue", "hipe": "darkorange"}
    rmse_ylim = smart_upper_ylim(records, "rmse")
    crps_ylim = smart_upper_ylim(records, "crps")

    def hide_outliers(values: list[float], upper: float) -> list[float]:
        return [value if value <= upper else float("nan") for value in values]

    for stack in stacks:
        stack_records = [record for record in records if record["stack"] == stack]
        seeds = sorted({int(record["seed"]) for record in stack_records})
        batches = sorted({int(record["batch"]) for record in stack_records})

        for seed in seeds:
            seed_records = [
                record for record in stack_records if int(record["seed"]) == seed
            ]
            seed_records = sorted(seed_records, key=lambda record: int(record["batch"]))
            n_observations = [int(record["n_observations"]) for record in seed_records]
            alpha = 0.35 if len(seeds) > 1 else 0.75
            axs[0].plot(
                n_observations,
                hide_outliers(
                    [float(record["rmse"]) for record in seed_records], rmse_ylim
                ),
                color=colors[stack],
                alpha=alpha,
                linewidth=1,
            )
            axs[1].plot(
                n_observations,
                hide_outliers(
                    [float(record["crps"]) for record in seed_records], crps_ylim
                ),
                color=colors[stack],
                alpha=alpha,
                linewidth=1,
            )

        rmse_means = []
        crps_means = []
        mean_n_observations = []
        for batch in batches:
            batch_records = [
                record for record in stack_records if int(record["batch"]) == batch
            ]
            rmse_means.append(
                sum(float(record["rmse"]) for record in batch_records)
                / len(batch_records)
            )
            crps_means.append(
                sum(float(record["crps"]) for record in batch_records)
                / len(batch_records)
            )
            mean_n_observations.append(int(batch_records[0]["n_observations"]))

        axs[0].plot(
            mean_n_observations,
            hide_outliers(rmse_means, rmse_ylim),
            marker=".",
            color=colors[stack],
            linewidth=2.5,
            label=f"{stack.upper()} mean",
        )
        axs[1].plot(
            mean_n_observations,
            hide_outliers(crps_means, crps_ylim),
            marker=".",
            color=colors[stack],
            linewidth=2.5,
            label=f"{stack.upper()} mean",
        )

    axs[0].set_xlabel("number of observations")
    axs[0].set_ylabel("RMSE")
    axs[0].set_ylim(bottom=0, top=rmse_ylim)
    axs[0].legend()
    axs[1].set_xlabel("number of observations")
    axs[1].set_ylabel("CRPS")
    axs[1].set_ylim(bottom=0, top=crps_ylim)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def mean_std(
    records: list[dict[str, float | int | str]], metric: str
) -> tuple[float, float]:
    values = torch.tensor([float(record[metric]) for record in records])
    mean = values.mean().item()
    std = values.std(unbiased=True).item() if len(values) > 1 else 0.0
    return mean, std


def main() -> None:
    args = parse_args()
    config = SMOKE_CONFIG if args.smoke else PRODUCTION_CONFIG
    csv_path = f"exploration_benchmark_{config.name}.csv"
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    plot_path = figures_dir / f"exploration_benchmark_{config.name}.png"

    if args.plot_only:
        records = read_csv(csv_path)
        plot_results(records, plot_path)
        print(f"Regenerated {plot_path} from {csv_path}")
        return

    problem = Branin(negate=True)
    native_bounds = problem.bounds
    unit_bounds = normalize(native_bounds, bounds=native_bounds)
    records = []

    print(
        f"Benchmarking Sobol vs NIPV vs HIPE with {config.name} settings "
        f"for seeds {args.seeds}"
    )

    for seed in args.seeds:
        torch.manual_seed(seed)
        initial_x = torch.rand(config.n_initial_points, problem.dim)
        eval_x = draw_sobol_samples(
            bounds=unit_bounds,
            n=config.n_eval_points,
            q=1,
            seed=seed + 10,
        ).squeeze(1)
        eval_y_true = problem(unnormalize(eval_x, bounds=native_bounds)).unsqueeze(-1)
        nipv_integration_x = draw_sobol_samples(
            bounds=unit_bounds,
            n=config.nipv_n_integration_points,
            q=1,
            seed=seed + 20,
        ).squeeze(1)
        hipe_integration_x = draw_sobol_samples(
            bounds=unit_bounds,
            n=config.hipe_n_integration_points,
            q=1,
            seed=seed + 30,
        ).squeeze(1)

        for stack in ["sobol", "nipv", "hipe"]:
            records.extend(
                run_stack(
                    stack=stack,
                    seed=seed,
                    config=config,
                    problem=problem,
                    unit_bounds=unit_bounds,
                    native_bounds=native_bounds,
                    initial_x=initial_x,
                    eval_x=eval_x,
                    eval_y_true=eval_y_true,
                    nipv_integration_x=nipv_integration_x,
                    hipe_integration_x=hipe_integration_x,
                )
            )

    write_csv(records, csv_path)
    plot_results(records, plot_path)

    print("\nFinal")
    for stack in ["sobol", "nipv", "hipe"]:
        final_records = [
            record
            for record in records
            if record["stack"] == stack and record["batch"] == config.n_batches
        ]
        rmse, rmse_std = mean_std(final_records, "rmse")
        crps, crps_std = mean_std(final_records, "crps")
        best_y, best_y_std = mean_std(final_records, "best_y")
        elapsed, elapsed_std = mean_std(final_records, "elapsed_sec")
        print(
            f"{stack.upper():5s} | "
            f"rmse={rmse:9.4f} +/- {rmse_std:7.4f} | "
            f"crps={crps:9.4f} +/- {crps_std:7.4f} | "
            f"best_y={best_y:9.4f} +/- {best_y_std:7.4f} | "
            f"elapsed={elapsed:7.1f}s +/- {elapsed_std:5.1f}s"
        )
    print(f"\nSaved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
