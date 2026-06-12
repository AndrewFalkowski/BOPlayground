# BOPlayground

A small BoTorch playground for teaching the basic Bayesian optimization loop.

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
for Python environment management. To generate the environment, run:

```bash
uv sync
```

To start a similar project from scratch, run:

```bash
uv init
uv add "botorch[fully-bayesian]" matplotlib
```

Run target optimization with:

```bash
uv run optimize_target.py
```

Run space exploration with:

```bash
uv run explore_space.py
```

Run model-space exploration with HIPE and a fully Bayesian GP with:

```bash
uv run explore_model_space.py
```

Run a benchmark comparing Sobol exploration, NIPV + `SingleTaskGP`, and HIPE +
`FullyBayesianSingleTaskGP` with:

```bash
uv run benchmark_exploration.py
```

For a quick correctness run, use:

```bash
uv run explore_space.py --smoke
uv run explore_model_space.py --smoke
uv run benchmark_exploration.py --smoke
```

If a benchmark has already run and you only want to redraw its plot from the
existing CSV, use:

```bash
uv run benchmark_exploration.py --plot-only
```

The target optimization script uses BoTorch's synthetic Branin problem, samples
and optimizes in the unit cube, maps candidates back to Branin's native bounds
for evaluation, fits a `SingleTaskGP`, and proposes new points with
`LogExpectedImprovement`.

The exploration script uses batches of 3 points and tracks the model RMSE on a
fixed set of synthetic test points. It uses BoTorch's
`qNegIntegratedPosteriorVariance` acquisition function, which is an active
learning / exploration acquisition. Instead of directly trying to maximize the
objective, it selects points that should reduce the model's average posterior
uncertainty across the design space.

The model-space exploration script uses
`qHyperparameterInformedPredictiveExploration` from `botorch_community` with a
`FullyBayesianSingleTaskGP`. This is slower because the model fit samples a
posterior over GP hyperparameters with NUTS, but it lets HIPE account for both
predictive uncertainty and hyperparameter uncertainty. The default settings are
intended for a higher-quality run; pass `--smoke` for a faster smoke test.

The benchmark script starts each stack from the same initial points and evaluates
them on the same fixed test set. It records RMSE, empirical CRPS, best observed
value, and elapsed time to `exploration_benchmark_<mode>.csv`, then writes a
comparison plot to `figures/exploration_benchmark_<mode>.png`. When run with
multiple seeds, the plot shows individual seed traces with transparency and a
heavier mean trace for each stack.
