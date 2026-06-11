# BOPlayground

A small BoTorch playground for teaching the basic Bayesian optimization loop.

Run target optimization with:

```bash
uv run optimize_target.py
```

Run space exploration with:

```bash
uv run explore_space.py
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
