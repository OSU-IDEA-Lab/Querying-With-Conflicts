# Fixed-domain MI bucketization

This experiment measures the utility/runtime tradeoff of bucketizing MI without
changing the evaluation problem. Amazon, PriceRunner, and Flights use a fixed
sample of 1,024 ordered groups; COMPAS and Census use their complete 65- and
16-group domains. Numeric and categorical ordering is identical to the scaled
conflict-learning experiment.

For every seeded bias run, the full groups and rank scale remain fixed. MI runs
on `B = 1, 2, 4, ..., m` nested buckets. Each matched bucket pair is lifted to
one-to-one pairs of its underlying groups. Quadratic utility is then evaluated
on those full-domain pairs using the true fine-group biases and fixed `z=m`.
Utility recovery is the resulting gain divided by MI's unbucketized `B=m` gain
for the same run. Raw files retain this per-run ratio. Summaries and plots use
the conflict-learning convention: mean bucketized gain divided by mean
unbucketized gain across runs. Thus recovery is exactly one at the full-domain
endpoint, avoids instability from individual near-zero denominators, and does
not inherit the old `B^3` change of units.

The paper experiment uses a single joint bias function of three designated
attributes. Each observed three-way bias value receives an independent seeded
draw in `[-lambda*z, 0]`; the prepared data estimate `P(A_b | e)`, and the
fixed-domain groups receive the corresponding conditional expected biases.
The default is `lambda=0.25`. Matching is timed for all five realizations.

MI uses the uniform rank prior by default.  A centered discrete Gaussian rank
prior is also available.  For variance `s2`, its mass function on
`{1, ..., z}` is proportional to
`exp(-(x - (z+1)/2)^2 / (2*s2))`.  The implementation computes its pairwise
difference distribution, posterior cell means, influence thresholds, and
quadratic gains numerically in log space.  This keeps concentrated Gaussian
tails stable and leaves the existing uniform closed forms unchanged.

```bash
python -m experiments.mi_bucketization.run_all
```

For example, the following runs MI with Gaussian kernel variance parameter `64`
(in squared-rank units):

```bash
python -m experiments.mi_bucketization.run_all \
  --prior-model gaussian \
  --prior-variance 64
```

When no output directory is supplied, Gaussian results are written to a
variance-specific directory such as
`results/mi_fixed_bucketization_gaussian_var_64/`, so they do not overwrite the
uniform results. The output tables include `prior_model`, `prior_variance`,
`prior_mean`, and `prior_actual_variance` columns. Because the prior is
truncated to `[z]`, the realized variance in `prior_actual_variance` can be
smaller than its Gaussian kernel parameter.

The paper's Gaussian sensitivity sweep uses the same seeded rule as the other
MI experiments: for run `r`, draw `s` uniformly from `(0,1)` and use standard
deviation `s*z`. The same normalized draw is shared across datasets, bias
strengths, and bucket resolutions, and plots average the five repetitions. To
produce the four-lambda utility-recovery data first, run:

```bash
python -m experiments.mi_bucketization.run_all \
  --bias-model joint \
  --lambdas 0.1,0.25,0.5,1 \
  --prior-model gaussian-random \
  --skip-general-timing
```

Then benchmark the missing exact-matching calls and regenerate the plots:

```bash
python -m experiments.mi_bucketization.benchmark_missing_timings \
  --output-dir results/mi_fixed_bucketization_gaussian_random_lambda_sweep
python -m experiments.mi_bucketization.plot \
  results/mi_fixed_bucketization_gaussian_random_lambda_sweep
```

Outputs are written to
`results/mi_fixed_bucketization_gaussian_random_lambda_sweep/`.

The no-order control uses the identical seeded draw values without sorting,
shifted into `[-lambda*z, 0]`.  It is written to a separate result directory so
the original smooth-bias result is retained:

```bash
python -m experiments.mi_bucketization.run_all \
  --bias-model joint \
  --output-dir results/mi_fixed_bucketization_iid_bias
```

The four-strength sweep used by the common experimental protocol is:

```bash
python -m experiments.mi_bucketization.run_all \
  --bias-model iid \
  --lambdas 0.1,0.25,0.5,1 \
  --output-dir results/mi_fixed_bucketization_iid_lambda_sweep
```

The sweep keeps all per-strength summaries. Its compact paper plots give each
strength equal weight when aggregating utility recovery and use the median
matching time across strengths.

Outputs are written to `results/mi_fixed_bucketization/`.
