# Fixed-domain conflict learning

This experiment adapts the supplied conflict-learning reference implementation
to every current dataset while separating the true domain from the learner's
bucket resolution.

## Domains

| Dataset | Attribute | Full distinct values | Fixed experiment domain |
| --- | --- | ---: | ---: |
| Amazon | price | 29,961 | 1,024 sampled values |
| PriceRunner | product model | 12,849 | 1,024 sampled values |
| COMPAS | age | 65 | all 65 values |
| Flights | price | 12,157 | 1,024 sampled values |
| Census | education level | 16 | all 16 values |

The sample is selected once with a stable dataset-specific seed, sorted, and
reused in every run. Thus the underlying groups and their biases never change.
Threshold learning and matching are consequently computed once per bucket
level; the five runs independently resample intents and within-bucket pair
orientations.

For each dataset, `m=z` remains fixed. Only the number of nested learning
buckets varies:

```text
B = 1, 2, 4, 8, ..., m
```

`B=m` is the no-additional-bucketization case. At `B=1`, every true group is
represented by one learning bucket.

## Bias and methods

Numeric values are min-max normalized to `[0,z-1]`. Categorical product models
receive their stable lexicographic rank on the same interval. The true bias is

```text
b(e) = alpha * normalized_attribute(e)
```

The default is `alpha=0.05`. Matching uses the thresholds learned for bucket
representatives (and threshold one within a bucket). Before the final query is
evaluated, every selected pair is probed again: its returned point-level
threshold replaces the learned threshold, and a pair with no influential
threshold is dropped. This is the equilibrium-certification step used by the
utility theorem.

The recorded metrics are quadratic utility gain and raw utility, Kendall tau-b,
probe-query complexity, relative-bias error and bound coverage, matched pairs,
and learning-plus-matching runtime. `learning_probe_queries` records the bucket
representative phase, `certification_probe_queries` records the additional
matched-pair phase, and `probe_queries` is their sum. Probe queries are
conflict-learning probes; they are not SQL WHERE queries.

The output includes combined plots against both learning-bucket count and bias
uncertainty `epsilon`. Here `epsilon` is twice the largest deviation between a
true group bias and its learning-bucket representative; the error guarantee is
reported separately as `1/2 + epsilon`. The epsilon axis uses a symmetric-log
scale so that the exact, no-additional-bucketization endpoint `epsilon=0`
remains visible.

Bucket-width plots define width as the maximum number of fixed true groups in
any learning bucket, equivalently `ceil(m / B)` for these nested dyadic levels.
This group-count definition is comparable across numeric and categorical
attributes. Width `1` is the exact, no-additional-bucketization case; larger
widths are coarser.

The primary utility figures report the equilibrium-certified query. Legacy
`dhat` and robust columns remain in the scaled comparison outputs only as
diagnostics for the pre-certification strategies.

## Run

```bash
python -m experiments.conflict_learning.run_all
```

A small smoke run is:

```bash
python -m experiments.conflict_learning.run_all \
  --datasets compas \
  --max-groups 16 \
  --runs 1 \
  --intents 20
```

## Scaled exact-versus-fast comparison

`run_comparison.py` is the fresh scaled experiment. For Amazon, PriceRunner,
and Flights it fixes `m=z=2048`; COMPAS and Census retain all 65 and 16 groups.
It sweeps `lambda={0.1,0.25,0.5,1}` and averages five fresh bias realizations.
For every realization, it draws `m` values independently from
`U[-lambda*m,0]`, sorts them, and assigns them in attribute order. Numeric and
ordinal attributes use their natural order; PriceRunner uses a deterministic
character-ngram similarity order. This gives nearby groups similar but not
exactly linear biases.

Conflict learning is performed once at every `k` and supplied unchanged to:

1. learned `dhat` with general maximum-weight matching;
2. learned `dhat` with the efficient sort-and-pair influential strategy; and
3. learned robust `z-1` with general maximum-weight matching.

After either matching method selects pairs, each pair is re-probed and only the
returned influential threshold is placed in the final query. Thus exact and
fast have identical learning probes and global bias error, but can have
different certification costs because they can select different pairs. The
reported matching-only times measure only the downstream influential-strategy
construction. General exact matching has a 600-second per-call limit; a timed
out point is stored as missing while the fast result and all other levels are
retained. The known-bias oracle pairs adjacent groups in true sorted bias order,
which is exact because Algorithm-2 weights are nonincreasing with one-dimensional
bias distance.

Run the comparison with:

```bash
python -m experiments.conflict_learning.run_comparison
```

The paper comparison uses a single joint bias function of three designated
attributes. Every observed three-way value receives an independent bias draw,
and the prepared data estimate the conditional expected group bias
`E[b(A_b) | e]`. Results and paper figures are written to
`results/conflict_learning_joint_uniform/` when that output directory is
selected. The default maximum learning resolution is 1024 buckets. To run a
single strength while preserving the fixed domain and resolution cap:

```bash
python -m experiments.conflict_learning.run_comparison \
  --bias-model joint --lambdas 1 --max-buckets 1024 \
  --output-dir results/conflict_learning_joint_lambda_1 --no-plots
```

The same certified comparison can be run under centered discrete Gaussian IID
rank priors. In each repetition, the paper draws a normalized width
`s` uniformly from `(0,1)`, uses standard deviation `s*z` for each dataset,
and holds `s` fixed across datasets, bias strengths, and learning-bucket
resolutions. Reported curves average the five
repetitions. The selected prior is used both to construct the influential
thresholds and to evaluate expected user utility:

```bash
python -m experiments.conflict_learning.run_comparison \
  --bias-model joint --prior-model gaussian-random
```

The paper writes these outputs to
`results/conflict_learning_joint_gaussian/`; the uniform-prior outputs are not
overwritten. A fixed-width Gaussian remains available as a controlled
sensitivity check through `--prior-model gaussian` and
`--gaussian-std-fraction`.
