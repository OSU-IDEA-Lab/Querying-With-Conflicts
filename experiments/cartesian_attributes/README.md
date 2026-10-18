# Cartesian attribute scalability and utility

This is the number-of-attributes version of `experiments/bucket_attributes`.
It uses the same Algorithm 1, Fast Algorithm 1, Random credible-pairs baseline,
ranking utility, and information-gain user utility. The x-axis is the number of
ranking attributes (`1, 2, 3`), not the number of groups.

## Fixed nested schemes

| Dataset | Attribute order | Individual cardinalities | Observed groups by prefix | Possible groups by prefix |
|---|---|---:|---:|---:|
| Amazon | whole stars × price decile × bought last month | 6 × 10 × 30 | 6, 60, 885 | 6, 60, 1,800 |
| PriceRunner | brand (top 10) × category × merchant (top 10 + Other) | 10 × 10 × 11 | 10, 58, 451 | 10, 100, 1,100 |
| COMPAS | prior-count bucket × age bucket × violence decile | 2 × 3 × 10 | 2, 6, 57 | 2, 6, 60 |
| Flights | price category × airline × days-left category | 5 × 6 × 13 | 5, 26, 327 | 5, 30, 390 |
| Census | sex × race × education number | 2 × 5 × 16 | 2, 10, 154 | 2, 10, 160 |

The algorithm domain contains observed tuple keys, matching SQL `GROUP BY`.
Impossible/empty combinations are not ranked, but the possible-product size and
observed density are recorded in every output.

## Complexity notation

`m` is the number of observed Cartesian tuple groups processed by a strategy.
Every attribute-count runtime plot displays `m` below its x tick, and
`runtime_vs_m.{png,pdf}` plots runtime directly against `m`.

The paper's heuristic bound, `O(m log m + m z)`, assumes that computing each
`delta*` may scan a length-`z` threshold grid. This implementation precomputes
that grid and uses binary search, so its implemented cost is closer to
`O(m log m + m log z + z)`. In the present experiment `z=m`; an independent
empirical test of the `m z` term would need to vary `z` while holding `m` fixed
or use the theorem's linear-scan `delta*` routine.

## Bias construction

For each run seed, every bucket value of every attribute receives a fixed seeded
`U[0,1)` component. A tuple's component value is the arithmetic mean of its
attribute components. For lambda `λ` and `z` observed tuple groups, its final bias
is `-λ z × mean(component biases)`, matching the scale used by the original bucket
experiment. The component mapping is reused unchanged when later attributes are
added.

## Queries and metrics

Each dataset receives 100 seeded WHERE queries built with the same process as the
bucket experiment: one to three predicates sampled from a real row, categorical
equality, numeric ranges, random ASC/DESC direction, and at least three returned
groups for every attribute prefix. COMPAS uses two because its first prepared
attribute has only two buckets. The exact same 100 predicates are evaluated at all
three x-axis points.

- Ranking utility is Kendall tau-b over the tuple groups returned by each query.
- User utility is the expected-information gain of selected credible pairs.
- The combined IQ utility plot uses a logarithmic y-axis because unnormalized
  information gain grows rapidly with the Cartesian group count; a linear axis
  visually collapses the one- and two-attribute values.
- Runtime measures the algorithmic pair-selection step. The number of observed
  Cartesian groups is denoted by `m`; in this implementation the posterior grid
  parameter is `z=m`. Workload construction and query evaluation are excluded.

Five methods are run and plotted: Algorithm 1, Fast Algorithm 1, Random credible
pairs (a multi-pair random matching), Single best pair, and Single random pair.
The latter two always select at most one credible pair. Single random pair scans
seeded randomly ordered candidates one at a time and stops at the first credible
pair; it does not enumerate the full pair set. Babbling and full revelation are
not included.

## Random Gaussian-prior MI experiment

The MI query-scalability experiment can replace the uniform rank prior with a
centered discrete Gaussian. Its complete prior-sampling rule is:

> For each run, we sample the Gaussian standard deviation uniformly between
> zero and the rank-domain size, using the same seeded normalized draw for every
> dataset.

Thus, if the shared draw is `s`, a prefix with rank domain `z` uses standard
deviation `s*z` and kernel variance `(s*z)^2`. The raw outputs record the draw,
kernel variance, and actual variance after truncation. The uniform prior remains
the default, and Gaussian results use a separate output directory.

The following runs MI only on all five datasets and creates a combined
three-panel query-scalability plot:

```bash
python -m experiments.cartesian_attributes.run_all \
  --prior-model gaussian-random \
  --strategies alg1
```

The combined figure is written to
`results/cartesian_attributes_gaussian_random/query_paper_plots/mi_query_scalability_combined.{png,pdf}`.

Run the full experiment from the repository root:

```bash
python -m experiments.cartesian_attributes.run_all
```

## Distribution-aware intent experiment

The distribution-aware variant leaves the existing WHERE/ASC/DESC workload and
its outputs unchanged. For each repetition it uses the full prepared dataset
to estimate the intent-group probabilities and conditional expected bias, then
runs exact influential matching under uniform and centered Gaussian iid rank
priors. The x-axis counts only intent attributes. All three designated bias
attributes are held outside the Cartesian intent tuple, and the implementation
rejects a bias source column that appears among the intent attributes.

Each dataset has three bias attributes: product category, best-seller status,
and review-count band (Amazon); brand, merchant, and category (PriceRunner);
race, sex, and charge degree (COMPAS); travel class, source city, and number of
stops (Flights); and age group, work class, and occupation (Census). They form
one joint bias attribute $A_b$: each observed three-way value receives one
independent bias draw. For an intent group $e$, the experiment uses

```text
P(e) = count(e) / sample size
b(e) = sum_a beta(a) P(A_b=a | e).
```

The default estimation sample is the full prepared dataset. Consequently, the
active Cartesian intent domain is exactly the same observed-group domain used
by the existing same-attribute experiment. Each observed joint bias value
receives a seeded source bias. At each prefix the source-bias scale is
$\beta(a)=\lambda zU_a$, matching the existing experiment's $\lambda z$ scale,
and the source response subtracts $b(e)$ from its posterior mean position.
Uniform has no width parameter. For Gaussian, each of the five repetitions
draws $s\sim\mathrm{Uniform}(0,1)$ and uses $\sigma=sz$; that normalized width
is shared across datasets, intent-prefix sizes, and bias strengths in the
repetition. Each plotted point averages four bias strengths and five
repetitions for the single joint three-attribute bias function.

For direct comparability with the existing same-attribute query experiment, the
workload draws 100 iid rank-vector intents from each prior.  Each exact intent
is translated into the conjunction selected by the matching, and ranking
utility is Kendall $\tau_b$ between that intent and the source's bias-adjusted
posterior ranking.  We report gain over the no-query (babbling) response.  The
exact expected information gain remains a separate user-utility metric, and
the same sampled intents provide its Monte Carlo check.

Run all datasets and create the runtime, expected-user-utility, and
ranking-utility plots with:

```bash
python -m experiments.cartesian_attributes.run_distributional
```

For the main query comparison, the four time/user-utility panels are grouped in
`paper_plots/query_time_and_user_utility.{pdf,png}`: priors form the two columns,
with matching time above expected information gain. The four individual paper
panels use the names `mi_execution_time_{uniform,gaussian}` and
`mi_user_utility_{uniform,gaussian}`.

The independent output directory is
`results/cartesian_attributes_distributional_joint_bias/`. A smaller sampled
run can be requested explicitly; for example:

```bash
python -m experiments.cartesian_attributes.run_distributional \
  --datasets compas \
  --runs 1 \
  --lambdas 0.25 \
  --sample-fraction 0.05 \
  --intents 100 \
  --no-plots
```

The original WHERE-workload smoke run is available with:

```bash
python -m experiments.cartesian_attributes.run_all \
  --datasets compas \
  --queries 10 \
  --runs 1 \
  --lambdas 0.5 \
  --no-plots
```
