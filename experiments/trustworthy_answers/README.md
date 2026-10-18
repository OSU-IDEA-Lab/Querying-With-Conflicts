# Trustworthy-answer detection scalability

This experiment implements the range-based detector from the closed-form
trustworthiness theorem and its optimized symmetric-prior version. It retains
the paper's original scalability setup: the returned result has `k=80` tuples,
while `z` ranges from 0.5% to 10% of each dataset. When a dataset fraction has
fewer than 80 rows, `z` is set to 80 so that `k` remains fixed.

For every repetition and fraction, the experiment uniformly samples `z` rows
and draws one iid intent from the selected rank prior. The source orders these
rows by the intended group rank adjusted by the row's bias and returns its
first 80 rows; this ordered result is `beta(I)`. Detection is the only timed
operation. Data loading, sampling, distribution estimation, construction of
`beta(I)`, and plotting are excluded.

The distinct-attribute condition uses one joint bias function over three
attributes outside the intent attributes.  Every observed joint value receives
one independently drawn bias, and the detector uses the empirical conditional
distribution of those joint values to compute each intent group's expected
bias.  In the coinciding condition, the joint bias attribute is the complete
three-attribute intent group, so tuple and expected group biases agree and the
optimized detector applies.  We use four bias strengths and five repetitions.
Gaussian repetition `r` draws the same normalized width `s` used by the other
experiments and sets `sigma=s*z`.

Run from the repository root:

```bash
python -m experiments.trustworthy_answers.run_scalability
```

Raw observations and summaries are written under
`results/trustworthy_answers_joint`; the four paper panels are in its
`paper_plots` directory.
