# Distribution-aware MI with distinct bias attributes

This directory contains the full prepared-domain experiment used by the four
paper panels in `paper_plots/`.

- Intent prefixes contain one, two, or three intent attributes.
- Each dataset uses three bias attributes whose source columns do not occur in
  its intent tuple.
- Priors are uniform and centered discrete Gaussian.
- Uniform has no width parameter. In Gaussian repetition `r`, a seeded draw
  `s_r ~ Uniform(0,1)` sets `sigma = s_r z`; the same `s_r` is shared across
  datasets, intent prefixes, bias attributes, and bias strengths.
- The five normalized Gaussian widths are `0.520868`, `0.539996`, `0.903151`,
  `0.060486`, and `0.917502`.
- Bias strengths are `0.1`, `0.25`, `0.5`, and `1.0`.
- `P(e)` and `E[bias(A) | e]` use 100% of each prepared dataset.
- Every plotted point averages three bias attributes, four bias strengths, and
  five repetitions (60 measurements).

`all_results.csv` is the 1,800-row raw result, `summary.csv` retains separate
means for each bias attribute, and `plot_summary.csv` contains the 30 values
plotted in the time and expected-user-utility panels.
