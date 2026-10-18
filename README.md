# Querying-COI Experiments

Experiments used in the paper's main text and appendix.

## Setup

```bash
python -m pip install -r requirements.txt
```

## 1. Attribute scalability

```bash
python -m experiments.cartesian_attributes.run_all

python -m experiments.cartesian_attributes.run_all \
  --prior-model gaussian-random \
  --strategies alg1

python -m experiments.cartesian_attributes.run_distributional
```

## 2. Trustworthy-answer detection

```bash
python -m experiments.trustworthy_answers.run_scalability
```

## 3. MI bucketization

```bash
python -m experiments.mi_bucketization.run_all
```

## 4. Conflict learning

```bash
python -m experiments.conflict_learning.run_all
```

## Tests

```bash
python -m pytest
```

Results and generated figures are in `results/`.
The Amazon experiments require a local `data/real/amazon_products.csv` file.
