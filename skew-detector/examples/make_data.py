"""Generate a realistic train.csv and serve.csv pair with KNOWN drift injected,
plus a control pair (train_clean.csv / serve_clean.csv) with no drift at all.

This makes the verification rigorous: we know exactly which columns drifted
and which didn't, so we can check that skew-detect catches the real drift and
doesn't false-positive on the unchanged columns.

Schema: a small loan-application style dataset.
  age           (numeric) -- drift: serve population is older
  income        (numeric) -- drift: serve income inflated 30%
  credit_score  (numeric) -- no drift
  debt_ratio    (numeric) -- mild drift: small shift up
  region        (numeric) -- no drift
  default       (label)   -- prevalence shifts upward
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_TRAIN, N_SERVE = 8000, 4000


def _frame(age, income, credit_score, debt_ratio, region, offset, n):
    risk = (-0.02 * (credit_score - 700) + 5 * debt_ratio
            - 0.00001 * income + 0.01 * (age - 38))
    default = (rng.random(n) < 1 / (1 + np.exp(-risk + offset))).astype(int)
    return pd.DataFrame({"age": age, "income": income, "credit_score": credit_score,
                         "debt_ratio": debt_ratio, "region": region, "default": default})


def make_train(n=N_TRAIN):
    age = rng.normal(38, 10, n).clip(18, 80)
    income = rng.lognormal(mean=10.7, sigma=0.5, size=n)
    credit_score = rng.normal(700, 60, n).clip(300, 850)
    debt_ratio = rng.beta(2, 5, n)
    region = rng.integers(0, 5, n)
    return _frame(age, income, credit_score, debt_ratio, region, 1.0, n)


def make_serve(n=N_SERVE):
    # KNOWN drift, deliberately injected:
    age = rng.normal(45, 10, n).clip(18, 80)                       # shifted older
    income = rng.lognormal(mean=10.7, sigma=0.5, size=n) * 1.30    # 30% inflation
    credit_score = rng.normal(700, 60, n).clip(300, 850)           # NO drift
    debt_ratio = rng.beta(2.2, 5, n)                               # mild shift
    region = rng.integers(0, 5, n)                                 # NO drift
    return _frame(age, income, credit_score, debt_ratio, region, 0.2, n)


def main():
    tr, se = make_train(), make_serve()
    tr.to_csv("data/train.csv", index=False)
    se.to_csv("data/serve.csv", index=False)
    print(f"wrote data/train.csv {tr.shape} default rate {tr['default'].mean():.3f}")
    print(f"wrote data/serve.csv {se.shape} default rate {se['default'].mean():.3f}")

    # control: both sides drawn from the training distribution
    trc, sec = make_train(N_TRAIN), make_train(N_SERVE)
    trc.to_csv("data/train_clean.csv", index=False)
    sec.to_csv("data/serve_clean.csv", index=False)
    print(f"wrote data/train_clean.csv {trc.shape} default rate {trc['default'].mean():.3f}")
    print(f"wrote data/serve_clean.csv {sec.shape} default rate {sec['default'].mean():.3f}")

    print("\nKNOWN drifts injected:")
    print("  age          : SHOULD detect (mean 38 -> 45)")
    print("  income       : SHOULD detect (inflated 30%)")
    print("  credit_score : should NOT detect (unchanged)")
    print("  debt_ratio   : SHOULD detect (mild shift)")
    print("  region       : should NOT detect (unchanged)")
    print("  default      : SHOULD detect (prevalence shift)")


if __name__ == "__main__":
    main()
