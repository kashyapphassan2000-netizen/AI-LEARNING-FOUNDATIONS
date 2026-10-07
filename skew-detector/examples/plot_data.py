"""Plot train (blue) vs serve (red) histograms for every column -> docs/distributions.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tr, se = pd.read_csv("data/train.csv"), pd.read_csv("data/serve.csv")
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.ravel(), tr.columns):
    lo, hi = np.percentile(np.r_[tr[col], se[col]], [0.5, 99.5])
    bins = np.arange(-0.5, 5.5) if col == "region" else (
        np.array([-0.5, 0.5, 1.5]) if col == "default" else np.linspace(lo, hi, 40))
    ax.hist(tr[col], bins=bins, density=True, alpha=0.55, color="#1f77b4", label="train")
    ax.hist(se[col], bins=bins, density=True, alpha=0.55, color="#d62728", label="serve")
    ax.set_title(col)
axes[0, 0].legend()
fig.suptitle("train (blue) vs serve (red): known drift in age, income, debt_ratio, default")
fig.tight_layout()
fig.savefig("docs/distributions.png", dpi=110)
print("wrote docs/distributions.png")
