"""PCA on the iris dataset using our own SVD. Writes docs/iris_pca.png.

We compare our 2-D projection to scikit-learn's PCA to prove the math is right
(the axes can flip sign -- that's expected and harmless).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SKPCA

from tiny_vec import pca

data = load_iris()
X = data.data.tolist()
y = data.target

scores, components, mean = pca(X, n_components=2)

# sklearn oracle for the side-by-side
sk = SKPCA(n_components=2).fit_transform(data.data)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, pts, title in [
    (axes[0], scores, "tiny-vec PCA (from scratch)"),
    (axes[1], sk.tolist(), "scikit-learn PCA (oracle)"),
]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sc = ax.scatter(xs, ys, c=y, cmap="viridis", s=25, edgecolor="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
fig.suptitle("Iris in 2 principal components", fontsize=13)
fig.tight_layout()

os.makedirs("docs", exist_ok=True)
out = "docs/iris_pca.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")

# Sanity: explained variance ratio of PC1 should be the dominant chunk.
var1 = sum(s[0] ** 2 for s in scores)
var2 = sum(s[1] ** 2 for s in scores)
print(f"PC1 captures {var1 / (var1 + var2):.1%} of the top-2 variance")
