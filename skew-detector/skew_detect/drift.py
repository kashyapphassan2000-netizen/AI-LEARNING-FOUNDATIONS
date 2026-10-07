"""Distribution-drift checks: KS for numeric, PSI for numeric, chi-squared for
categorical label shift.

Each function returns a small structured result so the CLI can render and the
JSON exporter can dump without re-doing work.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DriftResult:
    column: str
    statistic: float
    p_value: float
    alpha: float
    drifted: bool
    extra: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def ks_drift(train: pd.DataFrame, serve: pd.DataFrame, alpha: float = 0.01) -> List[DriftResult]:
    """Two-sample Kolmogorov-Smirnov test per numeric column, with Bonferroni
    correction (alpha / number_of_columns) so the family-wise false-positive
    rate stays bounded as the number of tests grows.
    """
    cols = sorted(set(_numeric_cols(train)) & set(_numeric_cols(serve)))
    if not cols:
        return []
    corrected = alpha / len(cols)
    out: List[DriftResult] = []
    for c in cols:
        a = train[c].dropna().to_numpy()
        b = serve[c].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        res = stats.ks_2samp(a, b)
        out.append(DriftResult(c, float(res.statistic), float(res.pvalue),
                               corrected, bool(res.pvalue < corrected), {}))
    return out


def psi(train_col: np.ndarray, serve_col: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Bin the training data (so both sides use the
    *training* edges), then sum (p_serve - p_train) * log(p_serve / p_train)
    over bins. Rules of thumb in industry:

        PSI < 0.1          -> no significant change
        0.1 <= PSI < 0.25  -> moderate shift, watch it
        PSI >= 0.25        -> material shift, retrain
    """
    a, b = train_col[~np.isnan(train_col)], serve_col[~np.isnan(serve_col)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(a.min(), a.max() + 1e-9, bins + 1)
    p_train, _ = np.histogram(a, bins=edges)
    p_serve, _ = np.histogram(b, bins=edges)
    p_train = p_train / max(1, p_train.sum())
    p_serve = p_serve / max(1, p_serve.sum())
    eps = 1e-6
    p_train = np.where(p_train == 0, eps, p_train)
    p_serve = np.where(p_serve == 0, eps, p_serve)
    return float(np.sum((p_serve - p_train) * np.log(p_serve / p_train)))


def psi_drift(train: pd.DataFrame, serve: pd.DataFrame, bins: int = 10,
              warn: float = 0.1, fail: float = 0.25) -> List[DriftResult]:
    cols = sorted(set(_numeric_cols(train)) & set(_numeric_cols(serve)))
    out: List[DriftResult] = []
    for c in cols:
        p = psi(train[c].to_numpy(dtype=float), serve[c].to_numpy(dtype=float), bins=bins)
        out.append(DriftResult(c, p, float("nan"), fail, p >= warn,
                               {"warn": warn, "fail": fail}))
    return out


def label_drift(train: pd.DataFrame, serve: pd.DataFrame, label_col: str,
                alpha: float = 0.01) -> DriftResult | None:
    """Chi-squared test of label prevalence shift between train and serve."""
    if label_col not in train.columns or label_col not in serve.columns:
        return None
    cats = sorted(set(train[label_col].unique()) | set(serve[label_col].unique()))
    t_counts = train[label_col].value_counts().reindex(cats, fill_value=0).to_numpy()
    s_counts = serve[label_col].value_counts().reindex(cats, fill_value=0).to_numpy()
    table = np.vstack([t_counts, s_counts])
    chi2, p, _, _ = stats.chi2_contingency(table + 1)  # +1 smoothing: no zero cells
    return DriftResult(label_col, float(chi2), float(p), alpha, bool(p < alpha),
                       {"train_prevalence": float(t_counts.sum() and t_counts[-1] / t_counts.sum()),
                        "serve_prevalence": float(s_counts.sum() and s_counts[-1] / s_counts.sum())})
