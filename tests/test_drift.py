import numpy as np
import pandas as pd
from skew_detect import check_schema, ks_drift, psi_drift, psi, label_drift


def _train_serve(seed=0, drift=False, n=2000):
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(5, 2, n),
        "label": (rng.random(n) < 0.3).astype(int),
    })
    shift = 1.5 if drift else 0.0
    rate = 0.55 if drift else 0.3
    serve = pd.DataFrame({
        "x": rng.normal(shift, 1, n),
        "y": rng.normal(5, 2, n),
        "label": (rng.random(n) < rate).astype(int),
    })
    return train, serve


def test_schema_clean_pair_has_no_issues():
    t, s = _train_serve()
    assert check_schema(t, s) == []


def test_schema_detects_drop_add_type_change():
    t, s = _train_serve()
    s2 = s.drop(columns=["y"]).copy()
    s2["new_col"] = 1.0
    s2["x"] = s2["x"].astype("float32")
    issues = check_schema(t, s2)
    kinds = {i.kind for i in issues}
    assert {"dropped", "added", "type_changed"} <= kinds


def test_ks_catches_known_drift():
    t, s = _train_serve(drift=True)
    res = {r.column: r for r in ks_drift(t, s)}
    assert res["x"].drifted is True   # we shifted x by 1.5 sigma
    assert res["y"].drifted is False  # y unchanged


def test_ks_no_false_positive_on_clean_pair():
    t, s = _train_serve(drift=False)
    for r in ks_drift(t, s):
        assert r.drifted is False


def test_psi_value_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b_clean = rng.normal(0, 1, 5000)
    b_shifted = rng.normal(2.0, 1, 5000)
    assert psi(a, b_clean) < 0.1
    assert psi(a, b_shifted) > 0.25


def test_label_drift_chi2_detects_prevalence_shift():
    t, s = _train_serve(drift=True)
    r = label_drift(t, s, "label")
    assert r is not None and r.drifted is True
