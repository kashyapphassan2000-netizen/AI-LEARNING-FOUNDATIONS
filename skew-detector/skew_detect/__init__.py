"""skew-detector: training-serving skew checks (schema, KS, PSI, label prevalence)."""
from .schema import check_schema, SchemaIssue
from .drift import ks_drift, psi_drift, psi, label_drift, DriftResult

__version__ = "0.1.0"
__all__ = ["check_schema", "SchemaIssue", "ks_drift", "psi_drift", "psi",
           "label_drift", "DriftResult"]
