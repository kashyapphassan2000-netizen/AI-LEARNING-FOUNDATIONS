"""Schema drift between training and serving data.

The simplest, most common, most overlooked production bug: the columns or types
silently changed between training and serving. Catching this before any
statistical test runs saves hours.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class SchemaIssue:
    kind: str           # "added", "dropped", "type_changed"
    column: str
    detail: str = ""

    def severity(self) -> str:
        return {"added": "warn", "dropped": "fail", "type_changed": "fail"}[self.kind]


def check_schema(train: pd.DataFrame, serve: pd.DataFrame) -> List[SchemaIssue]:
    issues: List[SchemaIssue] = []
    tcols, scols = set(train.columns), set(serve.columns)
    for c in sorted(tcols - scols):
        issues.append(SchemaIssue("dropped", c, "present in train, missing in serve"))
    for c in sorted(scols - tcols):
        issues.append(SchemaIssue("added", c, "new column in serve, never seen in train"))
    for c in sorted(tcols & scols):
        tt, ss = str(train[c].dtype), str(serve[c].dtype)
        if tt != ss:
            issues.append(SchemaIssue("type_changed", c, f"{tt} -> {ss}"))
    return issues
