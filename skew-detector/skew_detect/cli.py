"""skew-detect CLI: point at training and serving CSVs, get a drift report.

The CLI is a thin shell over schema.py + drift.py. It renders a Rich table to
the terminal, and optionally writes an HTML report and a JSON file suitable
for CI gating.
"""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path
from typing import List

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from .schema import check_schema, SchemaIssue
from .drift import ks_drift, psi_drift, label_drift, DriftResult


def _render_schema(console: Console, issues: List[SchemaIssue]) -> None:
    if not issues:
        console.print("[green]schema: OK -- columns and types match[/green]")
        return
    tbl = Table(title="schema diff", title_style="bold yellow")
    tbl.add_column("severity"); tbl.add_column("column"); tbl.add_column("issue")
    for it in issues:
        color = "red" if it.severity() == "fail" else "yellow"
        tbl.add_row(f"[{color}]{it.severity()}[/{color}]", it.column,
                    f"{it.kind}: {it.detail}")
    console.print(tbl)


def _render_drift(console: Console, title: str, results: List[DriftResult],
                  metric_name: str) -> None:
    tbl = Table(title=title, title_style="bold cyan")
    tbl.add_column("column"); tbl.add_column(metric_name, justify="right")
    tbl.add_column("p-value", justify="right"); tbl.add_column("status")
    for r in results:
        status = "[red]DRIFT[/red]" if r.drifted else "[green]ok[/green]"
        pval = "n/a" if r.p_value != r.p_value else f"{r.p_value:.4g}"  # nan check
        tbl.add_row(r.column, f"{r.statistic:.4f}", pval, status)
    console.print(tbl)


def _to_html(payload: dict) -> str:
    e = lambda v: html.escape(str(v))  # column names come from user CSVs
    rows_schema = "".join(
        f"<tr><td>{e(i['kind'])}</td><td>{e(i['column'])}</td><td>{e(i['detail'])}</td></tr>"
        for i in payload["schema"])
    rows_ks = "".join(
        f"<tr><td>{e(r['column'])}</td><td>{r['statistic']:.4f}</td>"
        f"<td>{r['p_value']:.4g}</td><td>{'DRIFT' if r['drifted'] else 'ok'}</td></tr>"
        for r in payload["ks"])
    rows_psi = "".join(
        f"<tr><td>{e(r['column'])}</td><td>{r['statistic']:.4f}</td>"
        f"<td>{'DRIFT' if r['drifted'] else 'ok'}</td></tr>" for r in payload["psi"])
    lab = payload.get("label")
    label_html = "" if lab is None else (
        f"<h2>label prevalence (chi-squared)</h2><table><tr><th>column</th><th>chi2</th>"
        f"<th>p</th><th>train rate</th><th>serve rate</th><th>status</th></tr>"
        f"<tr><td>{e(lab['column'])}</td><td>{lab['statistic']:.4f}</td><td>{lab['p_value']:.4g}</td>"
        f"<td>{lab['extra']['train_prevalence']:.3f}</td><td>{lab['extra']['serve_prevalence']:.3f}</td>"
        f"<td>{'DRIFT' if lab['drifted'] else 'ok'}</td></tr></table>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>skew report</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;padding:24px}}
table{{border-collapse:collapse;margin:12px 0}}td,th{{padding:6px 12px;border:1px solid #30363d}}
th{{background:#1f6feb;color:#fff}}h1,h2{{color:#f0f6fc}}</style></head>
<body><h1>skew report</h1>
<h2>schema</h2><table><tr><th>kind</th><th>column</th><th>detail</th></tr>{rows_schema or '<tr><td colspan=3>OK</td></tr>'}</table>
<h2>KS (numeric)</h2><table><tr><th>column</th><th>D</th><th>p</th><th>status</th></tr>{rows_ks}</table>
<h2>PSI (numeric)</h2><table><tr><th>column</th><th>PSI</th><th>status</th></tr>{rows_psi}</table>
{label_html}
</body></html>"""


@click.command()
@click.argument("train", type=click.Path(exists=True))
@click.argument("serve", type=click.Path(exists=True))
@click.option("--alpha", default=0.01, show_default=True, help="significance level for KS/chi2")
@click.option("--bins", default=10, show_default=True, help="PSI buckets")
@click.option("--label", default=None, help="label column for prevalence drift")
@click.option("--report", type=click.Choice(["text", "html"]), default="text")
@click.option("--out", default=None, help="path to write HTML report")
@click.option("--json-out", default=None, help="path to write JSON for CI gates")
@click.option("--fail-on-drift/--no-fail-on-drift", default=False,
              help="exit 1 if any DRIFT is found (for CI)")
def main(train, serve, alpha, bins, label, report, out, json_out, fail_on_drift):
    """Compare TRAIN and SERVE CSVs and produce a drift report."""
    console = Console()
    tdf = pd.read_csv(train)
    sdf = pd.read_csv(serve)
    console.print(f"[dim]loaded train {tdf.shape} serve {sdf.shape}[/dim]")

    schema_issues = check_schema(tdf, sdf)
    ks = ks_drift(tdf, sdf, alpha=alpha)
    p = psi_drift(tdf, sdf, bins=bins)
    lab = label_drift(tdf, sdf, label, alpha=alpha) if label else None

    _render_schema(console, schema_issues)
    _render_drift(console, "Kolmogorov-Smirnov (Bonferroni-corrected)", ks, "D")
    _render_drift(console, "Population Stability Index", p, "PSI")
    if lab is not None:
        _render_drift(console, f"Label prevalence (chi-squared) -- {label}", [lab], "chi2")

    payload = {"schema": [s.__dict__ for s in schema_issues],
               "ks": [r.to_dict() for r in ks],
               "psi": [r.to_dict() for r in p],
               "label": lab.to_dict() if lab is not None else None}

    if report == "html" or out:
        target = Path(out or "skew_report.html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_to_html(payload))
        console.print(f"[green]wrote {target}[/green]")
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(payload, indent=2, default=str))
        console.print(f"[green]wrote {json_out}[/green]")

    any_drift = (any(s.severity() == "fail" for s in schema_issues)
                 or any(r.drifted for r in ks)
                 or any(r.drifted for r in p)
                 or (lab is not None and lab.drifted))
    if fail_on_drift and any_drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
