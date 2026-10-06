"""Emit a Graphviz DOT description of a Value's computation graph.

Run: python examples/visualize.py | dot -Tpng -o docs/graph.png
"""
from .engine import Value


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def to_dot(root) -> str:
    nodes, edges = trace(root)
    lines = ["digraph G {", '  rankdir=LR;',
             '  bgcolor="#0d1117";',
             '  node [style=filled, fontname="Helvetica", fontcolor="#e6edf3", color="#30363d"];',
             '  edge [color="#8b949e"];']
    for n in nodes:
        nid = str(id(n))
        label = f"data {n.data:.3f} | grad {n.grad:.3f}"
        lines.append(f'  "{nid}" [label="{{ {label} }}", shape=record, fillcolor="#161b22"];')
        if n._op:
            op_id = nid + n._op
            lines.append(f'  "{op_id}" [label="{n._op}", shape=ellipse, fillcolor="#1f6feb"];')
            lines.append(f'  "{op_id}" -> "{nid}";')
    for a, b in edges:
        lines.append(f'  "{id(a)}" -> "{id(b)}{b._op}";')
    lines.append("}")
    return "\n".join(lines)
