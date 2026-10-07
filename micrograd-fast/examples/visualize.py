"""Build L = relu(a*b + c), run backward, and print the graph as Graphviz DOT.

python examples/visualize.py | dot -Tpng -o docs/graph.png
"""
from micrograd_fast import Value
from micrograd_fast.viz import to_dot

a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c).relu()
L.backward()
print(to_dot(L))
