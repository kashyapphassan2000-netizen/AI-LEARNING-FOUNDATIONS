"""tiny-vec: linear algebra and autograd from first principles."""
from .vector import Vector
from .matrix import Matrix
from .engine import Value
from .nn import Neuron, Layer, MLP
from .linalg import power_iteration, svd, pca

__version__ = "0.1.0"
__all__ = [
    "Vector", "Matrix", "Value", "Neuron", "Layer", "MLP",
    "power_iteration", "svd", "pca",
]
