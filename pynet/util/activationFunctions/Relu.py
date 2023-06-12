import torch as t
from .ActivationFunction import ActivationFunction


class Relu(ActivationFunction):

    def activate(self, x, derivative=False):
        if derivative:
            return (t.sign(x) + 1) / 2
        return t.max(x, t.zeros_like(x))
