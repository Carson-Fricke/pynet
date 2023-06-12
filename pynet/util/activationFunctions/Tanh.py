import torch as t
from .ActivationFunction import ActivationFunction


class Tanh(ActivationFunction):

    def activate(self, x, derivative=False):
        x = t.clamp(x, min=-15, max=15)
        if derivative:
            return 1 - t.tanh(x) ** 2
        return t.tanh(x)
