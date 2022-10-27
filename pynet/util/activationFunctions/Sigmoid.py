import torch as t
from .ActivationFunction import ActivationFunction


class Sigmoid(ActivationFunction):

    def activate(self, x, derivative=False):
        x = t.clamp(x, min=-15, max=15)
        if derivative:
            return self.activate(x) * self.activate(-x)
        return 1 / (1 + t.exp(-x))