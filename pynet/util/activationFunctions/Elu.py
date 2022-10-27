import torch as t
from ActivationFunction import ActivationFunction


class Elu(ActivationFunction):

    def activate(self, x, derivative=False):
        if derivative:
            return t.min(t.exp(x), t.ones_like(x))
        return t.min(t.exp(x) - 1, t.max(x, t.zeros_like(x)))
