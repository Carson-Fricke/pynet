import torch as t
from ActivationFunction import ActivationFunction


class Softmax(ActivationFunction):

    def activate(self, x, derivative=False):
        t.clamp(x, min=-30, max=30)
        if derivative:
            return t.zeros_like(x).fill_(t.sum(t.exp(x)))
        return t.exp(x) / t.sum(t.exp(x))