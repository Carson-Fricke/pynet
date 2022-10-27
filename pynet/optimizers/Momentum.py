from .Optimizer import Optimizer
import torch as t


class Momentum(Optimizer):

    def __init__(self, eta, batch, alpha=0.9):
        super().__init__()
        self.eta = eta
        self.alpha = alpha
        self.examples_parsed = 0
        self.weight_gradients = None
        self.bias_gradients = None
        self.w_previous = 0
        self.b_previous = 0
        self.batch = batch

    def optimize(self, weight_gradients, bias_gradients):

        if self.examples_parsed % self.batch == 0:
            self.weight_gradients = t.zeros_like(weight_gradients, device=self.cuda0)
            self.bias_gradients = t.zeros_like(bias_gradients, device=self.cuda0)

        self.examples_parsed += 1
        self.weight_gradients += weight_gradients
        self.bias_gradients += bias_gradients

        if self.examples_parsed == self.batch:
            self.examples_parsed = 0
            self.w_previous = self.eta * ((1 - self.alpha) * self.weight_gradients + self.alpha * self.w_previous)
            self.b_previous = self.eta * ((1 - self.alpha) * self.bias_gradients + self.alpha * self.b_previous)
            return self.w_previous, self.b_previous

        else:
            return t.zeros_like(self.weight_gradients, device=self.cuda0), \
                   t.zeros_like(self.bias_gradients, device=self.cuda0)