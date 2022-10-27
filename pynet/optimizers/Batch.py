from .Optimizer import Optimizer
import torch as t


class Batch(Optimizer):

    def __init__(self, eta, batch):
        super().__init__()
        self.eta = eta
        self.examples_parsed = 0
        self.weight_gradients = None
        self.bias_gradients = None
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
            return self.eta * self.weight_gradients, self.eta * self.bias_gradients

        else:
            return t.zeros_like(self.weight_gradients, device=self.cuda0), \
                   t.zeros_like(self.bias_gradients, device=self.cuda0)