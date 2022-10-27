import torch as t
from .Optimizer import Optimizer


class RMSprop(Optimizer):

    def __init__(self, eta, batch, epsilon=0.00000001):

        super().__init__()
        self.eta = eta
        self.epsilon = epsilon
        self.batch = batch
        self.examples_parsed = -1

        self.weight_gradients = None
        self.bias_gradients = None
        self.w_s = None
        self.b_s = None
        self.beta = 0.9

    def optimize(self, weight_gradients, bias_gradients):

        if self.examples_parsed == -1:

            self.examples_parsed = 0
            self.weight_gradients = t.zeros_like(weight_gradients, device=self.cuda0).to(t.device('cuda:0'))
            self.bias_gradients = t.zeros_like(bias_gradients, device=self.cuda0).to(t.device('cuda:0'))
            self.w_s = t.zeros_like(weight_gradients, device=self.cuda0).to(t.device('cuda:0'))
            self.b_s = t.zeros_like(bias_gradients, device=self.cuda0).to(t.device('cuda:0'))

        self.examples_parsed += 1
        self.weight_gradients += weight_gradients
        self.bias_gradients += bias_gradients

        if self.batch == self.examples_parsed:

            self.w_s = self.beta * self.w_s + (1 - self.beta) * self.weight_gradients ** 2
            self.b_s = self.beta * self.b_s + (1 - self.beta) * self.bias_gradients ** 2

            out_w = self.eta * self.weight_gradients / t.sqrt(self.w_s + self.epsilon)
            out_b = self.eta * self.bias_gradients / t.sqrt(self.b_s + self.epsilon)

            self.weight_gradients = t.zeros_like(weight_gradients, device=self.cuda0).to(t.device('cuda:0'))
            self.bias_gradients = t.zeros_like(bias_gradients, device=self.cuda0).to(t.device('cuda:0'))
            self.examples_parsed = 0

            return out_w, out_b

        else:
            return t.zeros_like(self.weight_gradients, device=self.cuda0), \
                   t.zeros_like(self.bias_gradients, device=self.cuda0)