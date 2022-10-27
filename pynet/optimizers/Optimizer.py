import torch as t


class Optimizer:

    cuda0 = t.device('cuda:0')

    def __init__(self):
        self.eta = 0.01

    def __call__(self, gradients, weights):
        return self.optimize(gradients, weights)

    def optimize(self, weight_gradients, bias_gradients):
        return weight_gradients * self.eta, bias_gradients * self.eta

    def get_eta(self):
        return self.eta

    def set_eta(self, x):
        self.eta = x
