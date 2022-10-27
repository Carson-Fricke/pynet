import torch as t
import numpy as np


cuda0 = t.device('cuda:0')


class Layer:

    def __init__(self, shape, input_shape, activation, optimizer):

        self.output = t.empty(shape, device=cuda0)
        self.sums = t.empty(shape, device=cuda0)
        self.shape = shape
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.activation_function = activation

    def __str__(self):
        return str(type(self))[21:-2][:int(len(str(type(self))[21:-2]) / 2)] + " " + str(self.shape)

    def set_output(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = t.from_numpy(x).float().to(cuda0)
        self.output = x

    def set_sums(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = t.from_numpy(x).float().to(cuda0)
        self.sums = x

    def set_optimizer(self, x):
        self.optimizer = x

    def get_output(self):
        return self.output

    def get_sums(self):
        return self.sums

    def get_optimmizer(self, x):
        return self.optimizer