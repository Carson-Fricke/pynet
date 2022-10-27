from pynet import Layer
import torch as t
import math

class Conv(Layer):

    def __init__(self, shape, number_of_filters, optimizer, conv_shape=(3, 3), input_shape=()):

        super().__init__(number_of_filters + shape, input_shape, optimizer)
        self.weights = t.randn(number_of_filters + conv_shape)

    def forward(self, x):

        t_sums = self.sums.view(math.floor(t.numel(self.sums) / 2), math.ceil(t.numel(self.sums) / 2))
        t_outputs = self.output.view(math.floor(t.numel(self.output) / 2), math.ceil(t.numel(self.output) / 2))

