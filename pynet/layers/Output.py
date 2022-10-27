import torch as t
from pynet.util.activationFunctions.Sigmoid import Sigmoid
from pynet.layers.Layer import Layer
from pynet.util.Util import prod
from math import sqrt


cuda0 = t.device('cuda:0')


class Output(Layer):

    def __init__(self,
                 shape,
                 optimizer,
                 in_shape=(),
                 activation=Sigmoid(),
                 init=True):

        super().__init__(shape,
                         in_shape,
                         activation,
                         optimizer)

        if init:
            self.biases = t.randn(shape, device=cuda0) * sqrt(2 / (prod(in_shape) + prod(self.shape)))

            self.weights = t.randn(shape + in_shape, device=cuda0) * sqrt(2 / (prod(in_shape) + prod(self.shape)))

    def error_function(self, y, a, type='true cross entropy'):
        if type == 'cross entropy':
            return y * t.log(a) + (1 - y) * t.log(1 - a)
        elif type == 'mse':
            return -2
        elif type == 'true cross entropy':
            return -((y / a) + ((1 - y) / (1 - a)))

    def forward(self, x):

        # init layer shaped temp tensors
        # flatten layer shaped tensors
        t_input = x.view(t.numel(x))
        t_biases = self.biases.view(t.numel(self.biases))

        # init weight shaped temp tensors
        # to be reshaped back into layer feilds
        t_weights = self.weights.view((t.numel(self.output), t.numel(x)))

        # generate dot products
        # sum over weights and biases
        t_sums = t.tensordot(t_weights, t_input, 1) + t_biases
        """ Equivilent to
            for i in range(len(weights)):
                for j in range(len(weights[i])):
                    sums[i] += input[j] * weights[i][j]
        """

        # activation function applied
        # adds nonlinearity
        t_output = self.activation_function(t_sums)

        # reshaping
        # convert back to layer feilds
        self.output = t_output.view(self.shape)
        self.sums = t_sums.view(self.shape)

        return self.get_output()

    def backward(self, target, next_layer_output):

        # init layer shaped temp tensors
        # flatten layer shaped tensors
        target = target.view(t.numel(target))
        t_output = self.output.view(t.numel(self.output))
        t_sums = self.sums.view(t.numel(self.sums))
        t_next_layer = next_layer_output.view(t.numel(next_layer_output))

        # init weight shaped temp tensors
        # reshape into 2D weights
        t_weights = self.weights.view((t.numel(self.output), t.numel(t_next_layer)))

        # output layer error
        # categorical cross entropy

        t_error = self.error_function(target, t_output) * self.activation_function(t_sums, derivative=True) * (target - t_output)

        # print(self.output, target, t_error, self.activation_function(t_sums, derivative=True))

        # next layer error
        # transpose weights for the for loop
        t_weights = t.t(t_weights)
        next_layer_error = t.tensordot(t_weights, t_error, 1)

        # previous updates
        # biases and weights
        t_w_updates = t.ger(t_error, t_next_layer)
        t_b_updates = t_error

        w_updates, b_updates = self.optimizer(t_w_updates.view(self.shape + self.input_shape), t_b_updates.view(self.shape))
        self.weights -= w_updates
        self.biases -= b_updates
        # self.weights += 0.001 * w_updates
        # self.biases += 0.001 * b_updates

        # return next layer error
        # for entire network
        return next_layer_error.view(next_layer_output.size())