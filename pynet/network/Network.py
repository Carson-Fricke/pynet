import pickle
import torch as t
import numpy as np


class Network:

    def __init__(self, topology, clone=False):

        # generate topology
        # from layers with unfilled input shapes
        self.layers = []
        previous_layer_shape = ()

        if clone:

            self.layers = topology
        else:
            for layer in topology:
                self.layers.append(type(layer)(layer.shape,
                                               layer.optimizer,
                                               in_shape=previous_layer_shape,
                                               activation=layer.activation_function,
                                               init=True))
                previous_layer_shape = self.layers[-1].shape

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        out = ""
        for i in self.layers:
            out += str(i) + "\n"
        return out

    def get_output(self):
        return self.layers[-1].get_output()

    def get_sub_network(self, *args):
        return Network(self.layers[args[0]:args[1]], clone=True)

    def forward(self, x):

        # forward prop each layer sequentially
        # use each previous layer's output
        self.layers[0].set_output(x)
        previous_layer = self.layers[0].get_output()
        for layer in self.layers[1:]:
            previous_layer = layer.forward(previous_layer)

        return self.get_output()

    def backward(self, target):

        # backward prop each layer sequentially backwards
        # use the previous and the next layer
        error = target
        i = len(self.layers) - 1

        for layer in self.layers[1:][::-1]:
            # print(error)
            error = layer.backward(error, self.layers[i - 1].get_output())
            i -= 1
        # print()
        return error

    def evaluate(self, inputs, outputs, show=True):

        cr = 0
        err = 0
        for x, y in zip(inputs, outputs):
            self.forward(x)
            output = self.get_output()
            diff = output - y
            err += t.sum(t.abs(diff)) / t.numel(output)
            if t.argmax(output) == t.argmax(y):
                if output.numel() == 1:
                    out = output.view((1,))[0]
                    y1 = y.view((1,))[0]
                    if (out > 0.5 and y1 > 0.5) or (out < 0.5 and y1 < 0.5):
                        cr += 1
                else:
                    cr += 1
        # print(t.numel(output) * len(inputs))
        cr = cr / len(inputs)
        err = err / len(inputs)
        if show:
            o1 = '|'
            for _ in range(int(np.floor(cr * 100)) - 1):
                o1 += '='
            o1 += '>'
            for _ in range(int(100 - np.floor(cr * 100))):
                o1 += '.'
            o1 += '| Loss: '
            print(o1 + str(float(err))[:15] + ' | CR: ' + str(cr)[:5])
        return cr, err

    def save(self, file):
        print('saving. please do not close', end='\r')
        with open(file + '.model', 'wb+') as p:
            pickle.dump(self, p)