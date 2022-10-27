import torch as t


class FeatureMap:

    def __init__(self, network):

        self.network = network

        self.features = []

        self.generate_map()

    def generate_map(self):

        layer1 = self.network.layers[1]

        biases = layer1.biases.view(layer1.biases.shape + (1, 1))

        start = layer1.activation_function(layer1.weights - biases)
        self.features.append(start)

        i = 0

        for layer in self.network.layers[2:]:

            biases = layer.biases.view(layer.biases.shape + (1, 1))

            self.features.append(layer.activation_function(t.tensordot(layer.weights, self.features[i], 1) - biases))
            i += 1