class ActivationFunction:

    def __call__(self, x, derivative=False):
        return self.activate(x, derivative=derivative)

    def activate(self, x, derivative=False):
        if derivative:
            return 1
        return x