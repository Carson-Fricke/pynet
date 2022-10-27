from pynet import Optimizer


class SGD(Optimizer):

    def __init__(self, eta):
        super().__init__()
        self.eta = eta
