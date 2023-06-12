pynet
-------------

Pynet is a small neural network library built upon pytorch tensors intended to demonstrate backpropogation with various activation functions and gradient descent optimizers. It was written in the spirit of "built from scratch," and only uses pytorch tensors to speed up performance and allow for GPU parallelization. 
## Getting Started

Simply download the code and import the pynet module to get started.

Alternatively, download the code and run the inbuilt test file which trains a Dense neural network on MNIST digit classification.

## Implementation
All of the basic mathematic tensor operations rely on the pytorch library, and could be easilly subbed out for numpy or for a handwritten implementation.

Forwards and backwards proppogation is implemented for various Layer types.
For example, Dense is a typical Dense layer, and DenseResidual carries previous layers activations on to the next layer.

A wide variety of activation functions and gradient descent opimizers are also implemented.
