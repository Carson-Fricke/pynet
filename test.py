from pynet.layers.Dense import Dense
from pynet.layers.Output import Output
from pynet.network.Network import Network
from pynet.optimizers.RMSprop import RMSprop
from pynet.util.Util import normalize
from pynet.util.activationFunctions.Relu import Relu

import numpy as np
import torch as t
import keras.datasets.mnist

cuda0 = t.device('cuda:0')
(images, t_lables), (test_x, ttm) = keras.datasets.mnist.load_data()
temp = []


for ex in t_lables:
    o = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    o[ex] = 1
    temp.append(o)

images  = normalize(t.tensor(np.array([np.reshape(k, (28, 28)) for k in images]), dtype=t.float)).to(cuda0)
labels = t.tensor(temp, dtype=t.float).to(cuda0)



x = [Dense(shape=(28, 28), in_shape=(), optimizer=RMSprop(0.002, 100)),
     Dense(shape=(10, 10), in_shape=(28, 28), activation=Relu(), optimizer=RMSprop(0.002, 100)),
     Dense(shape=(7, 7), in_shape=(10, 10), activation=Relu(), optimizer=RMSprop(0.002, 100)),
     Dense(shape=(6, 6), in_shape=(7, 7), activation=Relu(), optimizer=RMSprop(0.002, 100)),
     Output(shape=(10,), in_shape=(6, 6), optimizer=RMSprop(0.002, 100))
     ]
model = Network(x)


while True:

    for i in range(60000):
        model.forward(images[i])
        model.backward(labels[i])
        if i % 2000 == 0:
            model.evaluate(images[:1000], labels[:1000])
    
