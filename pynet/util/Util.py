import pickle
import torch as t

def import_model(file):
    with open(file + '.model', 'rb+') as p:
        return pickle.load(p)


def normalize(tensor):

    return (tensor - t.mean(tensor)) / t.std(tensor)

def prod(x):
    out = 1
    for i in x:
        out *= i
    return out
