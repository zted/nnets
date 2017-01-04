import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.max([x, np.zeros(x.shape)], axis=0)

def reluPrime(x):
    tmp = relu(x)
    return tmp/tmp

def tanh(x):
    numer = np.exp(x) - np.exp(-x)
    denom = np.exp(x) + np.exp(-x)
    return numer/denom

def tanhPrime(x):
    tmp = tanh(x)
    return 1-tmp*tmp

def linear(x):
    return x

def linearPrime(x):
    return np.ones(x.shape)
