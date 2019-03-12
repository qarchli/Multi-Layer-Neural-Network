import numpy as np


def sigmoid(z):
    """
    Computes the sigmoid activation function for a given numpy array.
    """
    return (1 / (1 + np.exp(-z)))


def sigmoid_p(z):
    """
    Computes the gradient of the sigmoid for a given numpy array.
    """
    return (sigmoid(z) * (1 - sigmoid(z)))


def relu(z):
    """
    Computes the relu activation function for a given numpy array.
    """
    result = np.maximum(0, z)

    return result


def relu_p(z):
    """
    Computes the gradient of the relu for a given numpy array.
    """
    result = z
    result[result <= 0] = 0
    result[result > 0] = 1

    return result


def tanh(z):
    """
    Computes the tanh activation function for a given numpy array.
    """
    return (np.tanh(z))


def tanh_p(z):
    """
    Computes the gradient of the tanh function for a given numpy array.
    """
    return (1 - np.tanh(z)**2)


def softmax(x):
    """
    Computes the softmax activation function for a given numpy array.
    """
    shiftx = x - np.max(x)
    return (np.exp(shiftx)) / np.sum(np.exp(shiftx), axis=0)


def softmax_p(z):
    """
    Computes the gradient of the softmax function for a given numpy array.
    """
    # return softmax(z) * (1 - softmax(z))
    return np.diagflat(z) - np.dot(z, z.T)
