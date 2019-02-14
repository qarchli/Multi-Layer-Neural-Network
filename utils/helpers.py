import itertools

import numpy as np


def dictionnary_to_vector(parameters):
    theta = []
    sizes = []
    for item in sorted(parameters.items()):
        sizes.append(item[1].shape[0] * item[1].shape[1])
        theta.append(item[1].reshape(item[1].shape[0] * item[1].shape[1], 1))

    flat = itertools.chain.from_iterable(theta)
    theta = np.array(list(flat))

    return theta, sizes


def vector_to_dictionnary(theta, layers_dims, sizes):

    parameters = {}
    vector = []

    L = len(layers_dims)  # total number of layers
    num_params = len(sizes)  # total number of parameters (W and b)
    num_W = int(num_params / 2)  # total number of Ws

    # Convert theta to multiple chunks referring to NN parameters
    start = 0
    stop = 0

    for i in range(len(sizes)):
        stop += sizes[i]
        vector.append(theta[start:stop])
        start = stop

    vector = np.array(vector)

    for w, b, l in zip(vector[:num_W], vector[num_W:], range(1, L)):
        parameters['W' + str(l)] = w.reshape(layers_dims[l],
                                             layers_dims[l - 1])
        parameters['b' + str(l)] = b.reshape(layers_dims[l], 1)

    return parameters


def J(clf, X, Y, parameters):
    AL, forward_caches = clf.deep_forward_propagation(X, parameters)

    m = AL.shape[1]
    cost = -1 / m * np.sum(
        np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))

    return cost.squeeze(), AL, forward_caches


def grad_backprop(clf, AL, Y, forward_caches):
    grads = clf.deep_back_propagation(AL, Y, forward_caches)

    return grads


def grad_approx(clf, X, Y, theta, sizes, epsilon=1e-07):
    dtheta_approx = []
    J_plus = []
    J_minus = []
    dim = len(theta)

    for i in range(dim):
        thetaplus = np.copy(theta)
        thetaminus = np.copy(theta)

        thetaplus[i] += epsilon
        thetaminus[i] -= epsilon

        J_plus = J(clf, X, Y,
                   vector_to_dictionnary(thetaplus, clf.layers_dims, sizes))[0]
        J_minus = J(clf, X, Y,
                    vector_to_dictionnary(thetaminus, clf.layers_dims,
                                          sizes))[0]

        dtheta_approx.append((J_plus - J_minus) / (2 * epsilon))

    return np.array(dtheta_approx).reshape((dim, 1))


def grad_checking(dtheta_backprop, dtheta_approx, epsilon=1e-07):

    numerator = np.linalg.norm(dtheta_backprop - dtheta_approx)
    denominator = np.linalg.norm(dtheta_approx) + \
        np.linalg.norm(dtheta_backprop)
    difference = numerator / denominator

    if difference > 2 * epsilon:
        print("\033[93m" +
              "There is a mistake in the backward propagation! difference = " +
              str(difference) + "\033[0m")
    else:
        print("\033[92m" +
              "Your backward propagation works perfectly fine! difference = " +
              str(difference) + "\033[0m")

    return difference
