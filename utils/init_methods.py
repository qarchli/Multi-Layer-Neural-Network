import numpy as np


def initialize_parameters_zeros(layers_dims):
    """
    Initializes the parameters to zeros.
    ---
    Argument:
        layers_dims: list containing the dimension of each layer of the
                    Neural Net.
    Returns:
        initial_parameters: python dictionnary containing the initialized
                    parameters "W1", "b1", ..., "WL", "bL".
    """
    parameters = {}
    # integer representing the number of layers
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l],
                                             layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    """
    Performs random parameters' initialization.
    ---
    Argument:
        layers_dims: list containing the dimension of each layer of the
                    Neural Net.
    Returns:
        initial_parameters: python dictionnary containing the initialized
                    parameters "W1", "b1", ..., "WL", "bL".
    """
    parameters = {}
    # integer representing the number of layers
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],
                                                   layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_xavier(layers_dims):
    """
    Performs Xavier parameters' initialization.
    ---
    Argument:
        layers_dims: list containing the dimension of each layer of the
                    Neural Net.
    Returns:
        initial_parameters: python dictionnary containing the initialized
                    parameters "W1", "b1", ..., "WL", "bL".
    """
    parameters = {}
    # integer representing the number of layers
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],
                                                   layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_he(layers_dims):
    """
    Performs HE parameters' initialization.
    ---
    Argument:
        layers_dims: list containing the dimension of each layer of the
                    Neural Net.
    Returns:
        initial_parameters: python dictionnary containing the initialized
                    parameters "W1", "b1", ..., "WL", "bL".
    """
    parameters = {}
    # integer representing the number of layers
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],
                                                   layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters
