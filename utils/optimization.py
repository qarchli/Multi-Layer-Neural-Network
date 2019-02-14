import numpy as np
import math


def get_random_minibatches(X, Y, minibatch_size=64):
    """
    Splits the original training set (X, Y) into equal-sized minibatches.
    --
    Arguments:
        X: input data, of shape (number of features, number of examples)
        Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        minibatch_size: size of each minibatch. (preferably a power of 2)
    Returns:
        minibatches: list of minibatches.
                    [(X_{1}, Y_{1}), ..., (X_{t}, Y_{t})]
    """

    m = X.shape[1]  # total number of examples
    minibatches = []

    remaining = False  # will the dataset split be even or not
    if m % minibatch_size != 0:
        remaining = True

    # Step 1: Shuffling the original dataset
    permutations = np.random.permutation(m)
    X_shuffled = X[:, permutations]
    Y_shuffled = Y[:, permutations]

    # Step 2: Splitting the shuffled dataset into equal-sized minibatches.

    # total number of minibatches of size = minibatch_size
    num_complete_minibatches = math.floor(m / minibatch_size)

    for k in range(num_complete_minibatches):
        complete_minibatch_X = X_shuffled[:, k *
                                          minibatch_size:(k + 1) * minibatch_size]
        complete_minibatch_Y = Y_shuffled[:, k *
                                          minibatch_size:(k + 1) * minibatch_size]
        minibatches.append((complete_minibatch_X, complete_minibatch_Y))

    # handling the remainded tiny minibatch
    if remaining:
        complete_minibatch_X = X_shuffled[:, (k + 1) *
                                          minibatch_size:]
        complete_minibatch_Y = Y_shuffled[:, (k + 1) *
                                          minibatch_size:]
        minibatches.append((complete_minibatch_X, complete_minibatch_Y))

    return minibatches


def update_parameters_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    --
    Arguments:
        parameters: python dictionary containing the parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
        grads: python dictionary containing the gradients to update each parameter:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
        learning_rate: the learning rate, scalar.
    Returns:
        updated_parameters: python dictionary containing the updated parameters
    """
    updated_parameters = {
        param_key:
        parameters[param_key] - learning_rate * grads[grad_key]
        for param_key, grad_key in zip(sorted(parameters.keys()), sorted(grads.keys()))
    }

    return updated_parameters

# ========================================================================
    # Momentum
# ========================================================================


def initialize_momentum(parameters):
    """
    Initializes velocity to zeros, for each layer in the NN
    --
    Arguments:
        parameters: python dictionary containing the parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    Returns:
    v: python dictionary containing the current velocity.
            v['dW' + str(l)] = velocity of dWl
            v['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2  # number of layers in the neural network
    v = {}

    # Initialize velocity for each layer
    for l in range(1, L + 1):
        v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)

    return v


def update_parameters_momentum(parameters, grads, learning_rate, v, beta):
    """
    Update parameters using one step of gradient descent with momentum.
    --
    Arguments:
        parameters: python dictionary containing the parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
        grads: python dictionary containing the gradients to update each
                parameter:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
        beta: velocity parameter, scalar.
        learning_rate: the learning rate, scalar.
    Returns:
        parameters: python dictionary containing your updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Momentum update for each parameter
    for l in range(1, L + 1):
        # compute velocities
        v['dW' + str(l)] = beta * v['dW' + str(l)] + \
            (1 - beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + \
            (1 - beta) * grads['db' + str(l)]

        # update parameters
        parameters['W' + str(l)] -= learning_rate * v['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * v['db' + str(l)]

    return parameters, v

# ========================================================================
    # RMSProp
# ========================================================================


def initialize_rmsprop(parameters):
    """
    Initializes velocity to zeros, for each layer in the NN
    Arguments:
        parameters: python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    Returns:
        s: python dictionary containing the current velocity.
            s['dW' + str(l)] = velocity of dWl
            s['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2  # number of layers in the neural network
    s = {}

    # Initialize velocity for each layer
    for l in range(1, L + 1):
        s['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        s['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)

    return s


def update_parameters_rmsprop(parameters, grads, learning_rate, s, beta, epsilon=10e-8):
    """
    Update parameters using one step of gradient descent with RMSProp.
    --
    Arguments:
        parameters: python dictionary containing the parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
        grads: python dictionary containing the gradients to update each
                parameter:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
        learning_rate: the learning rate, scalar.
        s: python dictionary containing the current velocity:
                    s['dW' + str(l)] = ...
                    s['db' + str(l)] = ...
        beta: the momentum hyperparameter, scalar
    Returns:
        parameters: python dictionary containing your updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Momentum update for each parameter
    for l in range(1, L + 1):
        # compute velocities
        s['dW' + str(l)] = beta * s['dW' + str(l)] + \
            (1 - beta) * grads['dW' + str(l)]**2
        s['db' + str(l)] = beta * s['db' + str(l)] + \
            (1 - beta) * grads['db' + str(l)]**2

        # update parameters
        parameters['W' + str(l)] -= learning_rate * grads['dW' +
                                                          str(l)] / (np.sqrt(s['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] -= learning_rate * grads['db' +
                                                          str(l)] / (np.sqrt(s['db' + str(l)]) + epsilon)
    return parameters, s

# ========================================================================
    # ADAM
# ========================================================================


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
        parameters: python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
        v: python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
        s: python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
        t: Adam counter, initialized to 0.
    """
    v = initialize_momentum(parameters)
    s = initialize_rmsprop(parameters)
    t = 0

    return v, s, t


def update_parameters_adam(parameters, grads, learning_rate, v, s, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using one step of gradient descent with momentum.
    --
    Arguments:
        parameters: python dictionary containing the parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
        grads: python dictionary containing the gradients to update each
                parameter:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.
        v -- Adam variable, moving average of the first gradient, python
            dictionary
        s -- Adam variable, moving average of the squared gradient, python
            dictionary
        t -- Adam counter
        beta1 -- Exponential decay hyperparameter for the first moment
            estimates
        beta2 -- Exponential decay hyperparameter for the second moment
            estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
        parameters -- python dictionary containing the updated parameters
        v -- Adam variable, moving average of the first gradient, python
            dictionary
        s -- Adam variable, moving average of the squared gradient, python
            dictionary
    """

    L = len(parameters) // 2  # number of layers in the neural network
    v_corrected = {}
    s_corrected = {}

    # Adam update for each parameter
    for l in range(1, L + 1):
        # compute velocities
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + \
            (1 - beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + \
            (1 - beta1) * grads['db' + str(l)]

        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + \
            (1 - beta2) * grads['dW' + str(l)]**2
        s['db' + str(l)] = beta2 * s['db' + str(l)] + \
            (1 - beta2) * grads['db' + str(l)]**2

        # bias correction
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)

        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

        # update parameters
        parameters['W' + str(l)] -= learning_rate * v_corrected["dW" +
                                                                str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters['b' + str(l)] -= learning_rate * v_corrected["db" +
                                                                str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s
