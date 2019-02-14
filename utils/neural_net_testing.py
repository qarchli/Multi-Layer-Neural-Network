import numpy as np

import activation_functions
import performance
import init_methods
import debug_utils


class MLNN:
    def __init__(self, layers_dims, init_method, activations, lambd, keep_prob, learning_rate, num_iterations):
        """
        Initializes a Muli-Layer Neural Network.
        ---
        Arguments:
            layers_dims: list containing the dimension of each layer of the
                        Neural Net.
            init_method: flag refering to the init method to use e.g.
                        ('xavier', 'he')
            activations: list containing the activation functions to be used
                        in the hidden layers and the output layer respectively. e.g. ('relu', 'sigmoid').
            lambd: L2 Regularization parameter.
            keep_prob: probability of keeping a neuron in dropout
                    regularization.
            learning_rate: learning rate of the gradient descent update rule
            num_iterations: number of iterations of the optimization loop
        Returns:
            MLNN: instance of MLNN.
        """
        self.__layers_dims = layers_dims
        self.__init_method = init_method
        self.__activations = activations
        self.__lambd = lambd
        self.__keep_prob = keep_prob
        self.__learning_rate = learning_rate
        self.__num_iterations = num_iterations
        # Initializing the parameters
        self.__parameters = self.__initialize_parameters(init_method)
        self.__costs = None
        self.__layer_tracker = 0

    # ========================================================================
    # Initialization
    # ========================================================================

    def __initialize_parameters(self, init_method):
        """
        Randomly initialize the W and initialize b to zeros for every layer.
        ---
        Argument:
            layers_dims: list containing the dimension of each layer of the
                        NN.
            init_method: flag refering to the init method to use e.g.
                        ('xavier', 'he')
        Returns:
            initial_parameters: python dictionnary containing the initialized
                        parameters "W1", "b1", ..., "WL", "bL".
        """

        initial_parameters = {}

        if init_method == 'xavier':
            initial_parameters = init_methods.initialize_parameters_xavier(
                self.__layers_dims)
        elif init_method == 'he':
            initial_parameters = init_methods.initialize_parameters_he(
                self.__layers_dims)
        else:
            initial_parameters = init_methods.initialize_parameters_random(
                self.__layers_dims)

        return initial_parameters

    # ========================================================================
    # Forward Propagation
    # ========================================================================

    def __one_layer_forward_propagation(self, A_prev, W_current, b_current,
                                        activation_current):
        """
        Performs forward propagation only for one layer, on all the training examples.
        ---
        Arguments:
            A_prev: activation from the previous layer. Numpy array of
                    shape   (# units of the previous layer, number of examples)
            W_current: weights. Numpy array of shape (# units of current
                    layer, # units of the previous layer)
            b_current: bias. array of shape (# units of current layer, 1)
            activation_current: activation function of the current layer as
                    string e.g. ('relu', 'sigmoid', 'tanh')
        Returns:
            A_current: the output of the activation function of the current
                    layer. numpy array of size (# units of current layer, 1)
            cache_current: cache from the current layer. (values to be used
                    later in the backprop step.)
        """
        self.__layer_tracker += 1
        print('I\'m in layer {} moving forward'.format(self.__layer_tracker))
        dispatcher = {
            'sigmoid': activation_functions.sigmoid,
            'relu': activation_functions.relu,
            'tanh': activation_functions.tanh
        }
        activation = dispatcher[activation_current]

        Z_current = np.dot(W_current, A_prev) + b_current
        A_current = activation(Z_current)

        # Forward Prop Inverted Dropout
        if self.__layer_tracker == (len(self.__layers_dims) - 1):
            D_current = np.ones(A_current.shape)
        else:
            D_current = np.random.rand(*A_current.shape)
            D_current = (D_current < self.__keep_prob)

        A_current *= D_current
        A_current /= self.__keep_prob

        cache_current = (A_prev, Z_current, W_current, D_current)

        print('A_prev.shape = ', A_prev.shape)
        print('A_current.shape = ', A_current.shape)
        print()

        return (A_current, cache_current)

    def __deep_forward_propagation(self, X):
        """
        Performs forward propagation for every layer, on all the training examples.
        ---
        Arguments:
            X: input matrix. Numpy array of shape (input size, # of examples)
        Returns:
         AL: (y_hat) the output of the last layer L.
         deep_forward_caches: caches from every layer during the forward prop.
        """

        deep_forward_caches = []
        L = int(len(self.__parameters) / 2)  # total number of layers
        m = X.shape[1]  # number of training examples
        A_current = X  # A0: initial activation

        for l in range(1, L):
            A_prev = A_current
            W_current = self.__parameters['W' + str(l)]
            b_current = self.__parameters['b' + str(l)]

            A_current, cache_current = self.__one_layer_forward_propagation(
                A_prev, W_current, b_current, self.__activations[0])
            deep_forward_caches.append(cache_current)

        # Activation of the last layer (L) using the sigmoid function
        A_prev = A_current
        WL = self.__parameters['W' + str(L)]
        bL = self.__parameters['b' + str(L)]

        AL, cacheL = self.__one_layer_forward_propagation(
            A_prev, WL, bL, self.__activations[1])
        deep_forward_caches.append(cacheL)

        assert (AL.shape == (1, m))
        return (AL, deep_forward_caches)

    # ========================================================================
    # Back Propagation
    # ========================================================================

    def __one_layer_back_propagation(self, dA_current, cache_current,
                                     activation_current):
        """
        Performs back propagation only for one layer, on all the training examples.
        ---
        Arguments:
         A_prev -- activation from the previous layer. Numpy array of shape (# units of the previous layer, number of examples)
         W_current: weights matrix. Numpy array of shape (# units of current layer, # units of the previous layer)
         b_current: bias vector. Numpy array of shape (# units of current layer, 1)
         do: boolean for dropout

        Returns:
         dA_prev -- the output of the activation function. numpy array of size (# units of current layer, 1)
        """
        print('I\'m in layer {} moving backward.'.format(self.__layer_tracker))
        n, m = dA_current.shape
        A_prev, Z_current, W_current, D_current = cache_current

        # Back Prop Inverted Dropout
        if self.__layer_tracker != (len(self.__layers_dims) - 1):
            dA_current *= D_current
            dA_current /= self.__keep_prob

        dispatcher = {
            'sigmoid': activation_functions.sigmoid_p,
            'relu': activation_functions.relu_p,
            'tanh': activation_functions.tanh_p
        }
        activation_p = dispatcher[activation_current]

        dZ_current = dA_current * activation_p(Z_current)
        dW_current = 1. / m * \
            np.dot(dZ_current, A_prev.T) + self.__lambd / m * W_current
        db_current = 1. / m * np.sum(dZ_current, axis=1, keepdims=True)
        dA_prev = np.dot(W_current.T, dZ_current)

        print('dA_prev.shape = ', dA_prev.shape)
        print('dA_current.shape = ', dA_current.shape)
        print()

        self.__layer_tracker -= 1

        return (dA_prev, dW_current, db_current)

    def __deep_back_propagation(self, AL, Y, deep_forward_caches):
        """
        Performs backward propagation for every layer, on all the training examples.
        ---
        Arguments:
         AL: probability vector, output of the deep forward propagation (deep_forward_propagation()).
         Y: true "label" vector.
         deep_forward_caches: list of caches from the deep_forward_propagation containing:
                      -"relu" activation caches (deep_caches[l], for l in range(L-1) i.e l = 0...L-2)
                      -"sigmoid" activation cache (deep_caches[L-1])
        Returns:
         grads -- A dictionary with the gradients
                  grads["dA" + str(l)] = ...
                  grads["dW" + str(l)] = ...
                  grads["db" + str(l)] = ...
        """
        L = len(deep_forward_caches)  # total number of layers
        n, m = AL.shape  # number of training examples
        Y = Y.reshape(AL.shape)
        grads = {}

        def write_grads(dA, dW, db, layer):
            """
            Writes gradient values to the returned dictionnary (grads).
            """
            # grads['dA' + str(layer)] = dA
            grads['dW' + str(layer)] = dW
            grads['db' + str(layer)] = db

        # Performing backprop on the last layer
        dA_current = -Y / AL + (1 - Y) / (
            1 - AL)  # = dAL (backprop initialization)
        cache_current = deep_forward_caches[L - 1]
        dA_prev, dW_current, db_current = self.__one_layer_back_propagation(
            dA_current, cache_current, self.__activations[1])

        write_grads(dA_current, dW_current, db_current, L)

        for l in range(L - 2, -1, -1):
            # Updating the current cache and the current derivative
            cache_current = deep_forward_caches[l]
            dA_current = dA_prev

            dA_prev, dW_current, db_current = self.__one_layer_back_propagation(
                dA_current, cache_current, self.__activations[0])

            write_grads(dA_current, dW_current, db_current, l + 1)

        # Reset layer tracker
        # self.__layer_tracker = 0

        return (grads)

    # ========================================================================
    # Weights update
    # ========================================================================

    def __update_parameters(self, grads):
        """
        Update parameters using gradient descent
        --
        Arguments:
        parameters: dictionary containing all the parameters
        grads: dictionary containing all the gradients, output of deep_backward_propagation()

        Returns:
        parameters: dictionary containing updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """
        self.__parameters = {
            param_key:
            self.__parameters[param_key] -
                self.__learning_rate * grads[grad_key]
            for param_key, grad_key in zip(
                sorted(self.__parameters.keys()), sorted(grads.keys()))
        }

    # ========================================================================
    # Training phase
    # ========================================================================

    def train(self, X, Y):
        """
        Implements a two-layer neural network.

        Arguments:
            X: input data, of shape (n_x, number of examples)
            Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        Returns:
            parameters: a dictionary containing the learnt parameters W, b.
            costs: list of costs computed after each forward prop.
        """

        # Array containing the costs
        costs = []

        # Initializing the parameteres
        # parameters = self.__initialize_parameters()

        for iteration in range(self.__num_iterations):
            print('\tIteration {}'.format(iteration))
            # Performing forward prop
            AL, deep_forward_caches = self.__deep_forward_propagation(
                X)
            # print('iteration = ', iteration)
            # print('AL ', AL)
            # Computing the cost including L2 Regularization
            cost = performance.compute_cost(
                Y, AL, self.__parameters, self.__lambd)
            costs.append(cost)

            # Performing backprop
            grads = self.__deep_back_propagation(AL, Y, deep_forward_caches)

            # Updating the parameters
            self.__update_parameters(grads)

        # Adding the costs to the MLNN params
        self.__costs = costs

        return (self)

    # ========================================================================
    # Prediction making
    # ========================================================================

    def predict(self, X, threshold=0.5):
        """
        Using the learnt parameters to predict a label for each example in X.
        --
        Arguments:
            X: input data as a numpy array of size
                (# of features, # of examples)
            parameters: the learnt parameters of the trained model

        Returns:
            predictions: vector of the predicted labels corresponding to X.
        """

        AL, _ = self.__deep_forward_propagation(X)  # Vector of probabilities

        predictions = (AL >= threshold)

        return (predictions.astype(int))

    # ========================================================================
    # Getting the params and costs
    # ========================================================================

    def get_params(self):
        """
        Returns a dictionnary of the NN hyperparameters.
        """
        params = {
            'layers_dims': self.__layers_dims,
            'init_method': self.__init_method,
            'activations': self.__activations,
            'lambda': self.__lambd,
            'learning_rate': self.__learning_rate,
            'num_iterations': self.__num_iterations,
            'weights': self.__parameters,
        }

        if self.__costs is not None:
            params['costs'] = self.__costs

        return (params)

    def set_params(self, params):
        self.__parameters = params

    # ========================================================================
    # Gradient checking to debug backpropagation step
    # ========================================================================

    def gradient_checking(self, X, Y, epsilon=1e-07):
        temp = self.__parameters  # save initial model parameters

        def grad_approx(Y, theta, sizes, epsilon=1e-07):
            """
            Computes an approximation of the gradient
            """
            dtheta_approx = []
            J_plus = []
            J_minus = []
            dim = len(theta)

            for i in range(dim):
                thetaplus = np.copy(theta)
                thetaminus = np.copy(theta)

                thetaplus[i] += epsilon
                thetaminus[i] -= epsilon

                thetaplus_dict = debug_utils.vector_to_dictionnary(
                    thetaplus, self.__layers_dims, sizes)

                thetaminus_dict = debug_utils.vector_to_dictionnary(
                    thetaminus, self.__layers_dims, sizes)

                self.__parameters = thetaplus_dict
                AL, _ = self.__deep_forward_propagation(
                    X)  # Vector of probabilities

                J_plus = performance.compute_cost(
                    Y, AL, thetaplus_dict, self.__lambd)

                self.__parameters = thetaminus_dict
                AL, _ = self.__deep_forward_propagation(
                    X)  # Vector of probabilities

                J_minus = performance.compute_cost(
                    Y, AL, thetaminus_dict, self.__lambd)

                dtheta_approx.append(
                    (J_plus - J_minus) / (2 * epsilon))

            return np.array(dtheta_approx).reshape((dim, 1))

        def compare_gradients(dtheta_backprop, dtheta_approx, epsilon=1e-07):

            numerator = np.linalg.norm(dtheta_backprop - dtheta_approx)
            denominator = np.linalg.norm(dtheta_approx) + \
                np.linalg.norm(dtheta_backprop)
            difference = numerator / denominator

            if difference > 2 * epsilon:
                CRED = '\033[91m'
                CEND = '\033[0m'
                print(CRED +
                      "There is a mistake in the backward propagation! difference = " +
                      str(difference) + ', epsilon = ' + str(epsilon) + CEND)
            else:
                print("\033[92m" +
                      "Your backward propagation works perfectly fine! difference = " +
                      str(difference) + ', epsilon = ' + str(epsilon) + "\033[0m")

        # Getting the grads of backprop
            # Performing forward prop
        AL, deep_forward_caches = self.__deep_forward_propagation(
            X)

        # Performing backprop to get the grads
        grads = self.__deep_back_propagation(AL, Y, deep_forward_caches)

        # Gradient checking
        theta, sizes = debug_utils.dictionnary_to_vector(
            self.__parameters)

        dtheta_backprop, sizes = debug_utils.dictionnary_to_vector(
            grads)
        dtheta_approx = grad_approx(Y, theta, sizes)

        compare_gradients(dtheta_backprop, dtheta_approx)

        self.__parameters = temp
