# V3.0
import numpy as np

from utils import activation_functions
from utils import performance
from utils import init_methods
from utils import helpers
from utils import optimization


class MLNN:

    __layer_tracker = 0  # to track the current layer

    def __init__(self, layers_dims, init_method, activation, lambd, optimizer_name, learning_rate, num_epochs, beta1=0.9, beta2=0.999, minibatch_size=64, track=False):
        """
        Initializes a Muli-Layer Neural Network.
        ---
        Arguments:
            layers_dims: list containing the dimension of each layer of the Neural Net.
            init_method: flag refering to the init method to use e.g.
                        ('xavier', 'he')
            activation: string. activation for the hidden layers.
                        ('relu', 'tanh', 'sigmoid')
            lambd: L2 regularization parameter.
            optimizer_name: the optimizer to use in the parameters' update
                        step. e.g.('gd', 'momentum', 'rmsprop', 'adam') for standard gradient descent, momentum, rmsprop and adam, respectively.
            learning_rate: learning rate of the gradient descent update step.
            num_epochs: number of epochs for the training loop.
            beta1: Exponential decay hyperparameter for the first moment
                        estimates
            beta2: Exponential decay hyperparameter for the second moment
                        estimates
            minibatch_size: size of minibatches in gradient descent update step.
            track: boolean to track NN evolution.
        Returns:
            MLNN: instance of MLNN.
        """
        self.__layers_dims = layers_dims

        # boolean for the classification type
        if self.__layers_dims[-1] == 1:
            self.__binary_class = True
            self.__activations = [activation, 'sigmoid']
        else:
            self.__binary_class = False
            self.__activations = [activation, 'softmax']

        self.__init_method = init_method
        self.__lambd = lambd
        self.__optimizer_name = optimizer_name
        self.__optimizer = None
        self.__learning_rate = learning_rate
        self.__num_epochs = num_epochs
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__minibatch_size = minibatch_size
        self.__track = track
        self.__parameters = None
        self.__costs = None

        MLNN.__layer_tracker = 0  # reinit layer tracker

    # ========================================================================
    # Initialization
    # ========================================================================

    def __initialize_parameters(self):
        """
        Initializes the NN parameters (W and b) for every layer, depending on the required method.
        ---
        Arguments:
            init_method: flag refering to the init method to use e.g.
                        ('xavier', 'he')
        Returns:
            initial_parameters: python dictionnary containing the initialized
                        parameters "W1", "b1", ..., "WL", "bL".
                        initial_parameters["W1"] = ...
                        initial_parameters["b1"] = ...
        """

        initial_parameters = {}

        if self.__init_method == 'xavier':
            initial_parameters = init_methods.initialize_parameters_xavier(
                self.__layers_dims)
        elif self.__init_method == 'he':
            initial_parameters = init_methods.initialize_parameters_he(
                self.__layers_dims)
        elif self.__init_method == 'random':
            initial_parameters = init_methods.initialize_parameters_random(
                self.__layers_dims)
        elif self.__init_method == 'zeros':
            initial_parameters = init_methods.initialize_parameters_zeros(
                self.__layers_dims)
        else:
            raise ValueError(
                'init_method must be either "zeros","random","he" or "xavier"')

        self.__parameters = initial_parameters

    def __initialize_optimizer(self):
        """
        Initializes the NN optimizer depending on the required algorithm.
        ---
        Arguments:
            optimizer_name: flag refering to the algorithm method to use in
                    the optimization process e.g. ('gd', 'momentum', 'rmsprop', 'adam')
        Returns:
            optimizer: python dictionnary containing the initialized
                        optimizer with its name and parameters.
                        e.g. in the case of 'momentum'
                        optimizer["name"] = 'momentum'
                        optimizer["v"] = [...] # velocity
        """
        optimizer = {}

        if self.__optimizer_name == 'gd':
            optimizer['name'] = 'gd'

        elif self.__optimizer_name == 'momentum':
            optimizer['name'] = 'momentum'
            optimizer['v'] = optimization.initialize_momentum(
                self.__parameters)

        elif self.__optimizer_name == 'rmsprop':
            optimizer['name'] = 'rmsprop'
            optimizer['s'] = optimization.initialize_rmsprop(
                self.__parameters)

        elif self.__optimizer_name == 'adam':
            optimizer['name'] = 'adam'
            v, s, t = optimization.initialize_adam(self.__parameters)
            optimizer['v'] = v
            optimizer['s'] = s
            optimizer['t'] = t

        else:
            raise ValueError(
                'optimizer_name must be either "gd","momentum","rmsprop" or "adam"')

        self.__optimizer = optimizer

    # ========================================================================
    # Forward Propagation
    # ========================================================================

    def __one_layer_forward_propagation(self, A_prev, W_current, b_current,
                                        activation_current):
        """
        Performs forward propagation for one layer only, on all the training examples.
        ---
        Arguments:
            A_prev: Activation from the previous layer. Numpy array of
                    shape   (# units of the previous layer, number of examples)
            W_current: Weights matrix of the current layer. Numpy array of
                    shape (# units of current layer, # units of the previous layer)
            b_current: Bias vector of the current layer. Numpy array of shape
                    (# units of current layer, 1)
            activation_current: String, the activation function of the current layer. e.g. ('relu', 'sigmoid', 'tanh')
        Returns:
            A_current: the output of the activation function of the current
                    layer. numpy array of size (# units of current layer, 1)
            cache_current: cache from the current layer. (values to be used
                    later in the backprop step.)
        """
        MLNN.__layer_tracker += 1
        if self.__track:
            print('I\'m in layer {} moving forward'.format(MLNN.__layer_tracker))
        dispatcher = {
            'sigmoid': activation_functions.sigmoid,
            'relu': activation_functions.relu,
            'tanh': activation_functions.tanh,
            'softmax': activation_functions.softmax
        }
        activation = dispatcher[activation_current]

        Z_current = np.dot(W_current, A_prev) + b_current
        A_current = activation(Z_current)
        cache_current = (A_prev, Z_current, W_current)

        return (A_current, cache_current)

    def __deep_forward_propagation(self, X):
        """
        Performs forward propagation for every layer of the NN, on all the training examples.
        ---
        Arguments:
            X: input matrix. Numpy array of shape (# of features, # of
                examples)
        Returns:
            AL: (y_hat) the output of the last layer L.
            deep_forward_caches: list of caches from every layer during the forward prop.
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

        assert (AL.shape == (self.__layers_dims[L], m))
        return (AL, deep_forward_caches)

    # ========================================================================
    # Back Propagation
    # ========================================================================

    def __one_layer_back_propagation(self, dA_current, cache_current,
                                     activation_current, *args):
        """
        Performs back propagation for one layer only, on all the training examples.
        ---
        Arguments:
         dA_current: -gradient of the current layer activations from the
                    previous layer. Numpy array of shape (# units of the previous layer, # of examples).
                    - =None in the case of softmax regression. The function waits for dZ_current instead. (in *args).
        cache_current: cache from forward prop on the current layer.
        activation_current: String, the activation function of the current
                    layer. e.g. ('relu', 'sigmoid', 'tanh')
        args: dZ_current in the case of softmax regression.
        Returns:
            dA_prev: the output of backprop on the current layer. numpy array of size (# units of previous layer, 1)
        """
        if self.__track:
            print('I\'m in layer {} moving backward.'.format(MLNN.__layer_tracker))

        A_prev, Z_current, W_current = cache_current

        # selecting the activation function

        # the case of softmax activation
        if activation_current == 'softmax':
            dZ_current = args[0]
            n, m = dZ_current.shape
        else:
            dispatcher = {
                'sigmoid': activation_functions.sigmoid_p,
                'relu': activation_functions.relu_p,
                'tanh': activation_functions.tanh_p,
            }
            activation_p = dispatcher[activation_current]
            dZ_current = dA_current * activation_p(Z_current)
            n, m = dA_current.shape

        dW_current = 1. / m * \
            np.dot(dZ_current, A_prev.T) + self.__lambd / m * W_current
        db_current = 1. / m * np.sum(dZ_current, axis=1, keepdims=True)
        dA_prev = np.dot(W_current.T, dZ_current)

        MLNN.__layer_tracker -= 1

        return (dA_prev, dW_current, db_current)

    def __deep_back_propagation(self, AL, Y, deep_forward_caches):
        """
        Performs backward propagation for every layer of the NN, on all the training examples.
        ---
        Arguments:
            AL: probability vector, output of the deep forward propagation.
            Y: true "label" vector.
            deep_forward_caches: list of caches from the forward prop
                        containing:
                      -"relu" activation caches (deep_forward_caches[l], for l = 0...L-2)
                      -"sigmoid" activation cache (deep_forward_caches[L-1])
        Returns:
         grads -- A dictionary with the gradients
                  grads["dA" + str(l)] = ...
                  grads["dW" + str(l)] = ...
                  grads["db" + str(l)] = ...
        """
        L = len(deep_forward_caches)  # total number of layers
        n, m = AL.shape  # number of training examples
        Y = Y.reshape(AL.shape)
        # C = Y.shape[0]
        grads = {}

        def write_grads(dA, dW, db, layer):
            """
            Writes gradient values to the returned dictionnary (grads).
            """
            # grads['dA' + str(layer)] = dA
            grads['dW' + str(layer)] = dW
            grads['db' + str(layer)] = db

        # Performing backprop on the last layer
        cache_current = deep_forward_caches[L - 1]

        if self.__binary_class:
            # (binary class backprop initialization)
            dA_current = -Y / AL + (1 - Y) / (
                1 - AL)  # = dAL
            dA_prev, dW_current, db_current = self.__one_layer_back_propagation(
                dA_current, cache_current, self.__activations[1])
        else:
            # (multi class backprop initialization)
            dA_current = None
            dZ_current = AL - Y
            dA_prev, dW_current, db_current = self.__one_layer_back_propagation(
                dA_current, cache_current, self.__activations[1], dZ_current)

        write_grads(dA_current, dW_current, db_current, L)

        for l in range(L - 2, -1, -1):
            # Updating the current cache and the current derivative
            cache_current = deep_forward_caches[l]
            dA_current = dA_prev

            dA_prev, dW_current, db_current = self.__one_layer_back_propagation(
                dA_current, cache_current, self.__activations[0])

            write_grads(dA_current, dW_current, db_current, l + 1)

        # Reset layer tracker
        # MLNN.__layer_tracker = 0

        return (grads)

    # ========================================================================
    # Weights update
    # ========================================================================

    def __update_parameters(self, grads):
        """
        Update the NN parameters using the required algorithm.
        --
        Arguments:
            grads: dictionary containing all the gradients
                    grads["dW" + str(l)] = ...
                    grads["db" + str(l)] = ...
        Returns:
        parameters: dictionary containing updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        # Updating the parameters depending on the required algorithm
        if self.__optimizer['name'] == 'gd':
            self.__parameters = optimization.update_parameters_gd(
                self.__parameters, grads, self.__learning_rate)

        elif self.__optimizer['name'] == 'momentum':
            v = self.__optimizer['v']
            self.__parameters, self.__optimizer['v'] = optimization.update_parameters_momentum(
                self.__parameters, grads, self.__learning_rate, v, self.__beta1)

        elif self.__optimizer['name'] == 'rmsprop':
            s = self.__optimizer['s']
            self.__parameters, self.__optimizer['s'] = optimization.update_parameters_rmsprop(
                self.__parameters, grads, self.__learning_rate, s, self.__beta1)

        elif self.__optimizer['name'] == 'adam':
            v = self.__optimizer['v']
            s = self.__optimizer['s']
            t = self.__optimizer['t'] + 1

            self.__parameters, self.__optimizer['v'], self.__optimizer['s'] = optimization.update_parameters_adam(
                self.__parameters, grads, self.__learning_rate, v, s, t, self.__beta1, self.__beta2)

        else:
            raise ValueError(
                'optimizer can be either "gd", "momentum", "rmsprop" or "adam"')

    # ========================================================================
    # Training phase
    # ========================================================================

    def train(self, X, Y):
        """
        Trains a L-layer neural network.

        Arguments:
            X: input data, numpy array of shape (# of features, # of examples)
            Y: true "label" vector. numpy array of shape (1, # of examples)
        Returns:
            MLNN: instance of the L-layer NN with the optimal parameters and
                computed costs.
        """

        # Initialization
        # Initializing the parameters
        self.__initialize_parameters()
        # Initializing the optimizer
        self.__initialize_optimizer()

        # Check minibatch_size validity
        m = X.shape[1]
        if m < self.__minibatch_size:
            raise ValueError('minibatch_size ({}) greater than total number of training examples ({}).'.format(
                self.__minibatch_size, m))
        # Array to track the costs
        costs = []

        for epoch in range(self.__num_epochs):
            # Tracking the NN evolution if required
            if self.__track:
                print('\tEpoch {}'.format(epoch))

            for minibatch_X, minibatch_Y in optimization.get_random_minibatches(X, Y, self.__minibatch_size):

                # Performing forward prop on the given minibatch
                minibatch_AL, deep_forward_caches = self.__deep_forward_propagation(
                    minibatch_X)

                # Computing the cost depending on the classification type (binary or multi-class)
                if self.__binary_class:
                    minibatch_cost = performance.compute_cost(
                        minibatch_Y, minibatch_AL, self.__parameters, self.__lambd)
                else:
                    # Extracting logits of the last layer
                    minibatch_ZL = deep_forward_caches[-1][1]
                    minibatch_cost = performance.compute_cost(
                        minibatch_Y, minibatch_AL, self.__parameters, self.__lambd, minibatch_ZL)

                # Performing backprop on the given minibatch
                minibatch_grads = self.__deep_back_propagation(
                    minibatch_AL, minibatch_Y, deep_forward_caches)

                # Updating the parameters
                self.__update_parameters(minibatch_grads)

            # Save cost each 10 epochs if num_epochs <= 1000
            if self.__num_epochs <= 1000 and epoch % 10 == 0:
                costs.append(minibatch_cost)
            # if num_epochs are greater than 1000, save costs each 100 epochs
            elif epoch % 100 == 0:
                costs.append(minibatch_cost)

            MLNN.__layer_tracker = 0  # reinit layer tracker

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
        MLNN.__layer_tracker = 0  # reinit layer tracker

        if not self.__binary_class:
            return np.atleast_2d(AL.argmax(axis=0))

        predictions = (AL >= threshold)

        return (predictions).astype(int)

    # ========================================================================
    # Getting the params and costs
    # ========================================================================

    def get_params(self):
        """
        Returns a dictionnary of the NN hyperparameters.
        layers_dims, init_method, activations, lambd, optimizer, learning_rate, num_epochs, beta1=0.9, beta2=0.999, minibatch_size=64, track=False)
        """
        params = {
            'layers_dims': self.__layers_dims,
            'binary classification': self.__binary_class,
            'init_method': self.__init_method,
            'activations': self.__activations,
            'lambda': self.__lambd,
            'optimizer': self.__optimizer['name'],
            'learning_rate': self.__learning_rate,
            'num_epochs': self.__num_epochs,
            'beta1': self.__beta1,
            'beta2': self.__beta2,
            'minibatch_size': self.__minibatch_size,
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

                thetaplus_dict = helpers.vector_to_dictionnary(
                    thetaplus, self.__layers_dims, sizes)

                thetaminus_dict = helpers.vector_to_dictionnary(
                    thetaminus, self.__layers_dims, sizes)

                self.__parameters = thetaplus_dict
                AL, deep_forward_caches = self.__deep_forward_propagation(
                    X)

                if self.__binary_class:
                    J_plus = performance.compute_cost(
                        Y, AL, thetaplus_dict, self.__lambd)
                else:
                    # Extracting logits of the last layer
                    ZL = deep_forward_caches[-1][1]
                    J_plus = performance.compute_cost(
                        Y, AL, self.__parameters, self.__lambd, ZL)

                self.__parameters = thetaminus_dict

                AL, deep_forward_caches = self.__deep_forward_propagation(
                    X)
                if self.__binary_class:
                    J_minus = performance.compute_cost(
                        Y, AL, thetaminus_dict, self.__lambd)
                else:
                    # Extracting logits of the last layer
                    ZL = deep_forward_caches[-1][1]
                    J_minus = performance.compute_cost(
                        Y, AL, self.__parameters, self.__lambd, ZL)

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
        theta, sizes = helpers.dictionnary_to_vector(
            self.__parameters)

        dtheta_backprop, sizes = helpers.dictionnary_to_vector(
            grads)
        dtheta_approx = grad_approx(Y, theta, sizes)

        compare_gradients(dtheta_backprop, dtheta_approx)

        self.__parameters = temp
