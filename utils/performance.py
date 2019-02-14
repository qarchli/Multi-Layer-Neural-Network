import numpy as np


def compute_accuracy(y_true, y_pred):
    """
    Computes the accuracy of prediction.
    --
    Arguments:
        y_true: True labels.
        y_pred: Predicted labels.
    Returns:
        accuracy: the accuracy of the prediction. (%)
    """

    assert (y_true.shape == y_pred.shape)

    m = y_true.shape[1]  # number of training examples
    TP = np.dot(1 - y_true, 1 - y_pred.T)  # number of true positive
    TN = np.dot(y_true, y_pred.T)  # number of true negative
    accuracy = (TP + TN) / m * 100

    return (accuracy.item())

    # ========================================================================
    # Cost computing
    # ========================================================================


def compute_cost(Y, AL, parameters, lambd):
    """
    Computes the cost function J.
    Arguments:
     Y: true labels. Numpy array of shape (1, # of training examples)
     AL: probability over each predicted label. Numpy array of shape (1, # of training examples)
    Returns:
     J: the cost

    """
    m = AL.shape[1]
    J = -1 / m * np.sum(
        np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))

    if lambd != 0:
        # computing L2-regularization term
        L = int(len(parameters) / 2)
        reg_term = 0
        for l in range(1, L + 1):
            reg_term += np.nansum(np.square(parameters['W' + str(l)]))

        reg_term *= lambd / (2 * m)

        J += reg_term

    J = np.squeeze(J)

    assert (J.shape == ())
    return (J)
