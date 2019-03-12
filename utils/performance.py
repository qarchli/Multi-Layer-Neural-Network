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


def compute_cost(Y, AL, parameters, lambd, *args):
    """
    Computes the cost function J.
    Arguments:
     Y: true labels.
        Numpy array of shape (# of classes, # of training examples)
     AL: - probability over each predicted label.
        Numpy array of shape (# of classes, # of training examples)
        - None in the case of binary classification.
    *args: in the case of softmax regression, the function waits for logits
         to compute a numerically stable softmax loss.
    Returns:
     J: the cost
    """
    def cross_entropy_loss(Y, AL):
        """
        Computes binary classification loss.
        """
        return -np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))

    def softmax_loss(Y):
        """
        Computes multi-class classification loss L on all classes.
        """
        Z = args[0]
        b = np.max(Z, axis=0)
        stable_loss = b + np.log(np.sum(np.exp(Z - b), axis=0))
        # return -np.atleast_2d(np.sum(np.log(AL) * (Y), axis=0))
        return np.atleast_2d(stable_loss)

    m = AL.shape[1]
    C = Y.shape[0]  # number of classes

    if C == 1:
        # binary classification loss
        loss = cross_entropy_loss(Y, AL)
    else:
        # multi-class classification loss
        loss = softmax_loss(Y)

    J = np.mean(loss)

    # Adding regularization term if required
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
