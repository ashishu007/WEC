import numpy as np
from sklearn import metrics
import configs as cfg

def calc_loss(params, kfold_res, y_true):
    """To calculate the loss for each particle in swarm

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed loss which is the average of wrong classifications
    """
    
    num_classes = cfg.params_dict["num_classes"]
    num_classifs = cfg.params_dict["num_classifs"]

    W = params.reshape(num_classifs, num_classes)

    y = np.multiply(kfold_res, W)
    yi = np.argmax(np.sum(y, axis=1)/num_classes, axis=1) # Predicted after multiplying with random matrix
    
    # print(yi)
    # print(y_true)

    # This is another loss function which I think we can use, where we take the avereage of incorrect classification per class
    # and then again divide it with the number of total classes. Its equal to 1 - Macro F1 score.
    # loss = 1 - metrics.f1_score(yi, y_true, average='macro')

    # This is the 0-1 loss function, meaning the number of incorrect classification 
    # in whole data divided by number of samples in whole data.
    # Which is also euqal to error rate or 1-accuracy.
    loss = 1 - metrics.accuracy_score(yi, y_true)
    # loss = 1 - metrics.f1_score(yi, y_true, average='macro')

    return loss


def fa(x, kfold_res, y_true):
    """Higher-level method to iterate over every particle in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [calc_loss(x[i], kfold_res, y_true) for i in range(n_particles)]
    return np.array(j)


def predict(pos, X_test, y_test):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    num_classes = cfg.params_dict["num_classes"]
    num_classifs = cfg.params_dict["num_classifs"]

    # Roll-back the weights and biases
    W = pos.reshape(num_classifs, num_classes)
    
    y = np.multiply(X_test, W)
    yi = np.argmax(np.sum(y, axis=1)/num_classes, axis=1) # Predicted after multiplying with random matrix
    # yi = yii + 1

    print(set(yi))

    acc = (yi == y_test).mean()
    f1 = metrics.f1_score(yi, y_test, average="macro")

    # print("PSO F1")
    # print(metrics.f1_score(yi, y_test, average="macro"))
    
    return acc, f1
