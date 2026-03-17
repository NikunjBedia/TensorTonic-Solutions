import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """

    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)

    e = np.abs(y_true - y_pred)

    huber = np.where(e<=delta, (0.5 * e**2), delta * (e-(0.5*delta)))

    return np.mean(huber)