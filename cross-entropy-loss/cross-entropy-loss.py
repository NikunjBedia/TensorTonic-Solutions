import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)

    batch_size=y_pred.shape[0]

    needed_probs=y_pred[np.arange(batch_size),y_true]

    return np.mean(-np.log(needed_probs))
        