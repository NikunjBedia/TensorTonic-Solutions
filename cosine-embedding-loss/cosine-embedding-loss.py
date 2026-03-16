import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1=np.asarray(x1)
    x2=np.asarray(x2)
    label=np.asarray(label)

    dp=np.sum(x1*x2,axis=-1)

    mag1=np.linalg.norm(x1,axis=-1)
    mag2=np.linalg.norm(x2,axis=-1)

    cos_sim= dp / (mag1 * mag2)

    return np.mean(np.where(label==1,1 - cos_sim, np.maximum(0,cos_sim-margin)))

    