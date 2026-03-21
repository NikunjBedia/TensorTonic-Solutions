import numpy as np
def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.asarray(X)
    H,W = X.shape
    
    p=pool_size
    s=stride

    h_out = (H-p) // s + 1
    w_out = (W-p) // s + 1

    out = np.zeros((h_out,w_out))

    for i in range(h_out):
        for j in range(w_out):

            h_start= i * s
            h_end = h_start + p 

            w_start= j * s
            w_end = w_start + p

            out[i,j] = np.max(X[h_start:h_end, w_start:w_end])

    return out.tolist()