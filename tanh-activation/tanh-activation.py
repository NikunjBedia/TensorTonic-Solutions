import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x=np.asarray(x)
    a=np.exp(x)
    b=np.exp(-x)

    return (a - b)/(a + b)