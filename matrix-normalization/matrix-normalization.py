import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    m=np.asarray(matrix)

    if m.ndim!=2 or axis not in [0,1,None]:
        return None
    eps=1e-15

    if norm_type=='l2':
        norm = np.sqrt(np.sum(m**2,axis=axis,keepdims=True))
    elif norm_type=='l1':
        norm = np.abs(np.sum(m,axis=axis,keepdims=True))
    elif norm_type=='max':
        norm = np.max(np.abs(m),axis=axis,keepdims=True)
    else: return None

    return m/(norm+eps)

    