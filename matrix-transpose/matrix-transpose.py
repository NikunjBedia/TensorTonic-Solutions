import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A=np.asarray(A)

    #A.T is simplest solution, just to implement from scratch

    r,c = A.shape

    T=np.zeros((c,r)) # reversing the shape

    for i in range(r):
        for j in range(c):
            T[j,i]=A[i,j]

    return T
    
