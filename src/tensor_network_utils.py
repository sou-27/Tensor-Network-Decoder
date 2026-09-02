import numpy as np
import stim
from ncon import ncon
import scipy.linalg as SA

def qr(A):
    """ Performs a QR decomposition of the matrix A. The inbuilt qr module of numpy does not return a unique decomposition. This module fixes the gauge.

    Parameters
    ----------
    A : matrix
    Matrix to be decomposed

    Returns
    -------
    Q, R : matrices such that A = Q @ R, where Q is orthogonal and R is upper triangular.
    """
    Q, R = np.linalg.qr(A, mode = 'reduced')
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q * signs[np.newaxis, :]
    R = R * signs[:, np.newaxis]
    return Q, R




def contract_network(code, error_chain, chi):
    """
    Constructs tensor network corresponding to given error chain and contracts it.

    Parameters:
    code (SurfaceCode object) : Contains information about the input surface code/
    error_chain (list of tuples) : Coordinates of all data qubits at which the chosen error chain crosses.
    chi (int) : Maximum bond dimension to be kept during contraction of the tensor network.

    Returns:
    prob(float): Result of contraction of the tensor network
    """

    L = 2 * code.distance - 1
    S = code.S
    H = code.H
    H_error = code.H_error
    V = code.V
    V_error = code.V_error

    #Create initial MPS or first layer of tensor network
    state = []
    for j in range(L):
        if j%2 == 0:
            if (0,j) in error_chain:
                tensor = H_error
            else:
                tensor = H

            if j == 0:
                state.append(tensor[:,0,0,:])
            elif j == L-1:
                state.append(tensor[0,0,:,:])
            else:
                state.append(tensor[:,0,:,:])
        
        else:
            state.append(S.sum(axis = 1))

    #Create 2nd layer
    next_layer = []
    for j in range(L):
        if j%2 == 0:
            if j==0:
                next_layer.append(S.sum(axis = 2))
            elif j == L-1:
                next_layer.append(S.sum(axis = 0))
            else:
                next_layer.append(S)
        else:
            next_layer.append(V)

    state = mps_mpo_contract(state, next_layer, chi)

    #Contract through inner layers of tensor network

    for i in range(1,L-2):
        next_layer = []
        if (i+1)%2 == 0:
            #H-layer
            for j in range(L):
                if j%2 == 0:
                    if (i,j) in error_chain:
                        tensor = H_error
                    else:
                        tensor = H

                    if j == 0:
                        next_layer.append(tensor[:,:,0,:])
                    elif j == L-1:
                        next_layer.append(tensor[0,:,:,:])
                    else:
                        next_layer.append(tensor)
                else:
                    next_layer.append(S)

        else:
            # V-layer (cannot contain error chain)
            for j in range(L):
                if j%2 == 0:
                    if j==0:
                        next_layer.append(S.sum(axis = 2))
                    elif j == L-1:
                        next_layer.append(S.sum(axis = 0))
                    else:
                        next_layer.append(S)
                else:
                    next_layer.append(V)

        state = mps_mpo_contract(state,next_layer, chi)

    #Construct final layer
    next_layer = []
    for j in range(L):
        if j % 2 == 0:
            if (L-1,j) in error_chain:
                tensor = H_error
            else:
                tensor = H

            if j == 0:
                next_layer.append(tensor[:,:,0,0])
            elif j == L-1:
                next_layer.append(tensor[0,:,:,0])
            else:
                next_layer.append(tensor[:,:,:,0])
        else:
            next_layer.append(S.sum(axis = 3))

    prob = mps_mps_contract(state, next_layer, chi)

    return prob


def mps_mpo_contract(mps, mpo, chi):


    L = len(mps)
    for j in range(L):
        A = mps[j]
        B = mpo[j]
        if j == 0:
            mps[j] = ncon([A,B], [[-2,1],[-1,1,-3]]).reshape(B.shape[0] * A.shape[0], B.shape[2])
        elif j == L-1:
            mps[j] = ncon([A,B], [[-2,1], [1,-1,-3]]).reshape(B.shape[1] * A.shape[0], B.shape[2])
        else:

            mps[j] = ncon([A, B], [[-2,-4,1], [-1,1,-3,-5]]).reshape(B.shape[0] * A.shape[0], B.shape[2] * A.shape[1], B.shape[3])


    #mps = truncate_mps(mps)

    return mps



def mps_mps_contract(mps1,mps2, chi):

    L = len(mps1)

    for j in range(L):
        A = mps1[j]
        B = mps2[j]
        if j == 0:
            mps1[j] = ncon([A,B], [[-2,1],[-1,1]]).reshape(B.shape[0] * A.shape[0])
        elif j == L-1:
            mps1[j] = ncon([A,B], [[-2,1], [1,-1]]).reshape(B.shape[1] * A.shape[0])
        else:

            mps1[j] = ncon([A, B], [[-2,-4,1], [-1,1,-3]]).reshape(B.shape[0] * A.shape[0], B.shape[2] * A.shape[1])

    #mps1 = truncate_mps(mps1)

    overlap = ncon([mps1[0], mps1[1]], [[1],[-1,1]])
    for j in range(1,L-2):
        overlap = ncon([overlap, mps1[j+1]], [[1], [-1,1]])

    overlap = ncon([overlap, mps1[L-1]], [[1], [1]])

    return overlap
