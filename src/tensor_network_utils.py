import numpy as np
import stim
import ncon
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

    state = initial_contract(state, next_layer, chi)

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

        state = contract_layers(state,next_layer, chi)

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

    prob = final_contract(state, next_layer)

    return prob


def initial_contract(mps, mpo, chi):


    return mps


def contract_layers(mps, mpo, chi):


    return mps


def final_contract(mps1,mps2):
    overlap = 0

    return overlap
