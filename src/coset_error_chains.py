import numpy as np


def X_coset_error_chain(code, error_chain):
    """
    Takes an error chain and modifies it to contain a logical X-error (a cycle extending between the smooth boundaries along x = 2d - 1)

    Parameters:
    code (SurfaceCode object) : Contains relevant data about the input circuit
    error_chain (List[Tuple(int, int)]) : List containing qubits accross which the chosen error chain passes

    Returns:
    X_error_chain (List[Tuple(int,int)]) : List containing original error chain and a logical X-error
    
    """
    d = code.distance

    X_path = [(2*d - 2, k) for k in range(0,(2*d - 1), 2)]

    X_error_chain = set(X_path).symmetric_difference(set(error_chain))

    return list(X_error_chain)


def Z_coset_error_chain(code, error_chain):
    """
        Takes an error chain and modifies it to contain a logical Z-error (a cycle extending between the rough boundaries along y = 2d - 1)
    
        Parameters:
        code (SurfaceCode object) : Contains relevant data about the input circuit
        error_chain (List[Tuple(int, int)]) : List containing qubits accross which the chosen error chain passes
    
        Returns:
        Z_error_chain (List[Tuple(int,int)]) : List containing original error chain and a logical Z-error
        
        """

    d = code.distance

    Z_path = [(k, 2*d - 2) for k in range (0, (2*d - 1), 2)]

    Z_error_chain = set(Z_path).symmetric_difference((set(error_chain)))

    return list(Z_error_chain)


def Y_coset_error_chain(code, error_chain):
    """
        Takes an error chain and modifies it to contain a logical Y-error (i.e., both logical X and Z errors)
    
        Parameters:
        code (SurfaceCode object) : Contains relevant data about the input circuit
        error_chain (List[Tuple(int, int)]) : List containing qubits accross which the chosen error chain passes
    
        Returns:
        Y_error_chain (List[Tuple(int,int)]) : List containing original error chain and a logical Y-error
        
        """



    X_error_chain = X_coset_error_chain(code, error_chain)
    Y_error_chain = Z_coset_error_chain(code, X_error_chain)

    return Y_error_chain