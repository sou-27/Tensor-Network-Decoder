import numpy as np
from typing import Iterable, List, Tuple

PauliEntry = Tuple[int, int, int]          # (x, y, p) with p in {0:X, 1:Z, 2:Y}
Chain = Iterable[PauliEntry]

_TO_BITS   = {0: (1, 0), 1: (0, 1), 2: (1, 1)}
_FROM_BITS = {(1, 0): 0, (0, 1): 1, (1, 1): 2}



def X_coset_error_chain(code, error_chain):
    """
    Takes an error chain and modifies it to contain a logical X-error (a cycle extending between the smooth boundaries along x = 2d - 1)

    Parameters:
    code (SurfaceCode object) : Contains relevant data about the input circuit
    error_chain (List[Tuple(int, int)]) : List containing qubits accross which the chosen error chain passes

    Returns:
    X_error_chain (List[Tuple(int,int)]) : List containing original error chain and a logical X-error
    
    """
    d = code.code_distance

    X_path = [(2*d - 2, k, 0) for k in range(0,(2*d - 1), 2)]

    X_error_chain = add_chains(X_path, error_chain)
    return X_error_chain


def Z_coset_error_chain(code, error_chain):
    """
        Takes an error chain and modifies it to contain a logical Z-error (a cycle extending between the rough boundaries along y = 2d - 1)
    
        Parameters:
        code (SurfaceCode object) : Contains relevant data about the input circuit
        error_chain (List[Tuple(int, int)]) : List containing qubits accross which the chosen error chain passes
    
        Returns:
        Z_error_chain (List[Tuple(int,int)]) : List containing original error chain and a logical Z-error
        
        """

    d = code.code_distance

    Z_path = [(k, 2*d - 2, 1) for k in range (0, (2*d - 1), 2)]

    Z_error_chain = add_chains(Z_path, error_chain)

    return Z_error_chain


def Y_coset_error_chain(code, error_chain):
    """
        Takes an error chain and modifies it to contain a logical Y-error (i.e., both logical X and Z errors)
    
        Parameters:
        code (SurfaceCode object) : Contains relevant data about the input circuit
        error_chain (List[Tuple(int, int)]) : List containing qubits accross which the chosen error chain passes
    
        Returns:
        Y_error_chain (List[Tuple(int,int)]) : List containing original error chain and a logical Y-error
        
        """

    d = code.code_distance
    Z_path = [(k, 2*d - 2, 1) for k in range (0, (2*d - 1), 2)]

    X_error_chain = X_coset_error_chain(code, error_chain)
    Y_error_chain = add_chains(X_error_chain, Z_path)

    return Y_error_chain




def add_chains(chain_1: Chain, chain_2: Chain, *, strict: bool = False) -> List[PauliEntry]:
    """Multiply two Pauli chains, ignoring global phase.

    Each chain is a list of (x, y, p) with p = 0/1/2 for X/Z/Y. Qubits appearing
    in both chains are combined by the Pauli product: identical types cancel and
    vanish from the output, differing types give the remaining third type.

    Returns a canonical (sorted, identity-free) chain, so the operation is
    commutative and associative on its output form.

    If strict, raise on a qubit listed more than once within a single chain;
    otherwise such repeats are multiplied in like any other factor.
    """
    acc: dict[Tuple[int, int], Tuple[int, int]] = {}

    for chain_idx, chain in enumerate((chain_1, chain_2)):
        seen: set[Tuple[int, int]] = set()
        for entry in chain:
            try:
                x, y, p = entry
            except (TypeError, ValueError):
                raise ValueError(
                    f"chain_{chain_idx + 1}: expected (x, y, p) triples, got {entry!r}"
                ) from None
            if p not in _TO_BITS:
                raise ValueError(
                    f"chain_{chain_idx + 1}: bad Pauli code {p!r} at ({x}, {y}); "
                    f"expected 0 (X), 1 (Z) or 2 (Y)"
                )

            key = (x, y)
            if strict:
                if key in seen:
                    raise ValueError(
                        f"chain_{chain_idx + 1}: qubit {key} listed more than once"
                    )
                seen.add(key)

            xb, zb = _TO_BITS[p]
            ax, az = acc.get(key, (0, 0))
            acc[key] = (ax ^ xb, az ^ zb)

    return sorted(
        (x, y, _FROM_BITS[bits])
        for (x, y), bits in acc.items()
        if bits != (0, 0)
    )
