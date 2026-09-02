import numpy as np
from ncon import ncon
import pytest
from src.tensor_network_utils import *



def test_mps_mpo_contraction():
    """
    Verifies if mps-mpo contraction returns another mps of correct dimension
    """

    N = 4        # Number of sites
    d = 2        # Physical dimension
    chi_mps = 3  # Input MPS bond dimension
    chi_mpo = 2  # Input MPO bond dimension
    chi_max = 5  # Max truncation bond dimension

    # Synthetic input MPS: tensors of shape (left_bond, phys_dim, right_bond)
    mps = []
    for i in range(N):
        if i == 0:
            shape = (chi_mps,d)
        elif i == N-1:
            shape = (chi_mps, d)
        else:
            shape = (chi_mps,chi_mps,d)
        mps.append(np.random.randn(*shape))

    # Synthetic input MPO: tensors of shape (left_bond, phys_out, right_bond, phys_in)
    mpo = []
    for i in range(N):
        if i == 0:
            shape = (chi_mpo,d,d)
        elif i == N-1:
            shape = (d, chi_mpo, d)
        else:
            shape = (chi_mpo,d,chi_mpo,d)
        mpo.append(np.random.randn(*shape))

    # Execute function under test
    result_mps = mps_mpo_contract(mps, mpo, chi_max)

    # 1. Output type and chain length
    assert isinstance(result_mps, list), "Result should be a list"
    assert len(result_mps) == N, f"Expected {N} tensors in MPS chain, got {len(result_mps)}"

    # 2. Tensor ranks
    for i, tensor in enumerate(result_mps):
        assert isinstance(tensor, np.ndarray), f"Tensor at index {i} is not a NumPy array"
        if i == 0:
            assert tensor.ndim == 2, f"Tensor at index 0 has rank {tensor.ndim}, expected 2"
            assert tensor.shape[0] == result_mps[i+1].shape[1], f"Bond dimensions mismatch in resulting MPS"
        elif i == N-1:
            assert tensor.ndim == 2, f"Tensor at index N-1 has rank {tensor.ndim}, expected 2"
        else:
            assert tensor.ndim == 3, f"Tensor at index {i} has rank {tensor.ndim}, expected 3"
            if i < N-2:
                assert tensor.shape[0] == result_mps[i+1].shape[1], f"Bond dimensions mismatch in resulting MPS"
            if i == N-2:
                assert tensor.shape[0] == result_mps[i+1].shape[0], f"Bond dimensions mismatch in resulting MPS"
        

def test_mps_mps_contraction():
    """
    Verifies if mps-mps contraction returns a number
    """

    N = 4        # Number of sites
    d = 2        # Physical dimension
    chi_mps = 3  # Input MPS bond dimension
    chi_max = 5  # Max truncation bond dimension

    # Synthetic input MPS: tensors of shape (left_bond, phys_dim, right_bond)
    mps1 = []
    for i in range(N):
        if i == 0:
            shape = (chi_mps,d)
        elif i == N-1:
            shape = (chi_mps, d)
        else:
            shape = (chi_mps,chi_mps,d)
        mps1.append(np.random.randn(*shape))

    # Synthetic input MPO: tensors of shape (left_bond, phys_out, right_bond, phys_in)
    mps2 = []
    for i in range(N):
        if i == 0:
            shape = (chi_mps, d)
        elif i == N-1:
            shape = (d,chi_mps)
        else:
            shape = (chi_mps,d,chi_mps)
        mps2.append(np.random.randn(*shape))

    # Execute function under test
    result = mps_mps_contract(mps1, mps2, chi_max)

    # 1. Verify output is a scalar (primitive number, np.number, or 0D/size-1 ndarray)
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.size == 1), (
        f"Expected a scalar output, but got type {type(result)} with shape {getattr(result, 'shape', None)}"
    )

    # 2. Extract item and check numerical validity
    val = np.asarray(result).item()
    assert isinstance(val, (int, float, complex, np.number)), (
        f"Contracted value {val} is not a recognized numerical type"
    )
    assert np.isfinite(val), f"Contraction result is non-finite: {val}"
    