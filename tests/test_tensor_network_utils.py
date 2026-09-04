import numpy as np
from ncon import ncon
import pytest
from src.tensor_network_utils import *


def mps_to_vector(mps):
    """Contracts an MPS into a dense state vector for verification."""
    L = len(mps) 
    res = ncon([mps[0],mps[1]], [[1,-3], [-1,1,-2]])
    for j in range(2,L-1):
        res = ncon([mps[j], res], [[-1,1,-2], [1, *[-k for k in range(3,j+3)]]])

    res = ncon([mps[L-1], res], [[1,-1], [1, *[-k for k in range(2,L+1)]]])
    return res.flatten()

def mpo_to_matrix(mpo):
    #Hardcoded for N = 4, d = 2

    L = len(mpo)

    result = ncon([mpo[0],mpo[1],mpo[2],mpo[3]], [[1,-4,-8], [2,-3,1,-7], [3,-2,2,-6], [-1,3,-5]]).reshape(2**4,2**4)
    return result

def test_mps_mpo_contraction_structure():
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

def test_mps_mpo_contraction_result():
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

    vec = mps_to_vector(mps)

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

    mat = mpo_to_matrix(mpo)

    # Execute function under test
    result_mps = mps_mpo_contract(mps, mpo, chi_max)
    result_vec = mps_to_vector(result_mps)

    direct_result = vec@mat

    assert np.allclose(direct_result, result_vec), f"MPO-MPS contraction does not agree"

    
def test_mps_mps_contraction_structure():
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


def test_mps_mps_contraction_result():
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

    vec1 = mps_to_vector(mps1)

    # Execute function under test
    
    result = mps_mps_contract(mps1, mps2, chi_max)

    mps2[N-1] = mps2[N-1].T
    for j in range(1,N-1):
        mps2[j] = np.transpose(mps2[j], axes = (0,2,1))

    vec2 = mps_to_vector(mps2)

    vec_result = vec1.dot(vec2)

    assert np.allclose(result, vec_result), f"Expected result = {vec_result}, got {result}"


def test_left_canonical():


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

    new_mps, R = left_canonical_mps(mps)


    assert len(mps) == len(new_mps), f"Length of output MPS does not match length of input MPS"

    # Dense vector of original MPS
    vec_orig = mps_to_vector(mps)

    for i, A in enumerate(new_mps):
        if i == 0:
            identity_check = ncon([A, np.conj(A)], [[1,-2],[1,-1]])
            expected_identity = np.eye(A.shape[1])
        elif i == N-1:
            continue
        else:
            u_bond, d_bond, r_bond = A.shape
            # Reshape to matrix M of shape (left_bond * phys_dim, right_bond)
            M = np.transpose(A, axes = (0,2,1)).reshape(u_bond * r_bond, d_bond)

            identity_check = M.conj().T @ M
            expected_identity = np.eye(d_bond)

        np.testing.assert_allclose(
            identity_check,
            expected_identity,
            atol=1e-12,
            err_msg=f"Left-canonical isometry condition failed at site {i}",
        )

        # 4. Test Property 2: State vector equality after combining canonical_mps and R
    mps_reconstructed = [t.copy() for t in new_mps]

    # Contract R into the last tensor's right bond
    if isinstance(R, np.ndarray) and R.ndim == 2:
        mps_reconstructed[0] = ncon([mps_reconstructed[0], R], [[-1,1], [1,-2]])
    else:
        mps_reconstructed[0] = mps_reconstructed[0] * R

    vec_canonical = mps_to_vector(mps_reconstructed)


    np.testing.assert_allclose(
        vec_canonical,
        vec_orig,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Reconstructed state vector does not match original MPS",
    )

def test_svd_truncation():
    """
    Verifies svd truncation structurally. Checks if bond dimensions are correctly truncated.
    """

    N = 4        # Number of sites
    d = 2        # Physical dimension
    chi_mps = 5  # Input MPS bond dimension
    chi_max = 3  # Max truncation bond dimension


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

    new_mps = truncate_mps(mps, chi_max)
    vec = mps_to_vector(new_mps)

    assert len(mps) == len(new_mps), f"Length of truncated MPS does not match length input MPS"

    for j in range(N):

        shape = new_mps[j].shape

        assert all(x <= chi_max for x in shape), f"Bond dimension above chi_max at site {j}"


    
def test_truncate_mps_exact_when_chi_large():
    """
    Verifies that truncating with chi_max >= current_chi is exact 
    and preserves the state vector up to numerical precision.
    """
    N, d, chi_mps, chi_max = 4, 2, 3, 10
    mps = []
    for i in range(N):
        if i == 0:
            shape = (chi_mps,d)
        elif i == N-1:
            shape = (chi_mps, d)
        else:
            shape = (chi_mps,chi_mps,d)
        mps.append(np.random.randn(*shape))

    vec_orig = mps_to_vector(mps)
    
    # Truncate with an oversized chi_max (no actual singular values removed)
    truncated_mps = truncate_mps(mps, chi=chi_max)
    vec_truncated = mps_to_vector(truncated_mps)

    np.testing.assert_allclose(
        vec_truncated,
        vec_orig,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Loss of fidelity occurred when chi_max was larger than initial bond dimension",
    )


def test_truncate_mps_fidelity_bound():
    """
    Verifies that truncating a state yields a valid state vector 
    whose overlap with the original state satisfies 0 < |<psi_trunc|psi_orig>|^2 <= 1.
    """
    N, d, chi_mps, chi_max = 5, 2, 6, 2
    mps = []
    for i in range(N):
        if i == 0:
            shape = (chi_mps,d)
        elif i == N-1:
            shape = (chi_mps, d)
        else:
            shape = (chi_mps,chi_mps,d)
        mps.append(np.random.randn(*shape))
    vec_orig = mps_to_vector(mps)
    
    truncated_mps = truncate_mps(mps, chi=chi_max)
    vec_truncated = mps_to_vector(truncated_mps)

    # Compute inner product / normalized overlap
    norm_orig = np.linalg.norm(vec_orig)
    norm_trunc = np.linalg.norm(vec_truncated)
    
    assert norm_trunc > 0, "Truncated state vector norm is zero"
    
    overlap = np.abs(np.vdot(vec_truncated, vec_orig)) / (norm_orig * norm_trunc)
    assert 0.0 < overlap <= 1.0 + 1e-12, f"Invalid overlap value: {overlap}"