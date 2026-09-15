import stim
from typing import List, Tuple, Iterable

PauliEntry = Tuple[int, int, int]          # (x, y, p) with p in {0:X, 1:Z, 2:Y}
Chain = Iterable[PauliEntry]

_TO_BITS   = {0: (1, 0), 1: (0, 1), 2: (1, 1)}
_FROM_BITS = {(1, 0): 0, (0, 1): 1, (1, 1): 2}

def get_active_detector_coordinates(
    detection_event: List[bool], 
    dem: stim.DetectorErrorModel
) -> List[Tuple[float, float]]:
    """
    Extracts 2D (x, y) coordinates of detectors that fired (True/1) at time slice t=0.

    Parameters:
        detection_event (list or np.ndarray): Boolean array of detector outcomes for a single shot.
        dem (stim.DetectorErrorModel): The DEM instance to query coordinate metadata.

    Returns:
        active_coords (list of tuple): List of (x, y) spatial coordinates for triggered detectors.
    """
    active_coords = []
    
    for det_id, fired in enumerate(detection_event):
        if fired:
            coords = dem.get_detector_coordinates(det_id)[det_id]
            # Filter for spatial 2D detectors at time slice t=0
            if len(coords) >= 3 and coords[2] == 0:
                active_coords.append((float(coords[0]), float(coords[1])))
                
    return active_coords

def get_error_chain(
        active_detectors: List[Tuple[float, float]],) :

    """
    Returns possible error chain given syndrome : active_detectors. We deterministically choose to join all defects to the lower (smooth) boundary.

    Parameters:
        active_detectors (list of tuple): List of (x,y) spatial coordinates of triggered detectors.
    
    Returns:
        error_chain (set of tuples) : List of (x,y,p). (x,y) is spatial coordinate of qubit lying in error chain. p is type of pauli-error. In this experiment 
                                      we only have bit-flip errors so p=0. We use p=0,1,2 = {X,Z,Y} respectively.
    
    """

    x_chain = set()
    z_chain = set()
    error_chain = []

    for (det_x,det_y) in active_detectors:
        if det_x%2 == 0:
            for i in range(0,int(det_y),2):  
                path = (int(det_x), i, 0)
                x_chain.symmetric_difference_update({path})
        else:
            for i in range(0,int(det_x),2):
                path = (i, int(det_y), 1)
                z_chain.symmetric_difference_update({path})

    error_chain = add_chains(x_chain, z_chain)

    return error_chain


def get_logical_bitflips(error_chain):
    """
    Counts number of times chosen error chain crosses the L0 observable of the circuit. For stim's unrotated surface code, memory-z experiment
    we know that L0 lies along the bottom row of data qubits. Given our convention of joining error chains to the bottom boundary, the required count
    is simply the number of data qubits in our error chain lying on the x-axis.

    Returns:
    (int) :  (# of times error chain crosses L0) modulo 2
    """
    num = sum(1 for x, y, p in error_chain if y == 0 and p in (0,2))

    return num%2


def add_chains(chain_1, chain_2, *, strict: bool = False) -> List[PauliEntry]:
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
    if chain_1 == []:
        return chain_2
    elif chain_2 == []:
        return chain_1
    else:
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

