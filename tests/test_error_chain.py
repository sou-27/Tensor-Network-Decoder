import pytest
import numpy as np
import stim
from src.generate_surface_code import *
from src.parse_syndrome import *
from src.coset_error_chains import *


def test_error_chain():
    """
    Verifies if constructed error chain has same syndrome as the input syndrome.
    """
    d = 3
    noise_model = "bit-flip"
    noise = 0.3
    num_shots = 100

    code = SurfaceCode(d,noise_model,noise)

    circuit = code.circuit
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(shots = num_shots, separate_observables=True)

    for event in detection_events:
        active_coords = get_active_detector_coordinates(event, code.dem)
        error_chain = get_error_chain(active_coords)

        error_chain_detection = circuit_qubits_in_chain_flipped(d, error_chain)
        X_error_chain_detection = circuit_qubits_in_chain_flipped(d, X_coset_error_chain(code,error_chain))
        Y_error_chain_detection = circuit_qubits_in_chain_flipped(d, Y_coset_error_chain(code,error_chain))
        Z_error_chain_detection = circuit_qubits_in_chain_flipped(d, Z_coset_error_chain(code,error_chain))

        assert set(active_coords) == set(error_chain_detection), "Error chain does not reproduce observed syndrome"
        assert set(active_coords) == set(X_error_chain_detection), "X-error chain does not reproduce observed syndrome"
        assert set(active_coords) == set(Y_error_chain_detection), "Y-error chain does not reproduce observed syndrome"
        assert set(active_coords) == set(Z_error_chain_detection), "Z-error chain does not reproduce observed syndrome"



def circuit_qubits_in_chain_flipped(d,chain):
    """
    Takes an input chain, constructs a surface code circuit where the qubits on the input chain are slipped with 100% probability, and returns
    the detectors that fired in this circuit.

    Parameters:
    d (int) : Code distance
    chain (List[Tuple(int, int ,int)]) : Input chain of qubits to be flipped.

    Returns
    List[Tuple(int, int)] : Syndrome of circuit constructed from the input chain.
    """
    base_circuit = stim.Circuit.generated(
    "surface_code:unrotated_memory_z",
    distance=d,
    rounds=1,
    after_clifford_depolarization=0,
    after_reset_flip_probability=0,
    before_measure_flip_probability=0,
    )

    # In Stim's unrotated layout, data qubits sit at coordinates where x % 2 == y % 2
    coords = base_circuit.get_final_qubit_coordinates()
    data_qubits_x = [
        q for q, (x, y) in coords.items() 
        if (int(x) % 2 == int(y) % 2) and ((int(x),int(y),0) in chain)
    ]

    data_qubits_z = [
            q for q, (x, y) in coords.items() 
            if (int(x) % 2 == int(y) % 2) and ((int(x),int(y),1) in chain)
        ]

    data_qubits_y = [
            q for q, (x, y) in coords.items() 
            if (int(x) % 2 == int(y) % 2) and ((int(x),int(y),2) in chain)
        ]

    noisy_circuit = stim.Circuit()
    for instruction in base_circuit:
        noisy_circuit.append(instruction)
        # Inject bit-flip noise immediately following data qubit initialization
        if instruction.name == "R":
            for q in data_qubits_x:
                noisy_circuit.append("X_ERROR", [q], 1.0)
            for q in data_qubits_z:
                noisy_circuit.append("Z_ERROR", [q], 1.0)   
            for q in data_qubits_y:
                noisy_circuit.append("Y_ERROR", [q], 1.0)

    sampler = noisy_circuit.compile_detector_sampler()
    dem = noisy_circuit.detector_error_model(decompose_errors=True)
    detection_events, observable_flips = sampler.sample(shots = 1, separate_observables=True)

    return get_active_detector_coordinates(detection_events[0], dem) 

    