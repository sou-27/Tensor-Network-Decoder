import pytest
import stim
import numpy as np
from typing import Dict, Tuple
from src.generate_surface_code import *
from src.parse_syndrome import *







def test_error_chain_in_horizontal_qubits():


    code = SurfaceCode(code_distance=3, noise_model="depolarise", noise=0.01)
    circuit = code.circuit
    dem = code.dem


    qubit_coords: Dict[int, Tuple[float, float]] = circuit.get_final_qubit_coordinates()

    horizontal_qubits = []
    vertical_qubits = []

    # Separate data qubits into horizontal (h-nodes) and vertical (v-nodes)
    # Data qubits in Stim's unrotated surface code layout have x % 2 == y % 2
    for q_id, (x, y) in qubit_coords.items():
        x_int, y_int = int(x), int(y)

        # Skip ancilla check qubits (which have x % 2 != y % 2)
        if x_int % 2 != y_int % 2:
            continue

        # - Horizontal edges (h-nodes) connect left-right adjacent checks (x % 2 == 0, y % 2 == 0)
        # - Vertical edges (v-nodes) connect top-bottom adjacent checks (x % 2 != 0, y % 2 != 0)
        if y_int % 2 == 0:
            horizontal_qubits.append((x, y))
        else:
            vertical_qubits.append((x, y))

    sampler = circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(shots = 10, separate_observables=True)


    for event in detection_events:
        active_detector_coords = get_active_detector_coordinates(event, dem)
        error_chain = get_error_chain(active_detector_coords)

        assert(
            set(error_chain).issubset(set(horizontal_qubits))
            ),f"error chain not contained in horizontal qubits"



