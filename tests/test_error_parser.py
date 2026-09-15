import pytest
import stim
import numpy as np
from typing import Dict, Tuple
from src.generate_surface_code import *
from src.parse_syndrome import *



def test_error_chain_in_data_qubits():
    """Tests if error chains generated from SurfaceCode actually uses only data qubits"""
    code = SurfaceCode(code_distance=3, noise_model="depolarise", noise=0.1)
    circuit = code.circuit
    dem = code.dem


    sampler = circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(shots = 10, separate_observables=True)



    for event in detection_events:
        active_detector_coords = get_active_detector_coordinates(event, dem)
        error_chain = get_error_chain(active_detector_coords)

        for qx, qy, qtype in error_chain:
            assert qx%2 == qy%2, f"Error chain not contained in data qubits"