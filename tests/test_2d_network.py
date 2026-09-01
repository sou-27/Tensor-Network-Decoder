import pytest
import stim
import numpy as np
from src.generate_surface_code import *


import pytest


def test_dem_depolar_has_no_time_like_errors():
    """Verifies that DEM error mechanisms do not span multiple time slices and only target t=0."""
    code = SurfaceCode(code_distance=3, noise_model="depolarise", noise=0.01)
    dem = code.dem

    for instruction in dem.flattened():
        if instruction.type == "error":
            # Extract target detector indices for this specific error mechanism
            det_ids = [
                target.val
                for target in instruction.targets_copy()
                if target.is_relative_detector_id()
            ]

            # Extract the time (t) coordinate for each involved detector: (x, y, t)
            time_coords = set([dem.get_detector_coordinates(d)[d][2] for d in det_ids])

            # 1. Assert no error mechanism connects detectors across different time slices
            assert (
                len(time_coords) <= 1
            ), f"Error mechanism connects multiple time slices {time_coords}: {instruction}"

            # 2. Assert all triggered detectors lie strictly on time slice t=0
            if time_coords:
                assert (
                    0 in time_coords
                ), f"Error mechanism targets time slice {time_coords} instead of t=0: {instruction}"


def test_dem_bitflip_has_no_time_like_errors():
    """Verifies that DEM error mechanisms do not span multiple time slices and only target t=0."""
    code = SurfaceCode(code_distance=3, noise_model="bitflip", noise=0.01)
    dem = code.dem

    for instruction in dem.flattened():
        if instruction.type == "error":
            # Extract target detector indices for this specific error mechanism
            det_ids = [
                target.val
                for target in instruction.targets_copy()
                if target.is_relative_detector_id()
            ]

            # Extract the time (t) coordinate for each involved detector: (x, y, t)
            time_coords = set([dem.get_detector_coordinates(d)[d][2] for d in det_ids])

            # 1. Assert no error mechanism connects detectors across different time slices
            assert (
                len(time_coords) <= 1
            ), f"Error mechanism connects multiple time slices {time_coords}: {instruction}"

            # 2. Assert all triggered detectors lie strictly on time slice t=0
            if time_coords:
                assert (
                    0 in time_coords
                ), f"Error mechanism targets time slice {time_coords} instead of t=0: {instruction}"
    