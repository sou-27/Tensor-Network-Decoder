import numpy as np
import stim
from typing import Dict, List, Tuple, Set

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
        active_detectors: List[Tuple[float, float]],
) -> List[Tuple[float,float]]:

    """
    Returns possible error chain given syndrome : active_detectors. We deterministically choose to join all defects to the lower (smooth) boundary.

    Parameters:
        active_detectors (list of tuple): List of (x,y) spatial coordinates of triggered detectors.
    
    Returns:
        error_chain (set of tuples) : List of (x,y) spatial coordinates of data qubits lying in the path of the chosen error chain.
    
    """

    error_chain : Set[Tuple[float, float]] = set()

    for (det_x,det_y) in active_detectors:
        for i in range(0,int(det_y),2):  
            path = (int(det_x), i)
            error_chain.symmetric_difference_update({path})

    return list(error_chain)