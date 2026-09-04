import stim
import numpy as np
from .parse_syndrome import get_active_detector_coordinates, get_error_chain
from .tensor_network_utils import contract_network
from .coset_error_chains import X_coset_error_chain, Y_coset_error_chain, Z_coset_error_chain



def decoder(code, detection_events, chi):
    """
    Returns prediction of TN decoder about the occurrence of a logical X-error.

    Parameters:
    code(SurfaceCode object) : Contains information about surface code circuit.
    detection_events(List[Bool]) : List containing detection firing events in a single shot.
    chi(int) : Maximum bond dimensions to be kept during contraction of the tensor network.

    Returns:
    (bool) : True is prediction is that a logical X-error has occured
    """

    #Get coordinates of only those detectors that have fired.
    active_detection_coords = get_active_detector_coordinates(detection_events, code.dem)
    #Use coordinates of active detectors to generate an error chain representative within the identity coset.
    error_chain = get_error_chain(active_detection_coords)

    #Use above error chain to calculate the probability of identity coset.
    prob_I = contract_network(code, error_chain, chi)

    #Use a modified error chain to contain a logical X-error. This is used to calculate the probability of the X-error coset.
    X_error_chain = X_coset_error_chain(code, error_chain)
    prob_X = contract_network(code, X_error_chain, chi)

    #Use a modified error chain to contain a logical Y-error. This is used to calculate the probability of the Y-error coset.
    Y_error_chain = Y_coset_error_chain(code, error_chain)
    prob_Y = contract_network(code, Y_error_chain, chi)

    #Use a modified error chain to contain a logical Z-error. This is used to calculate the probability of the Z-error coset.
    Z_error_chain = Z_coset_error_chain(code, error_chain)
    prob_Z = contract_network(code, Z_error_chain, chi)


    noflip_prob = prob_I + prob_Z

    flip_prob = prob_X + prob_Y

    #print("I probability = ", prob_I)
    #print("X probability = ", prob_X)

    return flip_prob > noflip_prob


