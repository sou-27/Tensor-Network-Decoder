from src.generate_surface_code import *
from src.TN_decoder import decoder
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description= "Store (code_distance, noise_mode, noise, chi, nfail)")
    
    parser.add_argument('d', type=int)
    parser.add_argument('noise_model', type = str)
    parser.add_argument('noise', type = float)
    parser.add_argument('chi', type = int)
    parser.add_argument('nfail', type = int)
    #parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output files")

    args = parser.parse_args()

    #ouptut_dir = Path(args.output_dir)
    #ouptut_dir.mkdir(parents=True, exist_ok=True)

    code_distance = int(args.d)
    noise_model = str(args.noise_model)
    noise = float(args.noise)
    nfail = int(args.nfail)
    chi = int(args.chi)

    max_shots = 100000


    print(f"Starting experiment for code distance = {code_distance}, noise model =  "+noise_model+f" p = {noise}. Experiment will run until {nfail} incorrect predictions.")

    code = SurfaceCode(code_distance, noise_model, noise)
    sampler = code.circuit.compile_detector_sampler()

    fail = 0
    shot_count = 0

    while fail < nfail and shot_count < max_shots:
        detection_events, observable_flips = sampler.sample(shots = 1, separate_observables=True)

        prediction = decoder(code, detection_events[0], chi)

        if not prediction == observable_flips[0][0]:
            fail += 1

        shot_count += 1

    error_rate = fail / shot_count
    print(f"Error rate = {error_rate}")
    return error_rate

def run_shots(d, noise_model, noise, chi, shots, rng = None):
    """
    Run TN decoder for given number of shots. This involves generating the circuit, sampling it and
    then decoding it.

    Parameters:
    d (int) : code distance
    noise_model (str) : "depolarize" or "bit-flip"
    noise (float) : error probability
    chi (int) : Maximum bond dimension to be kept during tensor-network operations
    shots (int) : Number of shots to be sampled
    rng (rng object) : Used to seed any random operations.

    Returns:
    fails (int) : Number of failures of TN decoder.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    code = SurfaceCode(d, noise_model, noise)

    stim_seed = int(rng.integers(0, 2**64, dtype=np.uint64))

    sampler = code.circuit.compile_detector_sampler(seed=stim_seed)

    detection_events, observable_flips = sampler.sample(shots, separate_observables=True)
    
    predictions = [decoder(code, event, chi) for event in detection_events]

    fails = sum([1 if not predictions[j] == observable_flips[j] else 0 for j in range(shots)])

    return fails


if __name__ == "__main__":
    main()
