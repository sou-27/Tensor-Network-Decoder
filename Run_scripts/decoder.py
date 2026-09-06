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

if __name__ == "__main__":
    main()
