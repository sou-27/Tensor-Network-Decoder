import sys
from pathlib import Path

# Ensure Python can locate modules in src/ and run_scripts/
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "run_scripts"))

from run_scripts.decoder import main  # Import main from decoder.py

# Define parameter lists
noises = [0.001, 0.005, 0.01, 0.02, 0.05]
code_distance = 3

# Fixed parameters
noise_mode = "depolarizing"
chi = 6
nfail = 1000

output_file = "test_results.csv"

with open(output_file, "w") as f:
    # Write header
    f.write("code_distance,noise_mode,noise,chi,nfail,returned_float\n")

    for p in noises:
        # If main() accepts arguments directly:
        # result = main(code_distance=d, noise_mode=noise_mode, noise=p, chi=chi, nfail=nfail)

        # If main() uses argparse internally, pass CLI arguments via argv:
        sys.argv = [
            "decoder.py",
            "--code_distance", str(d),
            "--noise_mode", str(noise_mode),
            "--noise", str(p),
            "--chi", str(chi),
            "--nfail", str(nfail)
        ]
        
        # Execute main and capture return float
        result_float = main()

        # Write row to file
        f.write(f"{d},{noise_mode},{p},{chi},{nfail},{result_float}\n")
        f.flush()  # Ensures data is written immediately in case of timeout

print(f"Sweep complete. Results saved to {output_file}")