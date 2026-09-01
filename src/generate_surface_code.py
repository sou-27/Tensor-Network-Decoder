import stim
import numpy as np


class SurfaceCode:
    """
    Constructs an unrotated surface code memory-Z experiment under a 2D code-capacity noise model.
    """
    def __init__(self, code_distance: int, noise_model: str, noise: float):
        self.code_distance = code_distance
        self.noise_model = noise_model.lower().replace("-", "_")
        self.noise = noise
        
        if self.noise_model not in ["depolarise", "depolarize", "bit_flip", "bitflip"]:
            raise ValueError("noise_model must be 'depolarise' or 'bit-flip'")
            
        self.circuit = self._build_code_capacity_circuit()
        self.dem = self.circuit.detector_error_model(decompose_errors=True)

    def _build_code_capacity_circuit(self) -> stim.Circuit:
        d = self.code_distance
        p = self.noise

        if "depolar" in self.noise_model:
            # Stim natively supports data-qubit-only depolarizing noise before syndrome checks
            return stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                distance=d,
                rounds=1,
                before_round_data_depolarization=p,
                after_clifford_depolarization=0,
                after_reset_flip_probability=0,
                before_measure_flip_probability=0,
            )
        else:
            # For bit-flip noise, load noiseless circuit and inject X_ERROR on data qubits
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
            data_qubits = [
                q for q, (x, y) in coords.items() 
                if int(x) % 2 == int(y) % 2
            ]

            noisy_circuit = stim.Circuit()
            for instruction in base_circuit:
                noisy_circuit.append(instruction)
                # Inject bit-flip noise immediately following data qubit initialization
                if instruction.name == "R":
                    noisy_circuit.append("X_ERROR", data_qubits, p)

            return noisy_circuit