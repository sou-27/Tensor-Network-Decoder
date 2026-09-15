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
        
            
        self.circuit = self._code_capacity_channel()
        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.S = self.create_S()
        self.H = self._create_H(error=(0,0))
        self.H_X = self._create_H(error=(1,0))
        self.H_Y = self._create_H(error=(1,1))
        self.H_Z = self._create_H(error=(0,1))
        self.V = self._create_V(error=(0,0))

    def _code_capacity_channel(self):
        d = self.code_distance
        noise_model = self.noise_model
        p = self.noise

        if "depolar" in noise_model:
            channel = "DEPOLARIZE1"
        else:
            channel = "X_ERROR"

        checks = [("X", s) for s in x_checks] + [("Z", s) for s in z_checks]
        m = len(checks) + 1                      # measurements per pass
        L = []
        n = d**2 + (d - 1)**2

        def pass_():
            L.append("MPP " + " ".join(term(b, s) for b, s in checks)
                        + " " + term("Z", z_logical))
        pass_()                                                  # project into code
        L.append(f"{channel}({p}) " + " ".join(map(str, range(n))))
        pass_()                                                  # re-measure
        for k in range(len(checks)):
            L.append(f"DETECTOR rec[{-2*m + k}] rec[{-m + k}]")
        L.append(f"OBSERVABLE_INCLUDE(0) rec[{-m-1}] rec[-1]")
        return stim.Circuit("\n".join(L))


    def model(self, op):
        """
        Stores i.i.d probability dfistribution for pauli operators.

        Parameters:
        code(SurfaceCode object) : Input surface code
        operator(Tuple of ints) : Encoding of which pauli operator appears (X,Z)

        Returns:
        p(float) : Probability of given operator according to noise model 
        """


        if 'depolar' in self.noise_model:
            model = {
                (0,0) : 1 - self.noise,
                (1,0) : self.noise/3,
                (0,1) : self.noise/3,
                (1,1) : self.noise/3
            }
        else:
            model = {
                (0,0) : 1 - self.noise,
                (1,0) : self.noise,
                (0,1) : 0,
                (1,1) : 0
            }

        return model[op]

    @staticmethod
    def create_S():
        """
        Creates the rank-4 tensor S to be used in the tensor network.

        Returns:
        S (np.ndarray) : Required tensor
        """

        S = np.zeros(2**4).reshape(2,2,2,2)
        for j in range(2):
            S[j,j,j,j] = 1

        return S

    def _create_H(self,error):
        """
            Creates the rank-4 tensor H to be used in the tensor network.

            Parameters:
            error (Tuple(int, int)) : Stores type of error. (0,0) -> I, (1,0) -> X, (0,1) -> Z, (1,1) -> Y
        
            Returns:
            H (np.ndarray) : Required tensor
        """

        H = np.zeros(2**4).reshape(2,2,2,2)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        op = ((j+l+error[0])%2, (i+k+error[1])%2)

                        H[i,j,k,l] = self.model(op)

        return H

    def _create_V(self,error):
        """
                Creates the rank-4 tensor V to be used in the tensor network.

                Parameters:
                error (Tuple(int, int)) : Stores type of error. (0,0) -> I, (1,0) -> X, (0,1) -> Z, (1,1) -> Y
            
                Returns:
                V (np.ndarray) : Required tensor
            """
        
        V = np.zeros(2**4).reshape(2,2,2,2)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        op = ((i+k+error[0])%2, (j+l+error[1])%2)

                        V[i,j,k,l] = self.model(op)
        return V

    def term(basis, supp):
            return "*".join(f"{basis}{q}" for q in supp)
    
