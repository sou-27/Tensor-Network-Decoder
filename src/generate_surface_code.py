import stim
import numpy as np


def term(basis, supp):
            """
            Returns a string representing pauli operator on given support. Eg. for basis = "X"
            and supp = [1,2,3,4], returns "X1*X2*X3*X4"

            Parameters:
            basis (str) : Pauli operator to be used
            supp (List of integers) : Support of pauli operator

            Returns:
            String representing given pauli operator.
            """
            return "*".join(f"{basis}{q}" for q in supp)
    

class SurfaceCode:
    """
    Class for storing all relevant data for an unrotated memory-z surface code experiment.

    Parameters:
    code_distance (int): code distance of surface code
    noise_model (str) : We support "depolarize" or "bit-flip" noise models.
    noise (float) : Probability of error.

    Attributes
    ----------

    code_distance(int)
    noise_model("depolarize" or "bit-flip")
    noise(float)
    lattice(dict[Tuple(lattice coordinates) : Qubit index])
    circuit(stim.circuit) : Unrotated memory-z surface code circuit with given parameters.
    dem(stim.dem) : DEM for the generated circuit.
    S,H,H_X,H_Z,H_Z,V,V_Y,V_Z (np.ndarray) : rank-4 tensors to be used to construct the tensor network for the decoder.
    """
    def __init__(self, code_distance: int, noise_model: str, noise: float):
        self.code_distance = code_distance
        self.noise_model = noise_model.lower().replace("-", "_")
        self.noise = noise
        
        if self.noise_model not in ["depolarise", "depolarize", "bit_flip", "bitflip"]:
            raise ValueError("noise_model must be 'depolarise' or 'bit-flip'")

        self.lattice = self.create_lattice()
        
            
        self.circuit = self._code_capacity_channel()
        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.S = self.create_S()
        self.H = self._create_H(error=(0,0))
        self.H_X = self._create_H(error=(1,0))
        self.H_Y = self._create_H(error=(1,1))
        self.H_Z = self._create_H(error=(0,1))
        self.V = self._create_V(error=(0,0))
        self.V_X = self._create_V(error=(1,0))
        self.V_Y = self._create_V(error=(1,1))
        self.V_Z = self._create_V(error=(0,1))

    def _code_capacity_channel(self):
        """
        Constructs stim circuit according to input parameters.

        Returns:
        Required stim.circuit
        """
        d = self.code_distance
        noise_model = self.noise_model
        p = self.noise

        if "depolar" in noise_model:
            channel = "DEPOLARIZE1"
        else:
            channel = "X_ERROR"

        x_checks = self.get_xchecks()
        z_checks = self.get_zchecks()
        #We set the logical operator to be the string across the lower boundary.
        z_logical = [k for k in range(d)]

        #x/z_checks is a list of lists: [[[support of stabilizer], (coordinate of correspinding detector)]]
        checks = [("X", s) for s in x_checks] + [("Z", s) for s in z_checks]
        m = len(checks) + 1                      # measurements per pass
        L = []
        n = d**2 + (d - 1)**2

        def pass_():
            L.append("MPP " + " ".join(term(b, s[0]) for b, s in checks)
                        + " " + term("Z", z_logical))
        pass_()                                                  # project into code
        L.append(f"{channel}({p}) " + " ".join(map(str, range(n))))
        pass_()                                            # re-measure
        for k in range(len(checks)):
            L.append(f"DETECTOR({checks[k][1][1][0]},{checks[k][1][1][1]},0) rec[{-2*m + k}] rec[{-m + k}]")
        L.append(f"OBSERVABLE_INCLUDE(0) rec[{-m-1}] rec[-1]")
        return stim.Circuit("\n".join(L))


    def create_lattice(self):
        """
        Creates a dictionary to store qubit indices for lattice coordinates on unrotated surface code geometry.
        We use the same geometry as used by stim.circuits.generated(unrotated_memory_z...)
        i.e., rough boundaries on the left/right and smooth boundaries on top/bottom. Data qubits sit
        at (x,y) s.t. x%2 == y%2. We leave lattice coordinates empty where ancillas would sit.

        Returns:
        lattice_coords (dict{coordinate (tuple) : qubit index (int)})
        """
        d = self.code_distance

        coords = []

        for y in range(2*d - 1):
            for x in range(2*d - 1):
                if x%2 == y%2:
                    coords.append((x,y))

        lattice_coords = {coord:i for i,coord in enumerate(coords)}

        return lattice_coords

    def get_xchecks(self):
        """
        Find all the vertex operators for given surface code

        Returns:
        xchecks(List[List[Tuple(coordinates of qubits in vertex operator)] , Tuple(coordinates of corresponding detector)])
        """
        d = self.code_distance
        lattice_coords = self.lattice
        xchecks = []
        coords = lattice_coords.keys()
        for y in range(2*d - 1):
            for x in range(2*d - 1):
                #Vertex operators sit at x%2 != y%2, vertext operators sit at x%2 == 1
                if x%2 == 1 and y%2 == 0:
                        detector_coord = (x,y)
                        neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                        xchecks.append([[lattice_coords[neighbor] for neighbor in neighbors if neighbor in coords], detector_coord])

        return xchecks

    def get_zchecks(self):
        """
            Find all the plaquette operators for given surface code
    
            Returns:
            zchecks(List[List[Tuple(coordinates of qubits in plquette operator)] , Tuple(coordinates of corresponding detector)])
        """
        d = self.code_distance
        lattice_coords = self.lattice
        zchecks = []
        coords = lattice_coords.keys()
        for y in range(2*d - 1):
            for x in range(2*d - 1):
                #Plaquette operators sit at x%2 != y%2, vertext operators sit at x%2 == 0
                if x%2 == 0 and y%2 == 1:
                        detector_coord = (x,y)
                        neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                        zchecks.append([[lattice_coords[neighbor] for neighbor in neighbors if neighbor in coords],detector_coord])

        return zchecks

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

