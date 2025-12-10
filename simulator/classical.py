import numpy as np
from scipy.optimize import least_squares

try:
    import cupy as cp  # optional GPU backend
except ImportError:  # pragma: no cover
    cp = None

class BaseMetropolisSimulation:
    """Base Metropolis sampler supporting custom energy calculations."""

    def __init__(
        self,
        lattice_shape,
        random_state=None,
        use_gpu: bool = False,
    ):
        self.use_gpu = bool(use_gpu and cp is not None)
        self.xp = cp if self.use_gpu else np
        self.rs = self.xp.random.RandomState(seed=random_state)
        self.lattice_shape = lattice_shape
        self.H = float('inf')
        self.t = 0  # track number of Metropolis updates performed
        self.d = len(lattice_shape)
        self.L = None
        self.initialize_lattice()

    def initialize_lattice(self):
        """Initialize the lattice to a given initial configuration."""
        return NotImplementedError

    def get_neighbors(self, pos):
        """Return the positions of the neighbors of a given site `pos` in the lattice."""
        raise NotImplementedError

    def make_step(self):
        """Perform a single Metropolis step."""
        return NotImplementedError

    def thermalize(self): 
        """Thermalize the system by performing a specified number of Metropolis steps."""
        return NotImplementedError

    def sweep(self, n_steps):
        return NotImplementedError

    def get_energy(self):
        """Compute the total energy of the current lattice configuration."""
        raise NotImplementedError
    
    def _get_delta_H(self, pos, new_val):
        """Compute the change in energy for a proposed update at position `pos` to `new_val`."""
        raise NotImplementedError

    def get_observables(self):
        raise NotImplementedError

    def to_numpy(self, array):
        """Convert backend array to numpy (no-op for numpy backend)."""
        if self.use_gpu and cp is not None:
            return cp.asnumpy(array)
        return np.asarray(array)

        
