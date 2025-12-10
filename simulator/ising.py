from classical import BaseMetropolisSimulation

class IsingModelMetropolisSimulation(BaseMetropolisSimulation):
    """Metropolis sampler for a hyper-rectangular Ising lattice with PBC."""
    def __init__(self, lattice_shape, random_state=None, use_gpu=False, h): 
        self.h = float(h)
        super().__init__(
            lattice_shape=lattice_shape,
            random_state=random_state,
            use_gpu=use_gpu,
        )

    def initialize_lattice(self):
        """Start from a random +/- 1 spin configuration."""
        self.L = self.rs.choice([-1, 1], size=self.lattice_shape)
        self.H = self.get_energy()

    def get_neighbors(self, pos):
        """Return the positions of the neighbors of a given position in the lattice."""
        neighbors = []
        for i in range(len(pos)):
            new_pos = list(pos)
            new_pos[i] += 1
            new_pos[i] %= self.L.shape[i]
            neighbors.append(tuple(new_pos))
            new_pos[i] -= 2
            new_pos[i] %= self.L.shape[i]
            neighbors.append(tuple(new_pos))
        return neighbors

    def get_energy(self):
        """Total energy with periodic boundary conditions."""
        energy = 0.0
        for axis in range(self.d):
            energy -= self.J * self.xp.sum(
                self.L * self.xp.roll(self.L, shift=-1, axis=axis)
            )
        if self.h != 0.0:
            energy -= self.h * self.xp.sum(self.L)
        return float(energy)

    def _get_delta_H(self, pos, new_val):
        """Energy delta for flipping the spin at `pos`."""
        old_spin = self.L[pos]
        if new_val == old_spin:
            return 0.0
        for neighbor in self.get_neighbors(pos):
            delta_H -= 2.0 * old_spin * (self.J * self.L[neighbor] + self.h)
        return float(delta_H)

    def _metroplis_step(self):
        """Single-site Metropolis update."""
        pos = tuple(self.rs.randint(dim) for dim in self.lattice_shape)
        delta_H = self._get_delta_H(pos, -self.L[pos])
        if delta_H <= 0.0 or (self.rs.rand() < self.xp.exp(-self.beta * delta_H)):
            self.L[pos] *= -1
            self.H += delta_H
        self.t += 1

    def sweep(self, n_sweeps: int = 1):
        """Perform `n_sweeps` full-lattice sweeps."""
        updates_per_sweep = int(np.prod(self.lattice_shape))
        for _ in range(n_sweeps * updates_per_sweep):
            self.make_step()

    def make_step(self):
        self._metropolis_step()
        self.t += 1
    
    def simulate(self, thermalize_steps, sweep_steps):
        for _ in range(thermalize_steps):
            for _ in range(sweep_steps):
                self.make_step()

    def magnetization(self):
        """Return the average magnetization per site."""
        return float(self.to_numpy(self.L).mean())

    def energy_density(self):
        """Return the energy per spin."""
        return float(self.H) / self.L.size