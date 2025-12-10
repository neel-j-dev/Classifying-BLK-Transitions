from classical import BaseMetropolisSimulation

class XYModelMetropolisSimulation(BaseMetropolisSimulation):
    """XY Metropolis simulation; H_matrix is valid only for 2D model."""

    def initialize_lattice(self):
        self.L = self.rs.rand(*self.lattice_shape)
        self.H = self.get_energy(self.L)

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

    def get_energy(self, config):
        """Compute energy for a given lattice configuration (normalized spins in [0,1))."""
        angles = 2 * self.xp.pi * config
        # Count each bond once in +x and +y directions.
        energy = -self.J * (
            self.xp.cos(angles - self.xp.roll(angles, -1, axis=1))
            + self.xp.cos(angles - self.xp.roll(angles, -1, axis=0))
        ).sum()
        return energy

    def _get_delta_H(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        for neighbour in self.get_neighbors(pos):
            ans += self.xp.cos(
                2 * self.xp.pi * (new_val - self.L[neighbour])
            ) - self.xp.cos(
                2 * self.xp.pi * (old_val - self.L[neighbour])
            )
        return -self.J * ans

    def _metroplis_step(self):
        change_pos = tuple([self.rs.randint(_) for _ in self.lattice_shape])
        old_val = self.L[change_pos]
        delta = (self.rs.rand() - 0.5) * 2 * 0.1
        new_val = (old_val + delta) % 1.0
        delta_H = self._get_delta_H(change_pos, new_val)
        if (delta_H > 0):
            if (self.rs.rand() < self.xp.exp(-self.beta * delta_H)):
                self.L[change_pos] = new_val
                self.H += delta_H
        else:
            self.L[change_pos] = new_val
            self.H += delta_H

    def make_step(self):
        self._metroplis_step()
        self.t += 1

    def simulate(self, thermalize_steps, sweep_steps):
        for _ in range(thermalize_steps):
            for _ in range(sweep_steps):
                self.make_step()