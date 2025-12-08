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
        beta,
        J=1,
        random_state=None,
        use_gpu: bool = False,
        twist_beta_scale: float = 1.0,
    ):
        self.beta = beta
        self.use_gpu = bool(use_gpu and cp is not None)
        self.xp = cp if self.use_gpu else np
        self.rs = self.xp.random.RandomState(seed=random_state)
        self.lattice_shape = lattice_shape
        self.initialize_lattice()
        self.t = 0  # track number of Metropolis updates performed
        self.d = len(lattice_shape)
        self.J = J
        self.twist_beta_scale = twist_beta_scale
        self.H = self.compute_H()

    def initialize_lattice(self):
        """Initialize the lattice to a given initial configuration."""
        return NotImplementedError

    def make_unwinding_step(self, force_nonzero: bool = False):
        """Optional nonlocal move; subclasses may override.

        Returns a dict with acceptance metadata so callers can visualize/debug.
        """
        return {"accepted": False, "delta_wx": 0, "delta_wy": 0, "delta_H": 0.0}

    def make_step(self):
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
        self.t += 1

    def simulate(self, steps, iters_per_step):
        for _ in range(steps):
            for _ in range(iters_per_step):
                self.make_step()
            # Attempt a global winding move after each sweep (if implemented).
                # self.make_unwinding_step()

    def compute_H(self):
        """Compute the total energy of the current lattice configuration."""
        raise NotImplementedError
    
    def _get_delta_H(self, pos, new_val):
        """Compute the change in energy for a proposed update at position `pos` to `new_val`."""
        raise NotImplementedError

    def to_numpy(self, array):
        """Convert backend array to numpy (no-op for numpy backend)."""
        if self.use_gpu and cp is not None:
            return cp.asnumpy(array)
        return np.asarray(array)



class XYModelMetropolisSimulation(BaseMetropolisSimulation):
    """XY Metropolis simulation; H_matrix is valid only for 2D model."""

    def initialize_lattice(self):
        self.L = self.rs.rand(*self.lattice_shape)
        # p = self.rs.randint(0, 3)
            
        # h, w = self.lattice_shape
        # x = np.arange(w)[None, :]
        # y = np.arange(h)[:, None]
        # phase = (nu_x * x / w) + (nu_y * y / h)
        # return np.mod(phase, 1.0).astype(np.float64)

    def _energy_of_config(self, config):
        """Compute energy for a given lattice configuration (normalized spins in [0,1))."""

        angles = 2 * np.pi * config
        # Count each bond once in +x and +y directions.
        energy = -self.J * (
            self.xp.cos(angles - self.xp.roll(angles, -1, axis=1))
            + self.xp.cos(angles - self.xp.roll(angles, -1, axis=0))
        ).sum()
        return energy

    def compute_H(self):
        H = 0
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                H -= self.xp.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]]))
                H -= self.xp.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]]))
                H -= self.xp.cos(2 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j]))
                H -= self.xp.cos(2 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j]))
        return H/2 * self.J    

    def _get_delta_H(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
        return -ans * self.J

    def make_unwinding_step(self, force_nonzero: bool = False):
        """Attempt a global twist move that changes the winding numbers."""

        if len(self.lattice_shape) != 2:
            return {"accepted": False, "delta_wx": 0, "delta_wy": 0, "delta_H": 0.0}

        h, w = self.lattice_shape
        delta_wx = int(self.rs.randint(-1, 2))
        delta_wy = int(self.rs.randint(-1, 2))

        if force_nonzero and delta_wx == 0 and delta_wy == 0:
            # Retry once to avoid a no-op proposal for visualization purposes.
            delta_wx = int(self.rs.choice([-1, 1]))
            delta_wy = int(self.rs.choice([-1, 1]))

        if delta_wx == 0 and delta_wy == 0:
            return {"accepted": False, "delta_wx": 0, "delta_wy": 0, "delta_H": 0.0}

        x = self.xp.arange(w)
        y = self.xp.arange(h)
        X, Y = self.xp.meshgrid(x, y, indexing="xy")
        twist = (delta_wx * X / w) + (delta_wy * Y / h)

        proposal = (self.L + twist) % 1.0
        H_new = self._energy_of_config(proposal)
        delta_H = float(H_new - self.H)

        effective_beta = self.beta * self.twist_beta_scale
        accepted = False
        if delta_H <= 0 or float(self.rs.rand()) < np.exp(-effective_beta * delta_H):
            self.L = proposal
            self.H = H_new
            accepted = True
        self.t += 1
        return {"accepted": accepted, "delta_wx": delta_wx, "delta_wy": delta_wy, "delta_H": delta_H}

class GXYModelMetropolisSimulation(BaseMetropolisSimulation):
    """gXY Metropolis simulation."""
    def __init__(self, lattice_shape, beta, g, J=1, random_state=None):
        self.g = g
        super().__init__(lattice_shape, beta, J, random_state)

    def compute_H(self):
        H = 0
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                H -= self.g * np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]])) + (1 - self.g) * np.cos(4 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]]))
                H -= self.g * np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]])) + (1 - self.g) * np.cos(4 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]]))
                H -= self.g * np.cos(2 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j])) + (1 - self.g) * np.cos(4 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j]))
                H -= self.g * np.cos(2 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j])) + (1 - self.g) * np.cos(4 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j]))
        return H/2 * self.J    
    
    def _get_delta_H(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            ans += self.g * np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - self.g * np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val)) + (1 - self.g) * np.cos(4 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - (1 - self.g) * np.cos(4 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            ans += self.g * np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - self.g * np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val)) + (1 - self.g) * np.cos(4 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - (1 - self.g) * np.cos(4 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
        return -ans * self.J
        

def GetXYAnimation(lattice_shape, beta, steps, iters_per_step, filename, J=1, random_state=None):
    import matplotlib.pylab as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    import matplotlib

    xy = XYModelMetropolisSimulation(lattice_shape=lattice_shape, beta=beta, J=J, random_state=random_state)

    X = np.arange(xy.L.size).reshape(xy.L.shape) % xy.L.shape[0]
    Y = (np.arange(xy.L.size).reshape(xy.L.shape) % xy.L.shape[1]).T

    L_np = xy.to_numpy(xy.L)
    U = np.cos(2 * np.pi * L_np)
    V = np.sin(2 * np.pi * L_np)

    fig, ax = plt.subplots(1,1)

    rects = []
    colors = []

    for i in range(xy.L.shape[0]):
        for j in range(xy.L.shape[1]):
            rect = patches.Rectangle(xy=(i - 0.5, j - 0.5), height=1, width=1, facecolor="red")
            rects.append(rect)
            colors.append(0.1)

    rects = PatchCollection(rects)
    rects.set_clim([0, 4])
    rects.set_animated(True)
    rects.set_array(np.array(colors))
    ax.add_collection(rects)

    Q = ax.quiver(X, Y, U, V, pivot='tail', color='b', units='inches')

    ax.set_xlim(-1, xy.L.shape[0])
    ax.set_ylim(-1, xy.L.shape[1])

    def update_quiver(num, rects, Q, steps, xy):
        for _ in range(steps):
            xy.make_step()

        rects.set_array(np.array(colors))

        L_np = xy.to_numpy(xy.L)
        U = np.cos(2 * np.pi * L_np)
        V = np.sin(2 * np.pi * L_np)

        Q.set_UVC(U,V)

        return rects, Q,

    ani = animation.FuncAnimation(
        fig,
        update_quiver,
        frames=steps,
        fargs=(rects, Q, iters_per_step, xy),
        interval=25,
        blit=False,
    )

    writer = animation.PillowWriter(fps=max(1, int(1000 / 25)))
    print(f"Saving animation to {filename} ...")
    ani.save(filename, writer=writer)

    # Avoid keeping GUI open when running headless; close figure after saving.
    plt.close(fig)
    return filename

if __name__ == "__main__":
    print("Generating XY model animation...")
    temp = 0.1
    GetXYAnimation((100, 100), 1/temp, 100, 1000, "xy_animation.gif", J=1, random_state=None)
