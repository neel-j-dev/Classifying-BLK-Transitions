import numpy as np
from scipy.optimize import least_squares

class BaseMetropolisSimulation:
    """Base Metropolis sampler supporting custom energy calculations."""

    def __init__(self, lattice_shape, beta, J=1, random_state=None):
        self.beta = beta
        self.rs = np.random.RandomState(seed=random_state)
        self.L = self.rs.rand(*lattice_shape)
        self.lattice_shape = lattice_shape
        self.d = len(lattice_shape)
        self.t = 0
        self.J = J
        self.H = self.compute_H()

    def make_step(self):
        change_pos = tuple([self.rs.randint(_) for _ in self.lattice_shape])
        new_val = self.rs.rand()
        delta_H = self._get_delta_H(change_pos, new_val)
        if (delta_H > 0):
            if (self.rs.rand() < np.exp(-self.beta * delta_H)):
                self.L[change_pos] = new_val
                self.H += delta_H / 2
        else:
            self.L[change_pos] = new_val
            self.H += delta_H / 2
        self.t += 1

    def simulate(self, steps, iters_per_step):
        for _ in range(steps):
            for _ in range(iters_per_step):
                self.make_step()

    def compute_H(self):
        """Compute the total energy of the current lattice configuration."""
        raise NotImplementedError
    
    def _get_delta_H(self, pos, new_val):
        """Compute the change in energy for a proposed update at position `pos` to `new_val`."""
        raise NotImplementedError



class XYModelMetropolisSimulation(BaseMetropolisSimulation):
    """XY Metropolis simulation; H_matrix is valid only for 2D model."""
    def compute_H(self):
        H = 0
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                H -= np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]]))
                H -= np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]]))
                H -= np.cos(2 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j]))
                H -= np.cos(2 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j]))
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

    xy = XYModelMetropolisSimulation(lattice_shape=lattice_shape, beta=beta, J=J, random_state=random_state)

    X = np.arange(xy.L.size).reshape(xy.L.shape) % xy.L.shape[0]
    Y = (np.arange(xy.L.size).reshape(xy.L.shape) % xy.L.shape[1]).T

    U = np.cos(2 * np.pi * xy.L)
    V = np.sin(2 * np.pi * xy.L)

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

        U = np.cos(2 * np.pi * xy.L)
        V = np.sin(2 * np.pi * xy.L)

        Q.set_UVC(U,V)

        return rects, Q,

    ani = animation.FuncAnimation(fig, update_quiver, frames=steps, fargs=(rects, Q, iters_per_step, xy),
                                   interval=25, blit=False)

    print("Saving animation...")
    plt.show()

print("Generating XY model animation...")
temp = 0.3
GetXYAnimation((20, 20), 1/temp, 100, 1000, "xy_animation.mp4", J=1, random_state=None)

