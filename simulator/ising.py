from classical import BaseMetropolisSimulation

class IsingModelMetropolisSimulation(BaseMetropolisSimulation):
    """Metropolis sampler for a hyper-rectangular Ising lattice with PBC."""
    def __init__(self, lattice_shape, random_state=None, use_gpu=False, h = 0.0, J = 1.0, beta=1.0): 
        self.h = float(h)
        self.J = float(J)
        self.beta = float(beta)
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

    def thermalize(self, n_thermalization: int = 1000):
        """Thermalize the system by performing `n_thermalization` steps."""
        for _ in range(n_thermalization):
            self.make_step()

    def magnetization(self):
        """Return the average magnetization per site."""
        return float(self.to_numpy(self.L).mean())

    def energy_density(self):
        """Return the energy per spin."""
        return float(self.H) / self.L.size


def generate_ising_snapshots(
    lattice_shape,
    temperature: float,
    n_samples: int,
    burn_in_sweeps: int,
    sweeps_per_sample: int,
    J: float = 1.0,
    h: float = 0.0,
    random_state=None,
    use_gpu: bool = False,
):
    """Helper that returns magnetization, energy, and spin grids."""

    beta = 1.0 / temperature
    gpu_flag = bool(use_gpu and cp is not None)
    if use_gpu and cp is None:
        warnings.warn("CuPy is not available; falling back to NumPy backend.", RuntimeWarning)
    sim = IsingModelMetropolisSimulation(
        lattice_shape=lattice_shape,
        beta=beta,
        J=J,
        h=h,
        random_state=random_state,
        use_gpu=gpu_flag,
    )

    sim.sweep(burn_in_sweeps)

    snapshots = []
    energies = []
    magnetizations = []

    for _ in range(n_samples):
        sim.sweep(sweeps_per_sample)
        snapshot = np.array(sim.to_numpy(sim.L), copy=True)
        snapshot = align_global_spin(snapshot)
        snapshots.append(snapshot)
        energies.append(sim.energy_density())
        magnetizations.append(float(snapshot.mean()))

    return {
        "snapshots": np.stack(snapshots),
        "energy_density": np.array(energies),
        "magnetization": np.array(magnetizations),
        "temperature": temperature,
    }


def _simulate_temp_batch(args):
    (
        temp,
        samples_per_temp,
        lattice_shape,
        burn_in_sweeps,
        sweeps_per_sample,
        J,
        h,
        seed,
        use_gpu,
    ) = args
    data = generate_ising_snapshots(
        lattice_shape=lattice_shape,
        temperature=temp,
        n_samples=samples_per_temp,
        burn_in_sweeps=burn_in_sweeps,
        sweeps_per_sample=sweeps_per_sample,
        J=J,
        h=h,
        random_state=seed,
        use_gpu=use_gpu,
    )
    return temp, data


def build_ising_dataset(
    temperatures: Sequence[float],
    samples_per_temp: int,
    lattice_shape: Tuple[int, ...],
    burn_in_sweeps: int,
    sweeps_per_sample: int,
    J: float = 1.0,
    h: float = 0.0,
    use_gpu: bool = False,
    seed: int | None = None,
    num_workers: int = 1,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[dict]]:
    """Generate flattened spin grids plus metadata across temperatures."""

    rng = np.random.default_rng(seed)
    flat_configs: List[np.ndarray] = []
    metadata: List[dict] = []
    start = time.time()

    temp_iter: Iterable[float] = list(temperatures)
    worker_use_gpu = use_gpu
    if use_gpu and num_workers > 1:
        warnings.warn(
            "CuPy acceleration is only supported when num_workers=1; falling back to CPU for "
            "parallel execution.",
            RuntimeWarning,
        )
        worker_use_gpu = False

    tasks = [
        (
            temp,
            samples_per_temp,
            lattice_shape,
            burn_in_sweeps,
            sweeps_per_sample,
            J,
            h,
            int(rng.integers(0, 1_000_000_000)),
            worker_use_gpu,
        )
        for temp in temp_iter
    ]

    progress = tqdm(total=len(tasks), desc="Temperatures", disable=not show_progress)

    def _consume(temp, data):
        snapshots = data["snapshots"].astype(np.int8, copy=False)
        mags = data["magnetization"]
        energies = data["energy_density"]

        for idx in range(samples_per_temp):
            flat_configs.append(snapshots[idx].reshape(-1))
            metadata.append(
                {
                    "temperature": temp,
                    "beta": 1.0 / temp,
                    "sample_idx": idx,
                    "magnetization": mags[idx],
                    "energy_density": energies[idx],
                    "lattice_shape": "x".join(str(x) for x in lattice_shape),
                    "J": J,
                    "h": h,
                    "use_gpu": bool(worker_use_gpu and cp is not None),
                }
            )
        if progress is not None:
            progress.update(1)

    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for temp, data in executor.map(_simulate_temp_batch, tasks):
                _consume(temp, data)
    else:
        for task in tasks:
            temp, data = _simulate_temp_batch(task)
            _consume(temp, data)

    if progress is not None:
        progress.close()

    duration = time.time() - start
    total = len(flat_configs)
    print(
        f"Generated {total} samples across {len(temp_iter)} temperatures "
        f"in {duration/60:.2f} minutes."
    )

    return np.stack(flat_configs), metadata