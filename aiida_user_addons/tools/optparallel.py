"""
Module for finding optimum parallelisation strategy
"""
from math import gcd, ceil
from warnings import warn


class JobScheme:
    """
    A class representing the scheme of the jobs
    """

    def __init__(self,
                 n_kpoints,
                 n_procs,
                 n_nodes=None,
                 cpus_per_node=None,
                 npw=None,
                 nbands=None,
                 ncore_within_node=True,
                 ncore_strategy='maximise',
                 wf_size_limit=1000):
        """
        Instantiate a JobScheme object

        Args:
            n_kpoints (int): Number of kpoints.
            n_procs (int): Number of processes.
            n_nodes (int): Number of nodes.
            cpus_per_node (int): Number of CPUs per node.
            npw (int): Number of plane waves.
            nbands (int): Number of bands
            ncore_within_node (bool): If True, will limit plane-wave parallelisation to be within each node.
            ncore_strategy (str): Strategy for optimise NCORE, choose from 'maximise' and 'balance'.
            wf_size_limit (float): Limit of the ideal wavefunction size per process in MB.
                This should be set to less than the actual memory limit by a good margin (eg. 500 MB) as other
                things such as the charge density, projector, and electronic solver will also occupy the memory.

        Returns:
            JobScheme: A `JobScheme` object
        """
        self.n_kpoints = n_kpoints
        self.n_procs = n_procs
        self.n_nodes = n_nodes
        self.cpus_per_node = cpus_per_node
        self.npw = npw
        self.nbands = nbands
        self.ncore_within_node = ncore_within_node
        self.ncore_strategy = ncore_strategy
        self.wf_size_limit = wf_size_limit

        self.n_kgroup = None  # KPOINT groups
        self.n_bgroup = None  # Band groups
        self.n_pgroup = None  # Plane wave groups

        self.kpar = None  # Value for the KPAR
        self.npar = None
        self.ncore = None  # Value for the ncore
        self.new_nbands = nbands  # Value for the new nbands
        self.nbands_amplification = None  # Amplification factor for the NBAND round up
        self.ncore_balance = None  # NCORE/NPAR balance factor

        self.solve_kpar()
        self.solve_ncore()

    @classmethod
    def from_dryrun(cls, dryrun_outcome, n_procs, **kwargs):
        """Construct from dryrun results"""
        kwargs['n_kpoints'] = dryrun_outcome.get('num_kpoints')
        kwargs['nbands'] = dryrun_outcome.get('num_bands')
        kwargs['npw'] = dryrun_outcome.get('num_plane_waves')
        kwargs['n_procs'] = n_procs
        return cls(**kwargs)

    def solve_kpar(self):
        """
        Solve for the optimum strategy
        """
        kpar = gcd(self.n_kpoints, self.n_procs)
        self.kpar = kpar
        # If we did not set nbands or npw, we cannot adjust KAR to avoid memory issues
        if any(map(lambda x: x is None, [self.nbands, self.npw])):
            warn('Cannot limit KAR for memory requirement without supplying both NBANDS and NPW', UserWarning)
            return kpar

        # Reduce the KPAR
        if self.size_wavefunction_per_proc > self.wf_size_limit:
            for candidate in factors(kpar):
                self.kpar = candidate
                if self.size_wavefunction_per_proc < self.wf_size_limit:
                    kpar = candidate
                    break
        if self.size_wavefunction_per_proc > self.wf_size_limit:
            warn(('Expected wavefunction size per process {} MB '
                  'is large than the limit {} MB').format(self.size_wavefunction_per_proc, self.wf_size_limit), UserWarning)
        return kpar

    @property
    def nk_per_group(self):
        return self.n_kpoints // self.kpar

    @property
    def procs_per_kgroup(self):
        return self.n_procs // self.kpar

    def solve_ncore(self):
        """
        Solve for NCORE

        The logic is that we prefer NPAR/NCORE close to one, and reject any combination that
        will result in an increase of the NBANDS more than 20%.
        Optionally, keep NCORE a factor of the number of CPUS per node - this will help keep
        the plane-wave parallelisation within each node as it is sensitive to network latency.
        """
        # Cannot solve if no nbands provided or does not know how many cpus per node
        if self.nbands is None:
            return
        if self.ncore_within_node and (self.cpus_per_node is None):
            return

        combs = []
        for ncore in factors(self.procs_per_kgroup):
            if ncore > 12:
                continue
            # Only consider ncore that is a multiple of the cpus per node
            if self.ncore_within_node and self.cpus_per_node % ncore != 0:
                continue
            npar = self.procs_per_kgroup // ncore
            new_nbands = ceil(self.nbands / npar) * npar
            factor = new_nbands / self.nbands  # Amplification factor for the ncore
            combs.append((ncore, factor, abs(ncore / npar - 1), new_nbands))  # Balance factor, the smaller the better

        combs = list(filter(lambda x: x[1] < 1.2, combs))
        if self.ncore_strategy == 'balance':
            combs.sort(key=lambda x: x[2])  # Sort by increasing balance factor
        elif self.ncore_strategy == 'maximise':
            combs.sort(key=lambda x: x[0], reverse=True)  # Sort by decreasing NCORE
        else:
            raise RuntimeError(f'NCORE strategy: <{self.ncore_strategy}> is invalid')

        # Take the maximum ncore
        ncore, factor, balance, new_nbands = combs[0]

        self.ncore = ncore
        self.npar = self.procs_per_kgroup // ncore
        self.nbands_amplification = factor
        self.new_nbands = new_nbands
        self.ncore_balance = balance
        return ncore

    @property
    def size_wavefunction(self):
        """Memory requirement for the wavefunction in MB"""
        return self.n_kpoints * self.new_nbands * self.npw * 16 / 1048576

    @property
    def size_wavefunction_per_proc(self):
        """Memory requirement for the wavefunction per process"""
        # No data distribution between K point groups
        return self.size_wavefunction / self.procs_per_kgroup


def factors(num):
    """Return all factors of a number in descending order, including the number itself"""
    result = [num]
    for i in range(num // 2 + 1, 0, -1):
        if num % i == 0:
            result.append(i)
    return result
