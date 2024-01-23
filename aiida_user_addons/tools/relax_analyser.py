"""
Analyse the outcome of relax
"""

import shutil
from itertools import zip_longest
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Tuple

import aiida.orm as orm
import numpy as np
from aiida.orm.nodes.data import StructureData
from aiida.orm.nodes.process.calculation import CalcJobNode
from aiida.orm.nodes.process.workflow import WorkChainNode
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from aiida_user_addons.common.repository import open_compressed


class RelaxationAnalyser:
    """
    A class for analysing relaxation workflows
    """

    def __init__(self, node: WorkChainNode):
        """Analyse the relaxation outcome"""
        self.node = node
        self._traj = None

    @property
    def input_structure(self) -> StructureData:
        return self.node.inputs.structure

    @property
    def output_structure(self) -> StructureData:
        if self.node.is_finished_ok:
            return self.node.outputs.relax.structure
        return self.last_relax_calc.outputs.structure

    @property
    def is_converged(self) -> bool:
        return self.node.is_finished_ok

    @property
    def workchains(self) -> List[WorkChainNode]:
        return sorted(self.node.get_outgoing().all(), key=lambda x: x.node.mtime)

    @property
    def last_relax_work(self) -> WorkChainNode:
        """Return the last relaxation workchain launch"""
        last_work = None
        for triple in sorted(self.node.get_outgoing().all(), key=lambda x: x.node.id):
            # Skip if last single point
            if triple.link_label == "singlepoint":
                continue
            last_work = triple.node
        return last_work

    @property
    def last_relax_calc(self) -> CalcJobNode:
        """Last relaxation calculation - excluding the singlepoint"""
        last_calc = None
        for calc in sorted(self.last_relax_work.called, key=lambda x: x.id):
            if "structure" in calc.outputs:
                last_calc = calc
        return last_calc

    @property
    def volume_change(self) -> float:
        """Return the change in volume"""
        vol_in = self.input_structure.get_cell_volume()
        vol_out = self.output_structure.get_cell_volume()
        return vol_out / vol_in - 1

    @property
    def in_out_structures(self) -> Tuple[Atoms]:
        return self.input_structure.get_ase(), self.output_structure.get_ase()

    def load_trajectory_from_vasprun(self):
        """Load trajectory from a series of output files"""
        tempdir = Path(mkdtemp())
        traj = []
        for work in sorted(self.node.called, key=lambda x: x.mtime):
            for calc in sorted(work.called, key=lambda x: x.mtime):
                if not calc.is_finished:
                    break
                with open_compressed(calc.outputs.retrieved, "vasprun.xml", mode="rb") as handle_source:
                    with open(tempdir / "vasprun.xml", "wb") as handle_destination:
                        shutil.copyfileobj(handle_source, handle_destination)
                all_atoms = read(str(tempdir / "vasprun.xml"), index=":")
                traj.extend(all_atoms)
        self._traj = traj
        shutil.rmtree(tempdir)
        return traj

    @property
    def trajectory(self) -> List[Atoms]:
        """A list of atoms of the trajectory"""
        if self._traj is None:
            self.load_trajectory_from_vasprun()
        return self._traj

    @property
    def energies(self) -> np.ndarray:
        """Return energies of each frame"""
        return np.array([x.get_potential_energy() for x in self.trajectory])

    @property
    def forces(self) -> np.ndarray:
        """Return forces of each frame"""
        return np.stack([x.get_forces() for x in self.trajectory], axis=0)

    @property
    def maximum_forces(self) -> np.ndarray:
        """Return maximum forces of each frame"""
        return np.linalg.norm(self.forces, axis=-1).max(axis=-1)

    @property
    def energy_diff(self) -> np.ndarray:
        return self.energies - self.energies[-1]

    def plot_energy_force_conv(self, e_per_atom_thresh=1e-3):
        """
        Plot for energy/force convergence
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True)
        natoms = len(self.input_structure.sites)
        axs[0].plot(self.energy_diff)
        axs[0].set_yscale("log")
        axs[0].hlines(e_per_atom_thresh * natoms, 0, len(self.energy_diff))
        axs[1].plot(self.maximum_forces)
        axs[1].set_ylim(0, 0.1)
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Maximum force (eV/A)")
        axs[0].set_ylabel("Energy change (eV)")
        return fig


def traj_node_to_atoms(traj: orm.TrajectoryData, energies=None) -> List[Atoms]:
    """
    Convert trajectory to atoms

    traj (TrajectoryData): The TrajectoryData node to be converged
    energies (np.ndarray): Energies of each frame. This needs to be supplied separately.
    """
    symbols = traj.base.attributes.get("symbols")
    cells, positions, forces = (traj.get_array(n) for n in ["cells", "positions", "forces"])
    atoms_list = []

    if energies is None:
        energies = []

    for cell, pos, force, eng in zip_longest(cells, positions, forces, energies):
        atoms = Atoms(scaled_positions=pos, cell=cell, symbols=symbols, pbc=True)
        calc = SinglePointCalculator(atoms, forces=force, energy=eng)
        atoms.set_calculator(calc)
        atoms_list.append(atoms)

    return atoms_list
