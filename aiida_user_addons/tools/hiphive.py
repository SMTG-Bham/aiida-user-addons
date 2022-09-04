"""
hiphive related tools
"""
import numpy as np
from aiida import orm
from aiida.engine import calcfunction
from ase import Atoms
from ase.build import sort
from ase.build.supercells import make_supercell
from hiphive import StructureContainer
from hiphive.input_output.phonopy import write_phonopy_fc2
from hiphive.structure_generation import generate_mc_rattled_structures
from hiphive.utilities import find_permutation
from phonopy.api_phonopy import PhonopyAtoms


def atoms2phonopy(atoms):
    """Converting ase.Atoms to PhonopyAtoms"""
    atoms_phonopy = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell.copy(),
    )
    return atoms_phonopy


def phonopy2atoms(phonon):
    """Convert PhonopyAtoms to Atoms"""
    return Atoms(
        cell=phonon.cell.copy(),
        scaled_positions=phonon.get_scaled_positions(),
        numbers=phonon.numbers.copy(),
        pbc=True,
    )


@calcfunction
def generate_rattle(prim, n_structures, cell_size, rattle_std, min_dist, **kwargs):
    """Generate rattled structure using MC"""
    atoms_ideal = sort(
        make_supercell(prim.get_ase(), cell_size.get_list())
    )  # supercell reference structure
    if "seed" in kwargs:
        seed = kwargs["seed"].value
    else:
        seed = 42
    structures = generate_mc_rattled_structures(
        atoms_ideal, n_structures.value, rattle_std.value, min_dist.value, seed=seed
    )
    out = {}
    for i, struct in enumerate(structures):
        s = orm.StructureData(ase=struct)
        out[f"structure_{i:02d}"] = s
    out[f"ideal_structure"] = orm.StructureData(ase=atoms_ideal)
    return out


class ShortRangeFitting:
    """
    Fitting the short range interactions

    This class helps to simplify the fitting of short-range interaction in ionic systems.
    """

    def __init__(self, container, born_file, supercell_mat, nac_factor=14.399652):
        """
        Instantiate ShortRangeFitting object

        Args

            container (StructureContainer): The container object used for generate fitting data.
            born_file (dirc or str): Path to the BORN file
            supercell_mat: supercell matrix
            nac_factr: Factor for converting BORN data, default is for VASP.
        """
        from phonopy.api_phonopy import Phonopy
        from phonopy.cui.load_helper import get_nac_params

        self.container = container
        prim = container.cluster_space.primitive_structure
        if isinstance(born_file, dict):
            self.nac_params = born_file
        else:
            self.nac_params = get_nac_params(
                atoms2phonopy(prim), born_filename=born_file
            )

        self.supercell_mat = supercell_mat

        self.atoms_supercell = make_supercell(prim, supercell_mat)

        unitcell_phonopy = PhonopyAtoms(
            symbols=prim.get_chemical_symbols(),
            scaled_positions=prim.get_scaled_positions(),
            cell=prim.cell.copy(),
        )

        phonon = Phonopy(
            unitcell_phonopy,
            supercell_mat.T,  # Phonopy uses a different convention for the matrix
            primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        phonon.generate_displacements()
        phonon.nac_params = self.nac_params
        supercell = phonon.supercell
        self.phonopy_supercell = supercell
        # Current factor for VASP BORN file....
        phonon.nac_params["factor"] = nac_factor  # Needed?

        phonon.set_force_constants(np.zeros((len(supercell), len(supercell), 3, 3)))
        dynmat = phonon.get_dynamical_matrix()
        dynmat.make_Gonze_nac_dataset()
        self.fc2_LR = -dynmat.get_Gonze_nac_dataset()[0]
        self.phonon = phonon

    def _rebuild_container(self):
        """
        Rebuild the container on-demand

        Ensure that the container
        """
        phonon_ref_cell = phonopy2atoms(self.phonopy_supercell)

        # Reconstruct the structure container with atoms permutated
        all_new_atoms = []
        need_reconstruct = False
        for frame in self.container:
            # Check if the supercell lattice vectors matches - sanity check
            if not np.all(frame.atoms.cell == self.atoms_supercell.cell):
                raise RuntimeError(
                    f"Mismatch in supercell lattice vectors, stored: {frame.atoms.cell}, reference: {self.atoms_supercell.cell}"
                )

            # Get the reference cell for finding the permutation
            ideal_cell = frame.atoms.copy()
            ideal_cell.positions -= frame.displacements
            perm = find_permutation(ideal_cell, phonon_ref_cell)
            # Check if permutation is needed``
            if np.all(perm == np.arange(len(perm))):
                new_atoms = frame.atoms
            else:
                new_atoms = frame.atoms[perm]
                need_reconstruct = True

            all_new_atoms.append(new_atoms)

        if need_reconstruct:
            new_container = StructureContainer(self.container.cluster_space)
            for atoms in all_new_atoms:
                new_container.add_structure(atoms)
            self.container = new_container
            return True
        return False

    def get_fit_data(self):
        """Get the fitting data with long-range forces subtracted"""
        self._rebuild_container()
        # Sanity check
        M, F = self.container.get_fit_data()
        displacements = np.array([frame.displacements for frame in self.container])
        F_LR = np.einsum("ijab,njb->nia", -self.fc2_LR, displacements).flatten()
        self.F_LR = F_LR
        F_SR = F - F_LR
        return M, F_SR

    def get_fc2(self, fcp):
        """Get the second-order force constants with LR included"""
        fc = fcp.get_force_constants(
            phonopy2atoms(self.phonopy_supercell)
        )  # Use the supercell that is created by the phonopy
        fc2 = fc.get_fc_array(2) + self.fc2_LR
        return fc2

    def write_phonopy_fc2(self, fcp, outfile):
        """Write second order force constants in phonopy format"""
        fc2 = self.get_fc2(fcp)
        write_phonopy_fc2(outfile, fc2)

    def get_fcp(self, opt):
        """Get an FCP using the fitted data"""
        from hiphive import ForceConstantPotential

        terms = [
            "seed",
            "fit_method",
            "standardize",
            "n_target_values",
            "n_parameters",
            "n_nonzero_parameters",
            "parameters_norm",
            "target_values_std",
            "rmse_train",
            "rmse_test",
            "R2_train",
            "R2_test",
            "AIC",
            "BIC",
            "train_size",
            "test_size",
        ]
        fcp = ForceConstantPotential(
            self.container.cluster_space,
            opt.parameters,
            metadata={
                "fc2_LR": self.fc2_LR,
                "summary": {
                    key: value for key, value in opt.summary.items() if key in terms
                },
            },
        )
        return fcp
