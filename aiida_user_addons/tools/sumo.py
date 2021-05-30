"""
Use sumo to plot the AiiDA BandsData
"""

import warnings
import numpy as np
from contextlib import contextmanager
import logging

from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Spin
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen import Lattice

from sumo.plotting.bs_plotter import SBSPlotter
from sumo.electronic_structure.dos import load_dos
from sumo.plotting import dos_plotter
from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter
from castepxbin import compute_pdos
from pymatgen.electronic_structure.dos import Dos, CompleteDos
from pymatgen.io.ase import AseAtomsAdaptor

from aiida.orm import BandsData, StructureData
from aiida_castep.utils import sort_atoms_castep

from aiida_user_addons.tools.vasp import pmg_vasprun


def get_sumo_dos_plotter(scf_node, **kwargs):
    """
    Get density of state by raeding directly from the vasprun.xml file

    Args:
        scf_node (ProcessNode): A node with `retrieved` output attached.
        kwargs: additional parameters passed to `load_dos` function from sumo

    Returns:
        A `SDOSPlotter` object to be used for plotting the density of states.
    """
    vasprun, _ = pmg_vasprun(scf_node, parse_outcar=False)
    tdos, pdos = load_dos(vasprun, **kwargs)
    dp = dos_plotter.SDOSPlotter(tdos, pdos)
    return dp


def get_pmg_bandstructure(bands_node, structure=None, efermi=None, **kwargs):
    """
    Return a pymatgen `BandStructureSymmLine` object from BandsData

    Arguments:
        bands_node: A BandsData object
        structure (optionsal): a StructureData object, required if `bands_node`
          does not have information about the cell.
        efermi (float): Explicit value of the fermi energy.

    Returns:
        A `BandStructureSymmLine` object
    """
    if not isinstance(bands_node, BandsData):
        raise ValueError('The input argument must be a BandsData')
    # Load the data
    bands = bands_node.get_array('bands')  # In (num_spin, kpoints, bands) or just (kpoints, bands)
    kpoints = bands_node.get_array('kpoints')  # in (num_kpoints, 3)
    try:
        occupations = bands_node.get_array('occupations')
    except (KeyError, AttributeError):
        occupations = None

    try:
        efermi_raw = bands_node.get_attribute('efermi')
    except (KeyError, AttributeError):
        efermi_raw = None

    if efermi:
        efermi_raw = efermi

    labels = bands_node.get_attribute('labels')
    label_numbers = bands_node.get_attribute('label_numbers')

    # Construct the band_dict
    bands_shape = bands.shape
    if len(bands_shape) == 3:
        if bands_shape[0] == 2:
            bands_dict = {
                Spin.up: bands[0].T,  # Have to be (bands, kpoints)
                Spin.down:
                    bands[1].T  # Have to be (bands, kpoints)
            }
        else:
            bands_dict = {
                Spin.up: bands[0].T,  # Have to be (bands, kpoints)
            }
    else:
        bands_dict = {Spin.up: bands.T}

    if 'cell' in bands_node.attributes_keys():
        lattice = Lattice(bands_node.get_attribute('cell'))
    else:
        lattice = Lattice(structure.cell)

    # Constructure the label dictionary
    labels_dict = {}
    for label, number in zip(make_latex_labels(labels), label_numbers):
        labels_dict[label] = kpoints[number]

    # get the efermi
    if efermi_raw is None:
        if occupations is not None:
            efermi = find_vbm(bands, occupations)
        else:
            efermi = 0
            warnings.warn('Cannot find fermi energy - setting it to 0, this is probably wrong!')
    else:
        efermi = efermi_raw

    bands_structure = BandStructureSymmLine(kpoints,
                                            bands_dict,
                                            lattice.reciprocal_lattice,
                                            efermi=efermi,
                                            labels_dict=labels_dict,
                                            **kwargs)
    return bands_structure


def get_sumo_bands_plotter(bands, efermi=None, structure=None, **kwargs):
    """
    Return a sumo `SBSPlotter` object

    Arguments:
        bands_node: A BandsData object
        structure (optionsal): a StructureData object, required if `bands_node`
          does not have information about the cell.
        efermi (float): Explicit value of the fermi energy.

    Returns:
        A `SBSPlotter` object
    """
    bands_structure = get_pmg_bandstructure(bands, efermi=efermi, **kwargs)
    return SBSPlotter(bands_structure)


def find_vbm(bands, occupations, tol=1e-4):
    """
    Find the fermi energy, put it at the top of VBM
    NOTE: this differs from the fermi energy reported in VASP when there is any
    electronic smearing.
    """
    return bands[occupations > tol].max()


def make_latex_labels(labels: list) -> list:
    """Convert labels to laxtex style"""
    label_mapping = {
        'GAMMA': r'\Gamma',
        'LAMBDA': r'\Lambda',
        'SIGMA': r'\Sigma',
    }
    out_labels = []
    for label in labels:
        for tag, replace in label_mapping.items():
            if tag in label:
                label = label.replace(tag, replace)
                break
        out_labels.append(f'{label}')
    return out_labels


def get_pymatgen_phonon_bands(band_structure: BandsData, input_structure: StructureData, has_nac=False) -> PhononBandStructureSymmLine:
    """
    Obtain a pymatgen phonon bandstructure plotter
    """
    qpoints = band_structure.get_kpoints()
    freq = np.transpose(band_structure.get_bands())  # Pymatgen uses (3 * natoms, number qpoints) for frequency
    structure = input_structure.get_pymatgen()
    lattice = structure.lattice.reciprocal_lattice
    idx, labels = zip(*band_structure.labels)
    labels = make_latex_labels(labels)
    labels_dict = {label: qpoints[idx] for idx, label in zip(idx, labels)}
    pbs = PhononBandStructureSymmLine(qpoints, freq, lattice, labels_dict=labels_dict, structure=structure, has_nac=has_nac)
    return pbs


def get_sumo_phonon_plotter(band_structure: BandsData, input_structure: StructureData, has_nac=False, imag_tol=-5e-2) -> SPhononBSPlotter:
    """
    Obtain a sumo phonon plotter object
    """
    bs = get_pymatgen_phonon_bands(band_structure, input_structure, has_nac)
    return SPhononBSPlotter(bs, imag_tol)


####### Routines for CASTEP  ########


def read_dos_castep(calculation_node,
                    bin_width=0.01,
                    gaussian=None,
                    padding=None,
                    emin=None,
                    emax=None,
                    efermi_to_vbm=True,
                    lm_orbitals=None,
                    elements=None,
                    atoms=None,
                    total_only=False):
    """Convert DOS data from CASTEP .bands file to Pymatgen/Sumo format

    The data is binned into a regular series using np.histogram

    Args:
        calculation_node: The calculation node to be processed
        bin_width (:obj:`float`, optional): Spacing for DOS energy axis
        gaussian (:obj:`float` or None, optional): Width of Gaussian broadening
            function
        padding (:obj:`float`, optional): Energy range above and below occupied
            region. (This is not used if xmin and xmax are set.)
        emin (:obj:`float`, optional): Minimum energy value for output DOS)
        emax (:obj:`float`, optional): Maximum energy value for output DOS
        efermi_to_vbm (:obj:`bool`, optional):
            If a bandgap is detected, modify the stored Fermi energy
            so that it lies at the VBM.

    Returns:
        :obj:`pymatgen.electronic_structure.dos.Dos`
    """
    from sumo.io.castep import get_pdos
    import logging

    bands = calculation_node.outputs.output_bands
    calc_efermi = bands.get_attribute('efermi')
    eigenvalues = bands_array_to_dict(bands.get_bands())  # Eigenvalues array in (spin, kpoints, bands)
    kpoints, weights = bands.get_kpoints(also_weights=True)

    if efermi_to_vbm and not _is_metal(eigenvalues, calc_efermi):
        logging.info('Setting energy zero to VBM')
        efermi = _get_vbm(eigenvalues, calc_efermi)
    else:
        logging.info('Setting energy zero to Fermi energy')
        efermi = calc_efermi

    emin_data = min(eigenvalues[Spin.up].flatten())
    emax_data = max(eigenvalues[Spin.up].flatten())
    if Spin.down in eigenvalues:
        emin_data = min(emin_data, min(eigenvalues[Spin.down].flatten()))
        emax_data = max(emax_data, max(eigenvalues[Spin.down].flatten()))

    if padding is None and gaussian:
        padding = gaussian * 3
    elif padding is None:
        padding = 0.5

    if emin is None:
        emin = emin_data - padding
    if emax is None:
        emax = emax_data + padding

    # Shift sampling window to account for zeroing at VBM/EFermi
    emin += efermi
    emax += efermi

    bins = np.arange(emin, emax + bin_width, bin_width)
    energies = (bins[1:] + bins[:-1]) / 2

    # Add rows to weights for each band so they are aligned with eigenval data
    weights = weights * np.ones([eigenvalues[Spin.up].shape[0], 1])

    dos_data = {spin: np.histogram(eigenvalue_set, bins=bins, weights=weights)[0] for spin, eigenvalue_set in eigenvalues.items()}

    dos = Dos(efermi, energies, dos_data)

    # Now process PDOS
    retrieved = calculation_node.outputs.retrieved
    obj_names = retrieved.list_object_names()
    pdos_bin = None
    for name in obj_names:
        if name.endswith('pdos_bin'):
            pdos_bin = name

    if pdos_bin is not None and not total_only:
        with calculation_node.outputs.retrieved.open(pdos_bin, mode='rb') as pdos_file:
            pdos_raw = compute_pdos(pdos_file, eigenvalues, weights, bins)
        # Also we, need to read the structure, but have it sorted with increasing
        # atomic numbers
        if 'structure' in calculation_node.inputs:
            structure = calculation_node.inputs.structure
        else:
            structure = calculation_node.inputs.calc__structure
        # Get the PMG structure - makes sure that the structure is sorted
        pmg_structure = structure.get_pymatgen().get_sorted_structure(key=lambda x: x.species.elements[0].Z)
        pdoss = {}
        for isite, site in enumerate(pmg_structure.sites):
            pdoss[site] = pdos_raw[isite]
        # Get the pdos dictionary for potting
        pdos = get_pdos(CompleteDos(pmg_structure, dos, pdoss), lm_orbitals=lm_orbitals, elements=elements, atoms=atoms)
        # Smear the PDOS
        for orbs in pdos.values():
            for dtmp in orbs.values():
                if gaussian:
                    dtmp.densities = dtmp.get_smeared_densities(gaussian)
    else:
        pdos = {}

    if gaussian:
        dos.densities = dos.get_smeared_densities(gaussian)

    return dos, pdos


def bands_array_to_dict(bands_array):
    """
    Construct band dictionary in the pymatgen style using the band array
    stored in BandsData with AiiDA's convention
    """
    # Construct the band_dict
    bands_shape = bands_array.shape
    if len(bands_shape) == 3:
        if bands_shape[0] == 2:
            bands_dict = {
                Spin.up: bands_array[0].T,  # Have to be (bands, kpoints)
                Spin.down:
                    bands_array[1].T  # Have to be (bands, kpoints)
            }
        else:
            bands_dict = {
                Spin.up: bands_array[0].T,  # Have to be (bands, kpoints)
            }
    else:
        bands_dict = {Spin.up: bands_array.T}

    return bands_dict


def _is_metal(eigenvalues, efermi, tol=1e-5):
    # Detect if material is a metal by checking if bands cross efermi
    from itertools import chain
    for band in chain(*eigenvalues.values()):
        if np.any(band < (efermi - tol)) and np.any(band > (efermi + tol)):
            logging.info('Electronic structure appears to be a metal')
            return True

    logging.info('Electronic structure appears to have a bandgap')
    return False


def _get_vbm(eigenvalues, efermi):
    from itertools import chain
    occupied_states_by_band = (band[band < efermi] for band in chain(*eigenvalues.values()))
    return max(chain(*occupied_states_by_band))
