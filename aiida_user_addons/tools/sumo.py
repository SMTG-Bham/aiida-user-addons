"""
Use sumo to plot the AiiDA BandsData
"""

import warnings

from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Spin
from pymatgen import Lattice

from sumo.plotting.bs_plotter import SBSPlotter
from sumo.electronic_structure.dos import load_dos
from sumo.plotting import dos_plotter

from aiida.orm import BandsData

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


def make_latex_labels(labels):
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
