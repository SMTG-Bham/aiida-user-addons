"""
Use sumo to plot the AiiDA BandsData
"""

import warnings

from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Spin
from pymatgen import Lattice

from sumo.plotting.bs_plotter import SBSPlotter

from aiida.orm import BandsData


def get_pmg_bandstructure(bands_node, efermi=None):
    """Return a pmg BandStructureSymmLine object from BandsData"""
    if not isinstance(bands_node, BandsData):
        raise ValueError('The input argument must be a BandsData')
    node = bands_node
    # Load the data
    bands = node.get_array('bands')  # In (num_spin, kpoints, bands) or just (kpoints, bands)
    kpoints = node.get_array('kpoints')  # in (num_kpoints, 3)
    try:
        occupations = node.get_array('occupations')
    except (KeyError, AttributeError):
        occupations = None
    try:
        efermi_raw = node.get_attribute('efermi')
    except (KeyError, AttributeError):
        efermi_raw = None

    if efermi:
        efermi_raw = efermi

    labels = node.get_attribute('labels')
    label_numbers = node.get_attribute('label_numbers')

    # Construct the band_dict
    bands_shape = bands.shape
    if len(bands_shape) == 3:
        bands_dict = {
            Spin.up: bands[0].T,  # Have to be (bands, kpoints)
            Spin.down:
                bands[1].T  # Have to be (bands, kpoints)
        }
    else:
        bands_dict = {Spin.up: bands.T}

    lattice = Lattice(node.get_attribute('cell'))

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

    bands_structure = BandStructureSymmLine(kpoints, bands_dict, lattice.reciprocal_lattice, efermi=efermi, labels_dict=labels_dict)
    return bands_structure


def get_sumo_bands_plotter(bands, efermi=None):
    """Return the sumo bands plotter"""
    bands_structure = get_pmg_bandstructure(bands, efermi)
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
