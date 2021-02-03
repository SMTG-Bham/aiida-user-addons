"""
WorkChain for scanning for magnetic configurations
"""
import numpy as np
import pymatgen as pmg

import aiida.orm as orm
from aiida.engine import WorkChain, calcfunction, append_
from aiida.common.extendeddicts import AttributeDict
from aiida.common.links import LinkType
from aiida.plugins import WorkflowFactory
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_user_addons.common.inputset.vaspsets import get_ldau_keys

from .common import OVERRIDE_NAMESPACE

__version__ = '0.0.1'

RELAX_WORKCHAIN = 'vaspu.relax'


class SpinEnumerateWorkChain(WorkChain):
    """
    This workchain enumerate the spin configuration for a given element, and dispatch them to
    individual RelaxWorkChains
    """
    _relax_workchain = 'vaspu.relax'

    @classmethod
    def define(cls, spec):
        """Define the class"""
        super().define(spec)

        # Expose the relaxation inputs
        spec.expose_inputs(WorkflowFactory(cls._relax_workchain), namespace='relax', exclude=('structure',))
        spec.input('structure', valid_type=orm.StructureData)
        spec.input(
            'ldau_mapping',
            serializer=to_aiida_type,
            required=False,  # I set it to be mandatory here as in most cases we will need a U
            help='Mapping for LDA+U, see `get_ldau_keys` function for details.',
            valid_type=orm.Dict)
        spec.input('moment_map', serializer=to_aiida_type, valid_type=orm.Dict, help='Mapping of the mangetic moments')
        spec.input('enum_options', serializer=to_aiida_type, valid_type=orm.Dict, help='Additional options to the Enumerator')

        spec.outline(
            cls.setup,
            cls.generate_magnetic_configurations,
            cls.run_relaxation,
            cls.result,
        )

        # The output would be summary dictionary for the relevant phases and energies
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.exit_code(501, 'ERROR_ALL_SUB_WORK_FAILED', message='All of the launched relaxations have failed.')

    def setup(self):
        """Setup the workspace"""
        self.ctx.decorated_structures = {}

    def generate_magnetic_configurations(self):
        """Generate frames of magnetic configurations"""
        structures = extend_magnetic_orderings(self.inputs.structure, self.inputs.moment_map, self.inputs.enum_options)
        self.ctx.decorated_structures = structures

    def run_relaxation(self):
        """Run the relaxations"""
        nst = len(self.ctx.decorated_structures)
        for link_name, mag_struct in self.ctx.decorated_structures.items():

            magmom = mag_struct.get_attribute('MAGMOM')
            orig = mag_struct.get_attribute('magnetic_origin')
            inputs = self.exposed_inputs(WorkflowFactory(self._relax_workchain), 'relax')
            inputs.structure = mag_struct
            # Apply the MAGMOM to the input parameters
            inputs.vasp.parameters = nested_update_dict_node(inputs.vasp.parameters, {OVERRIDE_NAMESPACE: {'magmom': magmom}})
            incar_dict = inputs.vasp.parameters.get_dict()[OVERRIDE_NAMESPACE]

            # Setup LDA+U - we cannot use the original since atoms can be reordered!!
            if 'ldau_mapping' in self.inputs:
                ldau_settings = self.inputs.ldau_mapping.get_dict()
                ldau_keys = get_ldau_keys(mag_struct, **ldau_settings)
                inputs.vasp.parameters = nested_update_dict_node(inputs.vasp.parameters, {OVERRIDE_NAMESPACE: ldau_keys})
            elif 'laduu' in incar_dict:
                raise RuntimeError('Using LDA+U but not explicity mapping given. Please set ldu_mapping input.')

            # Index of the structure
            ist = int(link_name.split('_')[-1])
            label = link_name + '_' + orig
            inputs.metadata.label = label
            inputs.metadata.call_link_label = f'relax_{ist:03d}'
            node = self.submit(WorkflowFactory(self._relax_workchain), **inputs)
            self.report('Submitted {} for structure {} out of {}'.format(node, ist, nst))
            self.to_context(workchains=append_(node))
        return

    def result(self):
        """
        Collect the summarise the results, construction a list of dictionary of records.
        """

        miscs = {}
        for workchain in self.ctx.workchains:
            link_name = workchain.get_incoming(link_type=LinkType.CALL_WORK).one().link_label
            if not workchain.is_finished_ok:
                self.report('Relaxation {} ({}) did not finished ok - not attaching the results'.format(workchain, link_name))
            else:
                miscs[link_name] = workchain.outputs.misc

        if not miscs:
            self.report('None of the work has finished ok - serious problem must occurred')
            return self.exit_codes.ERROR_ALL_SUB_WORK_FAILED
        record_node = compose_magnetic_data(**miscs)
        self.out('output_parameters', record_node)


@calcfunction
def compose_magnetic_data(**miscs):
    """Compose magnetc data from a list of misc Dict nodes"""
    Relax = WorkflowFactory(RELAX_WORKCHAIN)
    records = []
    for _, misc in miscs.items():

        # Locate the workchain node it is the relaxation that returned this MISC
        q = orm.QueryBuilder()
        q.append(Relax, project=['*'])
        q.append(orm.Node, filters={'id': misc.pk})
        workchain = q.one()[0]

        link_name = workchain.get_incoming(link_type=LinkType.CALL_WORK).one().link_label
        ist = int(link_name.split('_')[-1])

        mag_struct = workchain.inputs.structure
        mag_pstruct = mag_struct.get_pymatgen()
        num_fu = mag_pstruct.composition.get_reduced_composition_and_factor()[1]
        orig = mag_struct.get_attribute('magnetic_origin')

        misc = workchain.outputs.misc.get_dict()
        miscs[link_name] = misc  # Record the misc nodes for linking
        magnetisation = misc.get('magnetization', [])
        if magnetisation:
            total_mag = sum(magnetisation)
        else:
            total_mag = None

        entry = {
            'config_id': ist,
            'relaxation': workchain.uuid,
            'output_structure': workchain.outputs.relax__structure.uuid,
            'input_structure': mag_struct.uuid,
            'energy': misc['total_energies']['energy_no_entropy'] / num_fu,
            'origin': orig,
            'total_magnetisation': total_mag / num_fu,
            'magnetisation': magnetisation,
        }
        records.append(entry)

    # Compose a dictionary of lists from a list of dictionaries
    keys = list(records[0].keys())
    record_dict = {key: [] for key in keys}
    for record in records:
        for key in keys:
            record_dict[key].append(record[key])

    record_node = orm.Dict(dict=record_dict)
    record_node.label = 'magnetic_ordering_data'
    record_node.description = 'A node contains the magnetic ordering data'
    return record_node


@calcfunction
def extend_magnetic_orderings(struct, moment_map, enum_options):
    """
    Use pymatgen to compute all possible magnetic orderings for
    a structure.

    Arguments:
      struct: a StructureData instance
      moment_map: a Dict containing the mapping of the magnetic moments
      options: a Dict containing the reset of the options passed to the
        `MangeticStructureEnumerator`

    Returns a collection with structures containing a `MAGMOM` attribute
    for the per-site magnetisations, and a `mangetic_origin` attribute about
    the nature of the transform applied.
    """
    from pymatgen.analysis.magnetism import MagneticStructureEnumerator

    moment_map = moment_map.get_dict()
    pstruc = struct.get_pymatgen()
    kwargs = enum_options.get_dict()

    enum = MagneticStructureEnumerator(pstruc, moment_map, **kwargs)
    structs = {}
    for idx, (ptemp, orig) in enumerate(zip(enum.ordered_structures, enum.ordered_structure_origins)):
        magmom = _get_all_spins(ptemp)
        for site in ptemp.sites:
            # This is a bit of hack - I set the specie to be the element
            # This avoids AiiDA added addition Kind to reflect the spins
            site.species = site.species.elements[0].name
        astruc = orm.StructureData(pymatgen=ptemp)
        astruc.set_attribute('MAGMOM', magmom)  # Stores the magnetic moments
        astruc.set_attribute('magnetic_origin', orig)  # Stores the type of transforms - AFM/FM etc
        structs[f'out_structure_{idx:03d}'] = astruc
    return structs


def count_magnetic_ordering(struct, moment_map, **kwargs):
    """
    Use pymatgen to compute all possible magnetic orderings for
    a structure.

    Return the number of possible configurations
    """
    from pymatgen.analysis.magnetism import MagneticStructureEnumerator
    moment_map = moment_map.get_dict()
    pstruc = struct.get_pymatgen()
    enum = MagneticStructureEnumerator(pstruc, moment_map, **kwargs)
    return len(enum.ordered_structures)


def _get_all_spins(pstruc):
    """Get all the spins from pymatgen structure"""
    out_dict = []
    for site in pstruc.sites:
        if isinstance(site.specie, pmg.core.Element):
            out_dict.append(0.0)
            continue
        out_dict.append(site.specie._properties.get('spin', 0.0))
    return out_dict


def nested_update(dict_in, update_dict):
    """Update the dictionary - combine nested subdictionary with update as well"""
    for key, value in update_dict.items():
        if key in dict_in and isinstance(value, (dict, AttributeDict)):
            nested_update(dict_in[key], value)
        else:
            dict_in[key] = value
    return dict_in


def nested_update_dict_node(dict_node, update_dict):
    """Utility to update a Dict node in a nested way"""
    pydict = dict_node.get_dict()
    nested_update(pydict, update_dict)
    if pydict == dict_node.get_dict():
        return dict_node
    else:
        return orm.Dict(dict=pydict)
