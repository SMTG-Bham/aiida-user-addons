"""
Redesigned convergence testing workchain.

Make it more simple and easy to use.

Inputs
- just like a normal VaspWorkChain
- run with different cut off energies
- run with different k spacing
- summarise the results

No added wrapper etc.
"""

import aiida.orm as orm
from aiida.engine import WorkChain, append_, calcfunction
from aiida.common.extendeddicts import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.common.utils import classproperty

from aiida_user_addons.common.opthold import OptionHolder, typed_field


class VaspConvergenceWorkChain(WorkChain):
    """
    A workchain to perform convergence tests.
    """

    _sub_workchain_string = 'vaspu.vasp'
    _sub_workchain = WorkflowFactory(_sub_workchain_string)

    @classmethod
    def define(cls, spec):
        super(VaspConvergenceWorkChain, cls).define(spec)

        spec.expose_inputs(cls._sub_workchain)
        spec.input('conv_settings', help='Settings of the workchain', valid_type=orm.Dict)
        spec.outline(cls.setup, cls.launch_conv_calcs, cls.analyse)

        spec.exit_code(401, 'ERROR_SUBWORKFLOW_ERRORED', message='At leaste one of the launched sub-workchain has failed')
        spec.output('kpoints_conv_data')
        spec.output('cutoff_conv_data')

    def setup(self):
        """Setup the convergence workflow"""
        settings = self.inputs.conv_settings.get_dict()
        self.ctx.settings = settings

        # Planewave cut off energies
        cut = settings['cutoff_start']
        cutoff_list = [cut]
        while True:
            cut += settings['cutoff_step']
            if cut < settings['cutoff_stop']:
                cutoff_list.append(cut)
            else:
                cutoff_list.append(settings['cutoff_stop'])
                break

        # kpoints spacing
        spacing = settings['kspacing_start']
        kspacing_list = [spacing]
        while True:
            spacing -= settings['kspacing_step']
            if spacing > settings['kspacing_stop']:
                kspacing_list.append(spacing)
            else:
                kspacing_list.append(settings['kspacing_stop'])
                break
        self.ctx.cutoff_list = cutoff_list
        self.ctx.kspacing_list = kspacing_list

    def launch_conv_calcs(self):
        """
        Setup and launch the convergence calculations
        """
        cutoff_for_kconv = self.ctx.settings.get('cutoff_kconv', min(self.ctx.cutoff_list))
        kspacing_for_cutoffconv = orm.Float(self.ctx.settings.get('kspacing_cutconv', max(self.ctx.kspacing_list)))

        # Launch cut off energy tests
        for cut in self.ctx.cutoff_list:
            inputs = self.exposed_inputs(self._sub_workchain)
            new_param = nested_update_dict_node(inputs.parameters, {'incar': {'encut': cut}})
            inputs.parameters = new_param
            inputs.kpoints_spacing = kspacing_for_cutoffconv
            inputs.metadata.label += f' CUTCONV {cut:.2f}'
            running = self.submit(self._sub_workchain, **inputs)
            self.report(f'Submitted {running} with cut off energy {cut:.1f} eV.')
            self.to_context(cutoff_conv_workchains=append_(running))

        # Launch kpoints convergence tests
        new_param = nested_update_dict_node(inputs.parameters, {'incar': {'encut': cutoff_for_kconv}})
        for kspacing in self.ctx.kspacing_list:
            inputs = self.exposed_inputs(self._sub_workchain)
            inputs.parameters = new_param
            inputs.kpoints_spacing = kspacing
            inputs.metadata.label += f' KCONV {kspacing:.3f}'
            running = self.submit(self._sub_workchain, **inputs)
            self.report(f'Submitted {running} with kpoints spacing {kspacing:.3f}.')
            self.to_context(kpoints_conv_workchains=append_(running))

    def analyse(self):
        """
        Analyse the output of the calculations.
        Collect data to be plotted/analysed against the cut off energy and kpoints spacing
        """

        def collect_data(workchain):
            """Collect the data from workchain output"""
            output = workchain.outputs.misc.get_dict()
            data = {}
            data['maximum_force'] = output.get('maximum_force')
            data['magnetization'] = output.get('magnetization', [None])[0]
            data['maximum_stress'] = output.get('maximum_stress', None)
            data['energy'] = list(output['total_energies'].values())[0]
            return data

        def unpack(name, input_data):
            """Unpack a dict with numberical keys"""
            output_dict = {name: []}
            for key, data in input_data.items():
                output_dict[name].append(key)
                # Append values to the corresponding lists
                for key_, value in data.items():
                    if key_ not in output_dict:
                        output_dict[key_] = []
                    output_dict[key_].append(value)
            return output_dict

        exit_code = None

        cutoff_data = {}
        cutoff_miscs = {}
        for iwork, workchain in enumerate(self.ctx.cutoff_conv_workchains):

            if workchain.exit_status != 0:
                exit_code = self.exit_codes.ERROR_SUBWORKFLOW_ERRORED
                self.report(f'Skipping workchain {workchain} with exit status {workchain.exit_status} ')
                continue

            cutoff = workchain.inputs.parameters['incar']['encut']
            cutoff_data[cutoff] = collect_data(workchain)
            cutoff_data[cutoff]['mesh'] = workchain.called[0].inputs.kpoints.get_kpoints_mesh()[0]
            cutoff_miscs[f'worchain_{iwork}'] = workchain.outputs.misc

        kspacing_data = {}
        kspacing_miscs = {}
        for iwork, workchain in enumerate(self.ctx.kpoints_conv_workchains):

            if workchain.exit_status != 0:
                exit_code = self.exit_codes.ERROR_SUBWORKFLOW_ERRORED
                self.report(f'Skipping Workchain {workchain} with exit status {workchain.exit_status} ')
                continue

            spacing = float(workchain.inputs.kpoints_spacing)
            kspacing_data[spacing] = collect_data(workchain)
            kspacing_data[spacing]['mesh'] = workchain.called[0].inputs.kpoints.get_kpoints_mesh()[0]
            kspacing_miscs[f'worchain_{iwork}'] = workchain.outputs.misc

        # Calcfunction to link with the calculation output to the summary data node
        @calcfunction
        def create_links_kconv(**miscs):
            """Alias calcfunction to link summary node with miscs"""
            return orm.Dict(dict=unpack('kpoints_spacing', kspacing_data))

        @calcfunction
        def create_links_cutconv(**miscs):
            """Alias calcfunction to link summary node with miscs"""
            return orm.Dict(dict=unpack('cutoff_energy', cutoff_data))

        self.out('kpoints_conv_data', create_links_kconv(**kspacing_miscs))
        self.out('cutoff_conv_data', create_links_cutconv(**cutoff_miscs))

        return exit_code

    @classproperty
    def option_class(cls):
        return ConvOptions


class ConvOptions(OptionHolder):
    _allowed_options = ('cutoff_start', 'cutoff_stop', 'cutoff_step', 'kspacing_start', 'kspacing_stop', 'kspacing_step', 'cutoff_kconv',
                        'kspacing_cutconv')

    cutoff_start = typed_field('cutoff_start', (float,), 'The starting cut-off energy', 300)
    cutoff_stop = typed_field('cutoff_stop', (float,), 'The Final cut-off energy', 700)
    cutoff_step = typed_field('cutoff_step', (float,), 'Step size of the cut-off energy', 50)
    kspacing_start = typed_field('kspacing_start', (float,), 'The starting kspacing', 0.07)
    kspacing_stop = typed_field('kspacing_stop', (float,), 'The final kspacing', 0.02)
    kspacing_step = typed_field('kspacing_step', (float,), 'Step size of the cut-off energy', 0.01)
    cutoff_kconv = typed_field('cutoff_kconv', (float,), 'The cut-off energy used for kpoints convergence tests', 450)
    kspacing_cutconv = typed_field('kspacing_cutconv', (float,), 'The kpoints spacing used for cut-off energy convergence tests', 0.07)


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
