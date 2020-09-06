from ase.build import bulk
from aiida.orm import StructureData, Dict, KpointsData, Str, Code
from aiida.engine import submit

from aiida_user_addons.vworkflows.bands import BandsWorkChain

silicon = StructureData(ase=bulk('Si', 'diamond', 5.43, 5.43, 5.43))
builder = BandsWorkChain.get_builder()

builder.structure = silicon
builder.kpoints = KpointsData()
builder.kpoints.set_cell(silicon.cell)
builder.kpoints.set_kpoints_mesh((4, 4, 4))

builder.parameters = Dict(dict={
    'vasp':{
        'encut': 300,
        'prec': 'accurate'
    }
})

builder.potential_family = Str('PBE.54')
builder.potential_mapping = Dict(dict={'Si': 'Si'})
builder.options = Dict(dict={
    'resources': {
        'tot_num_mpiprocs': 1,
        'num_machines': 1,
    }
})

builder.code = Code.get_from_string('vasp@localhost')

submit(builder)