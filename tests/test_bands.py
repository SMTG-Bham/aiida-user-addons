from ase.build import bulk
from aiida.orm import StructureData, Dict, KpointsData, Str, Code
from aiida.engine import submit

from aiida_user_addons.vworkflows.new_bands import VaspBandsWorkChain

silicon = StructureData(ase=bulk('Si', 'diamond', 5.43, 5.43, 5.43))
builder = VaspBandsWorkChain.get_builder()

builder.structure = silicon
builder.scf.kpoints = KpointsData()
builder.scf.kpoints.set_cell(silicon.cell)
builder.scf.kpoints.set_kpoints_mesh((4, 4, 4))

builder.scf.parameters = Dict(dict={
    'vasp':{
        'encut': 300,
        'prec': 'accurate'
    }
})

builder.scf.potential_family = Str('PBE.54')
builder.scf.potential_mapping = Dict(dict={'Si': 'Si'})
builder.scf.options = Dict(dict={
    'resources': {
        'tot_num_mpiprocs': 1,
        'num_machines': 1,
    }
})

builder.scf.code = Code.get_from_string('vasp@localhost')

builder.dos.kpoints = KpointsData()
builder.dos.kpoints.set_kpoints_mesh((8, 8, 8))
builder.dos.parameters = Dict(dict={
    'bands': {'lm': True}
})
builder.dos.code = builder.scf.code
builder.dos.options = builder.scf.options

builder.dos.potential_family = Str('PBE.54')
builder.dos.potential_mapping = Dict(dict={'Si': 'Si'})

submit(builder)