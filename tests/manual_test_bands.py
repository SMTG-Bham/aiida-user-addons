from aiida.engine import submit
from aiida.orm import Code, Dict, KpointsData, Str, StructureData
from ase.build import bulk

from aiida_user_addons.vworkflows.new_bands import VaspBandsWorkChain

silicon = StructureData(ase=bulk("Si", "diamond", 5.43, 5.43, 5.43))
builder = VaspBandsWorkChain.get_builder()

builder.structure = silicon
builder.scf.kpoints = KpointsData()
builder.scf.kpoints.set_cell(silicon.cell)
builder.scf.kpoints.set_kpoints_mesh((4, 4, 4))

builder.scf.parameters = Dict(dict={"vasp": {"encut": 300, "prec": "accurate"}})

builder.scf.potential_family = Str("PBE.54")
builder.scf.potential_mapping = Dict(dict={"Si": "Si"})
builder.scf.options = Dict(
    dict={
        "resources": {
            "tot_num_mpiprocs": 1,
            "num_machines": 1,
        }
    }
)

builder.scf.code = orm.load_code("vasp@localhost")

builder.dos_kpoints = KpointsData()
builder.dos_kpoints.set_kpoints_mesh((8, 8, 8))

submit(builder)
