"""
Collection of process functions for AiiDA, used for structure transformation
"""
import re
from typing import List, Tuple

import numpy as np
from aiida import orm
from aiida.engine import calcfunction
from aiida.orm import (
    ArrayData,
    CalcFunctionNode,
    Node,
    QueryBuilder,
    StructureData,
)
from ase import Atoms
from ase.build import sort
from ase.neb import NEB
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@calcfunction
def magnetic_structure_decorate(structure, magmom):
    """
    Create Quantum Espresso style decroated structure with
    given magnetic moments
    """
    from aiida_user_addons.common.magmapping import (
        create_additional_species,
    )

    magmom = magmom.get_list()
    assert len(magmom) == len(
        structure.sites
    ), f"Mismatch between the magmom ({len(magmom)}) and the nubmer of sites ({len(structure.sites)})."
    old_species = [
        structure.get_kind(site.kind_name).symbol for site in structure.sites
    ]
    new_species, magmom_mapping = create_additional_species(old_species, magmom)
    new_structure = StructureData()
    new_structure.set_cell(structure.cell)
    new_structure.set_pbc(structure.pbc)
    for site, name in zip(structure.sites, new_species):
        this_symbol = structure.get_kind(site.kind_name).symbol
        new_structure.append_atom(
            position=site.position, symbols=this_symbol, name=name
        )

    # Keep the label
    new_structure.label = structure.label
    return {"structure": new_structure, "mapping": orm.Dict(dict=magmom_mapping)}


@calcfunction
def magnetic_structure_dedecorate(structure, mapping):
    """
    Remove decorations of a structure with multiple names for the same specie
    given that the decoration was previously created to give different species
    name for different initialisation of magnetic moments.
    """
    from aiida_user_addons.common.magmapping import (
        convert_to_plain_list,
    )

    mapping = mapping.get_dict()
    # Get a list of decroated names
    old_species = [structure.get_kind(site.kind_name).name for site in structure.sites]
    new_species, magmom = convert_to_plain_list(old_species, mapping)

    new_structure = StructureData()
    new_structure.set_cell(structure.cell)
    new_structure.set_pbc(structure.pbc)

    for site, name in zip(structure.sites, new_species):
        this_symbol = structure.get_kind(site.kind_name).symbol
        new_structure.append_atom(
            position=site.position, symbols=this_symbol, name=name
        )
    new_structure.label = structure.label
    return {"structure": new_structure, "magmom": orm.List(list=magmom)}


@calcfunction
def make_vac(cell, indices, supercell):
    """Make a defect containing cell"""
    atoms = cell.get_ase()
    supercell = atoms.repeat(supercell.get_list())
    mask = np.in1d(np.arange(len(supercell)), indices.get_list())
    supercell = supercell[~mask]  ## Remove any atoms in the original indices
    supercell.set_tags(None)
    supercell.set_masses(None)
    # Now I sort the supercell in the order of chemical symbols
    supercell = sort(supercell)
    output = StructureData(ase=supercell)
    return output


@calcfunction
def make_vac_at_o(cell, excluded_sites, nsub, supercell):
    """
    Make lots of vacancy containing cells usnig BSYM

    Use BSYM to do the job, vacancies are subsituted with P and
    later removed. Excluded sites are subsituted with S and later
    converted back to O.
    """
    from bsym.interface.pymatgen import unique_structure_substitutions
    from pymatgen.core import Composition

    nsub = nsub.value
    struc = cell.get_pymatgen()
    excluded = excluded_sites.get_list()

    assert "Ts" not in struc.composition
    assert "Og" not in struc.composition

    for n, site in enumerate(struc.sites):
        if n in excluded:
            site.species = Composition("Ts")

    # Expand the supercell with S subsituted strucutre
    struc = struc * supercell.get_list()
    noxygen = int(struc.composition["O"])
    unique_structure = unique_structure_substitutions(
        struc, "O", {"Og": nsub, "O": noxygen - nsub}
    )
    # Convert back to normal structure
    # Remove P as they are vacancies, Convert S back to O
    for ustruc in unique_structure:
        p_indices = [
            n
            for n, site in enumerate(ustruc.sites)
            if site.species == Composition("Og")
        ]
        ustruc.remove_sites(p_indices)
        # Convert S sites back to O
        ustruc["Ts"] = "O"

    output_structs = {}
    for n, s in enumerate(unique_structure):
        stmp = StructureData(pymatgen=s)
        stmp.set_attribute("vac_id", n)
        stmp.set_attribute("supercell", " ".join(map(str, supercell.get_list())))
        stmp.label = cell.label + f" VAC {n}"
        output_structs[f"structure_{n:04d}"] = stmp

    return output_structs


@calcfunction
def make_vac_at_o_and_shake(cell, excluded_sites, nsub, supercell, shake_amp):
    """
    Make lots of vacancy containing cells usnig BSYM

    Use BSYM to do the job, vacancies are subsituted with P and
    later removed. Excluded sites are subsituted with S and later
    converted back to O.

    In addition, we shake the nearest neighbours with that given by shake_amp.
    """
    from bsym.interface.pymatgen import unique_structure_substitutions
    from pymatgen.core import Composition
    from pymatgen.transformations.standard_transformations import (
        PerturbStructureTransformation,
    )

    nsub = nsub.value
    struc = cell.get_pymatgen()
    excluded = excluded_sites.get_list()

    assert "Ts" not in struc.composition
    assert "Og" not in struc.composition

    for n, site in enumerate(struc.sites):
        if n in excluded:
            site.species = Composition("Ts")

    # Expand the supercell with S subsituted strucutre
    struc = struc * supercell.get_list()
    noxygen = int(struc.composition["O"])
    unique_structure = unique_structure_substitutions(
        struc, "O", {"Og": nsub, "O": noxygen - nsub}
    )
    # Convert back to normal structure
    # Remove P as they are vacancies, Convert S back to O
    for ustruc in unique_structure:
        p_indices = [
            n
            for n, site in enumerate(ustruc.sites)
            if site.species == Composition("Og")
        ]

        ustruc.remove_sites(p_indices)
        # Convert S sites back to O
        ustruc["Ts"] = "O"

    # Perturb structures
    trans = PerturbStructureTransformation(distance=float(shake_amp))
    unique_structure = [
        trans.apply_transformation(ustruc) for ustruc in unique_structure
    ]

    output_structs = {}
    for n, s in enumerate(unique_structure):
        stmp = StructureData(pymatgen=s)
        stmp.set_attribute("vac_id", n)
        stmp.set_attribute("supercell", " ".join(map(str, supercell.get_list())))
        stmp.label = cell.label + f" VAC {n}"
        output_structs[f"structure_{n:04d}"] = stmp

    return output_structs


@calcfunction
def rattle(structure, amp):
    """
    Rattle the structure by a certain amplitude
    """
    native_keys = ["cell", "pbc1", "pbc2", "pbc3", "kinds", "sites", "mp_id"]
    # Keep the foreign keys as it is
    foreign_attrs = {
        key: value
        for key, value in structure.attributes.items()
        if key not in native_keys
    }
    atoms = structure.get_ase()
    atoms.rattle(amp.value)
    # Clean any tags etc
    atoms.set_tags(None)
    atoms.set_masses(None)
    # Convert it back
    out = StructureData(ase=atoms)
    out.set_attribute_many(foreign_attrs)
    out.label = structure.label + " RATTLED"
    return out


def res2structure_smart(file):
    """Create StructureData from SingleFileData, return existing node if there is any"""
    q = QueryBuilder()
    q.append(Node, filters={"id": file.pk})
    q.append(CalcFunctionNode, filters={"attributes.function_name": "res2structure"})
    q.append(StructureData)
    if q.count() > 0:
        print("Existing StructureData found")
        return q.first()
    else:
        return res2structure(file)


@calcfunction
def res2structure(file):
    """Create StructureData from SingleFile data"""
    from aiida.orm import StructureData

    from aiida_user_addons.tools.resutils import read_res

    with file.open(file.filename) as fhandle:
        titls, atoms = read_res(fhandle.readlines())
    atoms.set_tags(None)
    atoms.set_masses(None)
    atoms.set_calculator(None)
    atoms.wrap()
    struct = StructureData(ase=atoms)
    struct.set_attribute("H", titls.enthalpy)
    struct.set_attribute("search_label", titls.label)
    struct.label = file.filename
    return struct


@calcfunction
def get_primitive(structure):
    """Create primitive structure use pymatgen interface"""
    from aiida.orm import StructureData

    pstruct = structure.get_pymatgen()
    ps = pstruct.get_primitive_structure()
    out = StructureData(pymatgen=ps)
    out.label = structure.label + " PRIMITIVE"
    return out


@calcfunction
def get_standard_primitive(structure, **kwargs):
    """Create the standard primitive structure via seekpath"""
    from aiida.tools.data.array.kpoints import get_kpoints_path

    parameters = kwargs.get("parameters", {"symprec": 1e-5})
    if isinstance(parameters, orm.Dict):
        parameters = parameters.get_dict()

    out = get_kpoints_path(structure, **parameters)["primitive_structure"]
    out.label = structure.label + " PRIMITIVE"
    return out


@calcfunction
def spglib_refine_cell(structure, symprec):
    """Create the standard primitive structure via seekpath"""
    from aiida.tools.data.structure import (
        spglib_tuple_to_structure,
        structure_to_spglib_tuple,
    )
    from spglib import refine_cell

    structure_tuple, kind_info, kinds = structure_to_spglib_tuple(structure)

    lattice, positions, types = refine_cell(structure_tuple, symprec.value)

    refined = spglib_tuple_to_structure((lattice, positions, types), kind_info, kinds)

    return refined


@calcfunction
def get_standard_conventional(structure):
    """Create the standard primitive structure via seekpath"""
    from aiida.tools.data.array.kpoints import get_kpoints_path

    out = get_kpoints_path(structure)["conv_structure"]
    out.label = structure.label + " PRIMITIVE"
    return out


@calcfunction
def get_refined_structure(structure, symprec, angle_tolerance):
    """Create refined structure use pymatgen's interface"""
    from aiida.orm import StructureData
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pstruct = structure.get_pymatgen()
    ana = SpacegroupAnalyzer(
        pstruct, symprec=symprec.value, angle_tolerance=angle_tolerance.value
    )
    ps = ana.get_refined_structure()
    out = StructureData(pymatgen=ps)
    out.label = structure.label + " REFINED"
    return out


@calcfunction
def get_conventional_standard_structure(structure, symprec, angle_tolerance):
    """Create conventional standard structure use pymatgen's interface"""
    from aiida.orm import StructureData
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pstruct = structure.get_pymatgen()
    ana = SpacegroupAnalyzer(
        pstruct, symprec=symprec.value, angle_tolerance=angle_tolerance.value
    )
    ps = ana.get_conventional_standard_structure()
    out = StructureData(pymatgen=ps)
    out.label = structure.label + " CONVENTIONAL STANDARD"
    return out


@calcfunction
def make_supercell(structure, supercell, **kwargs):
    """Make supercell structure, keep the tags in order"""
    from ase.build.supercells import make_supercell as ase_supercell

    if "tags" in kwargs:
        tags = kwargs["tags"]
    else:
        tags = None

    atoms = structure.get_ase()
    atoms.set_tags(tags)

    slist = supercell.get_list()
    if isinstance(slist[0], int):
        satoms = atoms.repeat(slist)
    else:
        satoms = ase_supercell(atoms, np.array(slist))
    if "no_sort" not in kwargs:
        satoms = sort(satoms)

    if tags:
        stags = satoms.get_tags().tolist()
    satoms.set_tags(None)

    out = StructureData(ase=satoms)
    out.label = structure.label + " SUPER {} {} {}".format(*slist)

    if tags:
        return {"structure": out, "tags": orm.List(list=stags)}
    else:
        return {"structure": out}


@calcfunction
def delithiate_by_wyckoff(structure, wyckoff, element):
    """Remove ALL lithium in a certain wyckoff sites for a given structure"""
    remove_symbol = kwargs.get("element", orm.Str("Li"))
    remove_wyckoff = wyckoff.value

    ana = SpacegroupAnalyzer(structure.get_pymatgen())
    psymm = ana.get_symmetrized_structure()
    natoms = len(psymm.sites)

    rm_indices = []
    for lsite, lidx, symbol in zip(
        psymm.equivalent_sites, psymm.equivalent_indices, psymm.wyckoff_symbols
    ):
        site = lsite[0]
        if site.species_string != remove_symbol:
            continue
        if symbol != remove_wyckoff:
            continue
        rm_indices.extend(lidx)
    assert rm_indices, f"Nothing to remove for wyckoff {remove_wyckoff}"
    psymm.remove_sites(rm_indices)
    out = StructureData(pymatgen=Structure.from_sites(psymm.sites))
    for kind in out.kinds:
        assert not re.search(
            r"\d", kind.name
        ), f"Kind name: {kind.name} contains indices"

    # Set some special attribute
    out.set_attribute("removed_specie", remove_symbol)
    out.set_attribute("removed_wyckoff", remove_wyckoff)
    out.label += f" delithiated {remove_wyckoff}"

    # Prepare a mask for the removed structures
    mask = []
    for i in range(natoms):
        if i in rm_indices:
            mask.append(False)
        else:
            mask.append(True)
    outdict = {"structure": out, "mask": orm.List(list=mask)}
    return outdict


@calcfunction
def delithiate_full(structure, **kwargs):
    """
    Perform full delithation via removing all Li ions

    Returns:
        A dictionary of the StructureData without Li under the key 'structure'.
        The mask of the sites that are kept during the process is given under the 'mask' key.
        It can be useful for transforming other properties such as MAGMOM and tags.
    """
    remove_symbol = kwargs.get("element", orm.Str("Li"))
    pstruct = structure.get_pymatgen()
    to_remove = [
        idx
        for idx, site in enumerate(pstruct.sites)
        if site.species_string == remove_symbol
    ]
    pstruct.remove_sites(to_remove)

    out = StructureData(pymatgen=pstruct)
    out.set_attribute("removed_specie", remove_symbol)
    out.label = structure.label + f" fully delithiated"
    out.description = f"A fully delithiated structure, crated from {structure.uuid}"

    # Create the mask
    mask = []
    natoms = len(structure.sites)
    for i in range(natoms):
        if i in to_remove:
            mask.append(False)
        else:
            mask.append(True)
    outdict = {"structure": out, "mask": orm.List(list=mask)}
    return outdict


@calcfunction
def delithiate_one(structure, **kwargs):
    """
    Remove one lithium atom, enumerate the possible structures

    Symmetry is not taken into account in this function

    Returns:
        A dictionary of the StructureData without 1 Li under the key 'structure_<id>'.
        The mask of the sites that are kept during the process is given under the 'mask_<id>' key.
        It can be useful for transforming other properties such as MAGMOM and tags.
    """

    remove_symbol = kwargs.get("element", orm.Str("Li"))
    pstruct = structure.get_pymatgen()
    to_remove = [
        idx
        for idx, site in enumerate(pstruct.sites)
        if site.species_string == remove_symbol
    ]
    outdict = {}
    for idx, site in enumerate(to_remove):
        tmp_struct = structure.get_pymatgen()
        tmp_struct.remove_sites([site])

        out = StructureData(pymatgen=tmp_struct)
        out.set_attribute("removed_specie", remove_symbol)
        out.set_attribute("removed_site", site)
        out.label = structure.label + f" delithiated 1 - {idx}"
        out.description = (
            f"A structure with one Li removed, crated from {structure.uuid}"
        )

        # Create the mask
        mask = []
        natoms = len(structure.sites)
        for i in range(natoms):
            if i == site:
                mask.append(False)
            else:
                mask.append(True)
        outdict.update({f"structure_{idx}": out, f"mask_{idx}": orm.List(list=mask)})
    return outdict


@calcfunction
def delithiate_unique_sites(cell, excluded_sites, nsub, atol, **kwargs):
    """
    Generate delithiated non-equivalent cells using BSYM
    Args:
        cell (StructureData): The cell to be delithiate
        excluded_sites (List): A list of site indices to be excluded
        nsub (Int): Number of sites to be delithiated
        atol (Float): Symmetry tolerance for BSYM
        limit (Int, optional): Maximum limit of the structures

    Returns:
        A dictionary of structures and corresponding site mappings
    """
    element = kwargs.get("element", orm.Str("Li"))
    return _delithiate_unique_sites(
        cell,
        excluded_sites,
        nsub,
        atol,
        limit=kwargs.get("limit"),
        pmg_only=False,
        element=element.value,
    )


def _delithiate_unique_sites(
    cell, excluded_sites, nsub, atol, pmg_only=False, limit=None, element="Li"
):
    """
    Make lots of delithiated non-equivalent cells using BSYM

    Use BSYM to do the job, vacancies are subsituted with P and
    later removed. Excluded sites are subsituted with S and later
    converted back to Li.

    Args:
        cell (StructureData): The cell to be delithiate
        excluded_sites (List): A list of site indices to be excluded
        nsub (Int): Number of sites to be delithiated
        atol (Float): Symmetry tolerance for BSYM
        pmg_only (Bool): Only return a list of pymatgen structures.
        limit (Int, optional): Maximum limit of the structures. Default to None - no limit.

    Returns:
        A dictionary of structures and corresponding site mappings
    """
    from bsym.interface.pymatgen import unique_structure_substitutions
    from pymatgen.core import Composition

    exclude_dummy = "Ar"
    vacancy_dummy = "He"
    nsub = nsub.value
    struc = cell.get_pymatgen()
    excluded = excluded_sites.get_list()

    for n, site in enumerate(struc.sites):
        if n in excluded:
            site.species = Composition(exclude_dummy)

    # Expand the supercell with S subsituted strucutre
    noli = int(struc.composition[element])
    li_left = noli - nsub
    if li_left > 0:
        unique_structure = unique_structure_substitutions(
            struc,
            element,
            {vacancy_dummy: nsub, element: noli - nsub},
            verbose=True,
            atol=float(atol),
        )
    elif li_left == 0:
        unique_structure = unique_structure_substitutions(
            struc, element, {vacancy_dummy: nsub}, verbose=True, atol=float(atol)
        )
    else:
        raise ValueError(
            f"There are {noli} {element} but requested to remove {nsub} of them!!"
        )

    # Convert back to normal structure
    for ustruc in unique_structure:
        p_indices = [
            n
            for n, site in enumerate(ustruc.sites)
            if site.species == Composition(vacancy_dummy)
        ]
        ustruc.remove_sites(p_indices)
        ustruc[exclude_dummy] = element

    if limit is not None:
        unique_structure = unique_structure[: int(limit)]

    if pmg_only:
        return unique_structure

    # Transform into StructureData
    output_dict = {}
    for n, s in enumerate(unique_structure):
        stmp = StructureData(pymatgen=s)
        stmp.set_attribute("delithiate_id", n)
        stmp.label = cell.label + f" delithiate {n}"
        output_dict[f"structure_{n:04d}"] = stmp

        # Create the mask to map old site to the new sites
        # can be used to redfine per-site properties such as the mangetic moments
        # Simply search for the close position matches.
        mapping = []
        for i_new, new_site in enumerate(s.sites):
            found = False
            for i_old, old_site in enumerate(struc.sites):
                dist = new_site.distance(old_site)
                if dist < 0.1:
                    mapping.append(i_old)
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Cannot found original site for {new_site}")
        map_array = ArrayData()
        map_array.set_array("site_mapping", np.array(mapping))
        output_dict[f"mapping_{n:04d}"] = map_array

    return output_dict


@calcfunction
def niggli_reduce(structure):
    """Peroform niggli reduction"""
    from ase.build import niggli_reduce as niggli_reduce_

    atoms = structure.get_ase()
    niggli_reduce_(atoms)
    new_structure = StructureData(ase=atoms)
    new_structure.label = structure.label + " NIGGLI"
    return new_structure


@calcfunction
def neb_interpolate(init_structure, final_strucrture, nimages):
    """
    Interplate NEB frames using the starting and the final structures

    Get around the PBC warpping problem by calculating the MIC displacements
    from the initial to the final structure
    """

    ainit = init_structure.get_ase()
    afinal = final_strucrture.get_ase()
    disps = []

    # Find distances
    acombined = ainit.copy()
    acombined.extend(afinal)
    # Get piece-wise MIC distances
    for i in range(len(ainit)):
        dist = acombined.get_distance(i, i + len(ainit), vector=True, mic=True)
        disps.append(dist.tolist())
    disps = np.asarray(disps)
    ainit.wrap(eps=1e-1)
    afinal = ainit.copy()

    # Displace the atoms according to MIC distances
    afinal.positions += disps
    neb = NEB([ainit.copy() for i in range(int(nimages) + 1)] + [afinal.copy()])
    neb.interpolate()
    out_init = StructureData(ase=neb.images[0])
    out_init.label = init_structure.label + " INIT"
    out_final = StructureData(ase=neb.images[-1])
    out_final.label = init_structure.label + " FINAL"

    outputs = {"image_init": out_init}
    for i, out in enumerate(neb.images[1:-1]):
        outputs[f"image_{i+1:02d}"] = StructureData(ase=out)
        outputs[f"image_{i+1:02d}"].label = init_structure.label + f" FRAME {i+1:02d}"
    outputs["image_final"] = out_final
    return outputs


@calcfunction
def fix_atom_order(reference, to_fix):
    """
    Fix atom order by finding NN distances bet ween two frames. This resolves
    the issue where two closely matching structures having diffferent atomic orders.
    Note that the two frames must be close enough for this to work
    """

    aref = reference.get_ase()
    afix = to_fix.get_ase()

    # Index of the reference atom in the second structure
    new_indices = np.zeros(len(aref), dtype=int)

    # Find distances
    acombined = aref.copy()
    acombined.extend(afix)
    # Get piece-wise MIC distances
    for i in range(len(aref)):
        dists = []
        for j in range(len(aref)):
            dist = acombined.get_distance(i, j + len(aref), mic=True)
            dists.append(dist)
        min_idx = np.argmin(dists)
        min_dist = min(dists)
        if min_dist > 0.5:
            print(
                f"Large displacement found - moving atom {j} to {i} - please check if this is correct!"
            )
        new_indices[i] = min_idx

    afixed = afix[new_indices]
    fixed_structure = StructureData(ase=afixed)
    fixed_structure.label = to_fix.label + " UPDATED ORDER"
    return fixed_structure


def match_atomic_order_(atoms: Atoms, atoms_ref: Atoms) -> Tuple[Atoms, List[int]]:
    """
    Reorder the atoms to that of the reference.

    Only works for identical or nearly identical structures that are ordered differently.
    Returns a new `Atoms` object with order similar to that of `atoms_ref` as well as the sorting indices.
    """

    # Find distances
    acombined = atoms_ref.copy()
    acombined.extend(atoms)
    new_index = []
    # Get piece-wise MIC distances
    jidx = list(range(len(atoms), len(atoms) * 2))
    for i in range(len(atoms)):
        dists = acombined.get_distances(i, jidx, mic=True)
        # Find the index of the atom with the smallest distance
        min_idx = np.where(dists == dists.min())[0][0]
        new_index.append(min_idx)
    assert len(set(new_index)) == len(atoms), "The detected mapping is not unique!"
    return atoms[new_index], new_index
