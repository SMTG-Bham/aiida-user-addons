# aiida-user-addons

Some (personal) addons to AiiDA to help my research. Mainly focusing using VASP through `aiida-vasp` and also some workflows for CASTEP are also included.

## Requirements

For phonon calculation the
`aiida-phonopy` package needs to be at commit *d410a9a309ff45d4c25a9b24bcaf06b65f0c4f1a*.
https://github.com/aiida-phonopy/aiida-phonopy/commit/d410a9a309ff45d4c25a9b24bcaf06b65f0c4f1a

for magnetic material, a patched versions might be needed:
https://github.com/zhubonan/aiida-phonopy/commit/7cfe9d357348606ea4914e935beb282e573f2e22

For  `aiida-core>2.0.1` is needed.
The `aiida-vasp` should be at the development version.
A recent version `aiida-castep` is needed.


## Command line tools

`vasp-dryrun` is a tool to *dryrun* VASP calculations. It can obtain key information such as the number of kpoints and bands for working out the best parallelisation strategy.

```
Usage: vasp-dryrun [OPTIONS]

  A simple tool to dryrun a VASP calculation. The calculation will be run
  for up to <timeout> seconds. The underlying VASP process will be
  terminated once it enters the main loop, which is signalled by the
  appearance of a `INWAV` keyword in the OUTCAR.

Options:
  --input-dir DIRECTORY  Where the VASP input is, default to the current
                         working directory.  [default: .]

  --vasp-exe TEXT        Executable for VASP  [default: vasp_std]
  --timeout INTEGER      Timeout in seconds to terminate VASP  [default: 10]
  --work-dir TEXT        Working directory for running
  --keep                 Wether to the dryrun files  [default: False]
  --help                 Show this message and exit.
```

This package acts as a plugin for `aiida-core` and provides workflows and other functionalities.

The command line interface is integrated with `verdi`, additional commands are available under `verdi data addons`:

```
Usage: verdi data addons [OPTIONS] COMMAND [ARGS]...

  Entry point for commands under aiida-user-addons

Options:
  -h, --help  Show this message and exit.

Commands:
  check-nelm    Perform a sweep to check if any VaspCalculation have...
  export_relax  Export a VASP relaxation workflow
  export_vasp   Export a VASP calculation, works for both `VaspCalculation`...
  remotecat     Print the content of a remote file to STDOUT This command...
  remotepull    Pull a calculation folder from the remote This command for...
  remotetail    Follow a file on the remote computer This command will...
```


## Additional VASP workflows

The main entry point has the `vaspu` prefix:

- `vaspu.relax`: `VaspRelaxWorkChain` with additional check and bug fixes.
- `vaspu.converge`: A simpler convergence testing workchain that runs tests in parallel.
- `vaspu.vasp`: Almost as the original one, used by other workchains.
- `vaspu.bands`: Includes pre-relaxation of the input structure with more functionalities such as dealing with AFM spin arrangement. Based on `castep.bands`.
- `vaspu.hybrid_bands`: Run band structure calculations normal calculations with *zero-weighted kpoints*. Useful if hybrid functional is used.
- `vaspu.phonopy`: Fully automated Phonon workflow from initial relaxation to final bandstructure/thermal properties.
- `vaspu.magnetic`: Magnetic enumeration workflow for finding lowest energy magnetic states.
- `vaspu.delithiate`: Workchain for delithiate structures and performing relaxation.
- `vaspu.voltage_curve`: Workchain for constructing voltage curve upon delithiation of certain lithiated structure.

## Related codes

This package also provide some convenient routine for the following packages:

- [pymatgen](https://pymatgen.org/): Routines for acquiring structures from Materials Project.
- [sumo](https://github.com/SMTG-UCL/sumo): Used for plotting band structures and density of states.
- [hiphive](https://hiphive.materialsmodeling.org/): Minimum wrapper for generating MC rattled structures.
- [clease](https://gitlab.com/computationalmaterials/clease): For depositing/extracting structures stored in its database.
- [phonopy](https://phonopy.github.io/phonopy/): Used for phonon calculations (vasp). Existing phonon workchains can be exported as files.

## Additional Scheduler plugins

- `sgenodetail`: SGE scheduler plugin that omits the `detailed_job_info` retrieval which can be slow on high-throughput cluster due to the large size of the accoutring file.


## Installation

Recommend to install in the editable mode to allow any changes to be applied immediately (after daemon restart, of course):

```
pip install -e .
```

## Required package version

Development version of `aiida-vasp` should be used.

## Usage

Custom workflows are registered as entry points, so `WorkflowFactory('vaspu.relax')` will load the relaxation workflow.
Check the outputs of `verdi plugin list aiida.workflows` to see the workflows available.

Example notebooks can be found in the `examples` folder.
