# aiida-user-addons

Some addons to AiiDA

## Requirements

`aiida-phonopy` needs to be at commit *d410a9a309ff45d4c25a9b24bcaf06b65f0c4f1a*.
https://github.com/aiida-phonopy/aiida-phonopy/commit/d410a9a309ff45d4c25a9b24bcaf06b65f0c4f1a

for magnetic material, a patched versions is needed:
https://github.com/zhubonan/aiida-phonopy/commit/7cfe9d357348606ea4914e935beb282e573f2e22

`aiida-core>1.3.0`


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

`verdi data addons` is the entry point for additional commands, availiable commands are:

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

- `vaspu.relax`: `RelaxWorkChain` with additional check and bug fixes.
- `vaspu.converge`: A convergence testing workchain that runs tests in parallel as much as possbile (NOT WORKING AT THE MOMENT).
- `vaspu.vasp`: Almost as the original one, used by other workcahins.
- `vaspu.bands`: Includes pre-relaxation of the input structure with more functionalities such as dealing with AFM spin arrangement. Based on `castep.bands`.
- `vaspu.phonopy`: Fully automated Phonon workflow from initial relaxation to final bandstructure/thermal properties.
- `vaspu.magnetic`: Magnetic enumeration workflow for finding lowest energy magnetic states.
- `vaspu.delithiate`: Workchain for delithiate structures and performing relaxation.
- `vaspu.master`: Same as the original, not used.

## Additional Scheduler plugins

- `pbsarcher`: ARCHER scheduler plugin allows selecting `bigmem` nodes
- `sgenodetail`: SGE scheduler plugin that omits the `detailed_job_info` retrieval which can be slow on high-throughput cluster due to the large size of the accoutring file.

## Installation

Recommend to install in the editable mode to allow any changes to be applied immediately (after daemon restart, of course):

```
pip install -e .
```

Don't forget to do:

```
reentry scan -r aiida
```

in case entrypoints changes

## Usage

Custom workflows are registered as entry points, so `WorkflowFactory('vaspu.relax')` will load the relaxation workflow.
