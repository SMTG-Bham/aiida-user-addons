# aiida-user-addons

Some addons to AiiDA

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

## Additional VASP workflows

- `vaspu.relax`: `RelaxWorkChain` with additional check and bug fixes
- `vaspu.converge`: A convergence testing workchain that runs tests in parallel as much as possbile.
- `vaspu.vasp`: Same as the original, not used.
- `vaspu.bands`: Same as the original for now, not used.
- `vaspu.master`: Same as the original for now, not used.

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
