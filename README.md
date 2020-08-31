# aiida-user-addons

Some addons to AiiDA

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