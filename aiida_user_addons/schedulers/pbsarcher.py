"""
Module for the pbs pro on Archer
ARCHER does not like explicit MPI number definition, so the number of mpi process is
not included in the submission script.

In addition, ARCHER supports a `bigmem` flag that can be used to request nodes with
larger memory size. For this we have to tweak the `PbsJobResource`.
"""
import logging
import os
from math import ceil

from aiida.common.escaping import escape_for_bash
from aiida.common.extendeddicts import AttributeDict
from aiida.schedulers import Scheduler
from aiida.schedulers.plugins.pbsbaseclasses import (
    PbsBaseClass,
    PbsJobResource,
)

_LOGGER = logging.getLogger(__name__)


class PbsArcherJobResource(PbsJobResource):
    """
    JobResource for ARCHER with bigmem flag for big memory nodes
    """

    _default_fields = (
        "num_machines",
        "num_mpiprocs_per_machine",
        "num_cores_per_machine",
        "num_cores_per_mpiproc",
        "bigmem",
    )

    @classmethod
    def validate_resources(cls, **kwargs):
        """Validate the resources against the job resource class of this scheduler.

        :param kwargs: dictionary of values to define the job resources
        :return: attribute dictionary with the parsed parameters populated
        :raises ValueError: if the resources are invalid or incomplete
        """
        resources = AttributeDict()

        def is_greater_equal_one(parameter):
            value = getattr(resources, parameter, None)
            if value is not None and value < 1:
                raise ValueError(f"`{parameter}` must be greater than or equal to one.")

        # Validate that all fields are valid integers if they are specified, otherwise initialize them to `None`
        for parameter in list(cls._default_fields) + ["tot_num_mpiprocs"]:
            # Special case bigmem tag is a bool type
            if parameter == "bigmem":
                try:
                    value = kwargs.pop(parameter)
                except KeyError:
                    setattr(resources, parameter, None)
                else:
                    if isinstance(value, bool):
                        setattr(resources, parameter, value)
                    else:
                        raise ValueError(
                            f"`{parameter}` must be an bool type when specified"
                        )
            else:
                try:
                    setattr(resources, parameter, int(kwargs.pop(parameter)))
                except KeyError:
                    setattr(resources, parameter, None)
                except ValueError:
                    raise ValueError(f"`{parameter}` must be an integer when specified")

        if kwargs:
            raise ValueError(
                "these parameters were not recognized: {}".format(
                    ", ".join(list(kwargs.keys()))
                )
            )

        # At least two of the following parameters need to be defined as non-zero
        if [
            resources.num_machines,
            resources.num_mpiprocs_per_machine,
            resources.tot_num_mpiprocs,
        ].count(None) > 1:
            raise ValueError(
                "At least two among `num_machines`, `num_mpiprocs_per_machine` or `tot_num_mpiprocs` must be specified."
            )

        for parameter in ["num_machines", "num_mpiprocs_per_machine"]:
            is_greater_equal_one(parameter)

        # Here we now that at least two of the three required variables are defined and greater equal than one.
        if resources.num_machines is None:
            resources.num_machines = ceil(
                resources.tot_num_mpiprocs / resources.num_mpiprocs_per_machine
            )
        elif resources.num_mpiprocs_per_machine is None:
            resources.num_mpiprocs_per_machine = 24  # Default for ARCHER
        elif resources.tot_num_mpiprocs is None:
            resources.tot_num_mpiprocs = (
                resources.num_mpiprocs_per_machine * resources.num_machines
            )

        if (
            resources.tot_num_mpiprocs
            > resources.num_mpiprocs_per_machine * resources.num_machines
        ):
            raise ValueError(
                "`tot_num_mpiprocs` is more than the `num_mpiprocs_per_machine * num_machines`."
            )

        is_greater_equal_one("num_mpiprocs_per_machine")
        is_greater_equal_one("num_machines")

        return resources


class PbsArcherScheduler(PbsBaseClass, Scheduler):
    """
    Subclass to support the PBSPro scheduler
    (http://www.pbsworks.com/).

    Tune for runing jobs on ARCHER

    I redefine only what needs to change from the base class
    """

    _logger = Scheduler._logger.getChild("pbsarcher")

    ## I change it the ARCHER resource
    _job_resource_class = PbsArcherJobResource

    ## For the time being I use a common dictionary, should be sufficient
    ## for the time being, but I can redefine it if needed.
    # _map_status = _map_status_pbs_common
    def _get_submit_script_header(self, job_tmpl):
        """
        Return the submit script header, using the parameters from the
        job_tmpl.

        Args:
           job_tmpl: an JobTemplate instance with relevant parameters set.

        TODO: truncate the title if too long
        """
        import re
        import string

        empty_line = ""

        lines = []
        if job_tmpl.submit_as_hold:
            lines.append("#PBS -h")

        if job_tmpl.rerunnable:
            lines.append("#PBS -r y")
        else:
            lines.append("#PBS -r n")

        if job_tmpl.email:
            # If not specified, but email events are set, PBSPro
            # sends the mail to the job owner by default
            lines.append(f"#PBS -M {job_tmpl.email}")

        email_events = ""
        if job_tmpl.email_on_started:
            email_events += "b"
        if job_tmpl.email_on_terminated:
            email_events += "ea"
        if email_events:
            lines.append(f"#PBS -m {email_events}")
            if not job_tmpl.email:
                _LOGGER.info(
                    "Email triggers provided to PBSPro script for job,"
                    "but no email field set; will send emails to "
                    "the job owner as set in the scheduler"
                )
        else:
            lines.append("#PBS -m n")

        if job_tmpl.job_name:
            # From qsub man page:
            # string, up to 15 characters in length.  It must
            # consist of an  alphabetic  or  numeric  character
            # followed  by printable, non-white-space characters.
            # Default:  if a script is used to submit the job, the job's name
            # is the name of the script.  If no script  is  used,  the  job's
            # name is "STDIN".
            #
            # I leave only letters, numbers, dots, dashes and underscores
            # Note: I don't compile the regexp, I am going to use it only once
            job_title = re.sub(r"[^a-zA-Z0-9_.-]+", "", job_tmpl.job_name)

            # prepend a 'j' (for 'job') before the string if the string
            # is now empty or does not start with a valid charachter
            if not job_title or (
                job_title[0] not in string.ascii_letters + string.digits
            ):
                job_title = "j" + job_title

            # Truncate to the first 15 characters
            # Nothing is done if the string is shorter.
            job_title = job_title[:15]

            lines.append(f"#PBS -N {job_title}")

        if job_tmpl.import_sys_environment:
            lines.append("#PBS -V")

        if job_tmpl.sched_output_path:
            lines.append(f"#PBS -o {job_tmpl.sched_output_path}")

        if job_tmpl.sched_join_files:
            # from qsub man page:
            # 'oe': Standard error and standard output are merged  into
            #       standard output
            # 'eo': Standard error and standard output are merged  into
            #       standard error
            # 'n' : Standard error and standard output are not merged (default)
            lines.append("#PBS -j oe")
            if job_tmpl.sched_error_path:
                _LOGGER.info(
                    "sched_join_files is True, but sched_error_path is set in "
                    "PBSPro script; ignoring sched_error_path"
                )
        else:
            if job_tmpl.sched_error_path:
                lines.append(f"#PBS -e {job_tmpl.sched_error_path}")

        if job_tmpl.queue_name:
            lines.append(f"#PBS -q {job_tmpl.queue_name}")

        queue_override = os.environ.get("ARCHER_QUEUE")
        if queue_override:
            lines.append(f"#PBS -q {queue_override}")

        if job_tmpl.account:
            lines.append(f"#PBS -A {job_tmpl.account}")

        if job_tmpl.priority:
            # Priority of the job.  Format: host-dependent integer.  Default:
            # zero.   Range:  [-1024,  +1023] inclusive.  Sets job's Priority
            # attribute to priority.
            # TODO: Here I expect that priority is passed in the correct PBSPro
            # format. To fix.
            lines.append(f"#PBS -p {job_tmpl.priority}")

        if not job_tmpl.job_resource:
            raise ValueError(
                "Job resources (as the num_machines) are required for the PBSPro scheduler plugin"
            )

        # NOTE HERE I Added the bigmem flag
        resource_lines = self._get_resource_lines(
            num_machines=job_tmpl.job_resource.num_machines,
            num_mpiprocs_per_machine=job_tmpl.job_resource.num_mpiprocs_per_machine,
            num_cores_per_machine=job_tmpl.job_resource.num_cores_per_machine,
            max_memory_kb=job_tmpl.max_memory_kb,
            bigmem=job_tmpl.job_resource.bigmem,
            max_wallclock_seconds=job_tmpl.max_wallclock_seconds,
        )

        lines += resource_lines

        if job_tmpl.custom_scheduler_commands:
            lines.append(job_tmpl.custom_scheduler_commands)

        # Job environment variables are to be set on one single line.
        # This is a tough job due to the escaping of commas, etc.
        # moreover, I am having issues making it work.
        # Therefore, I assume that this is bash and export variables by
        # and.

        if job_tmpl.job_environment:
            lines.append(empty_line)
            lines.append("# ENVIRONMENT VARIABLES BEGIN ###")
            if not isinstance(job_tmpl.job_environment, dict):
                raise ValueError(
                    "If you provide job_environment, it must be a dictionary"
                )
            for key, value in job_tmpl.job_environment.items():
                lines.append(f"export {key.strip()}={escape_for_bash(value)}")
            lines.append("# ENVIRONMENT VARIABLES  END  ###")
            lines.append(empty_line)

        # Required to change directory to the working directory, that is
        # the one from which the job was submitted
        lines.append('cd "$PBS_O_WORKDIR"')
        lines.append(empty_line)

        return "\n".join(lines)

    def _get_resource_lines(
        self,
        num_machines,
        num_mpiprocs_per_machine,
        num_cores_per_machine,
        max_memory_kb,
        max_wallclock_seconds,
        bigmem,
    ):
        """
        Return the lines for machines, memory and wallclock relative
        to pbspro.
        """
        # Note: num_cores_per_machine is not used here but is provided by
        #       the parent class ('_get_submit_script_header') method

        return_lines = []

        select_string = f"select={num_machines}"

        # Archer does not like these flag?
        # if num_mpiprocs_per_machine:
        #    select_string += ":mpiprocs={}".format(num_mpiprocs_per_machine)

        #        if num_cores_per_machine:
        #            select_string += ":ppn={}".format(num_cores_per_machine)

        if max_wallclock_seconds is not None:
            try:
                tot_secs = int(max_wallclock_seconds)
                if tot_secs <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    "max_wallclock_seconds must be "
                    "a positive integer (in seconds)! It is instead '{}'"
                    "".format(max_wallclock_seconds)
                )
            hours = tot_secs // 3600
            tot_minutes = tot_secs % 3600
            minutes = tot_minutes // 60
            seconds = tot_minutes % 60
            return_lines.append(
                f"#PBS -l walltime={hours:02d}:{minutes:02d}:{seconds:02d}"
            )

        if bigmem:
            select_string += ":bigmem=true"

        if max_memory_kb:
            try:
                virtualMemoryKb = int(max_memory_kb)
                if virtualMemoryKb <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    "max_memory_kb must be "
                    "a positive integer (in kB)! It is instead '{}'"
                    "".format(max_memory_kb)
                )
            select_string += f":mem={virtualMemoryKb}kb"

        return_lines.append(f"#PBS -l {select_string}")
        return return_lines
