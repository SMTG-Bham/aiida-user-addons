"""
Module for the pbs pro on Archer
ARCHER does not explicity MPI number definition, so the number of mpi process is
not inlcluded in the submission script

In addition, ARCHER supports a `bigmem` flag that can be used to request nodes with
larger memory size
"""
import logging

from aiida.schedulers import Scheduler
from aiida.schedulers.datastructures import NodeNumberJobResource
from aiida.common.escaping import escape_for_bash
from aiida.schedulers.plugins.pbsbaseclasses import PbsBaseClass, PbsJobResource

_LOGGER = logging.getLogger(__name__)


class PbsArcherJobResource(PbsJobResource):
    """
    JobResource for ARCHER with bigmem flag for big memory nodes
    """
    def __init__(self, *args, **kwargs):
        """
        Add additonal bigmem attribute
        """
        self.bigmem = kwargs.pop('bigmem', None)
        super(PbsArcherJobResource, self).__init__(*args, **kwargs)

    @classmethod
    def get_valid_keys(cls):
        """
        Extend with bigmem flag
        """
        return super(PbsArcherJobResource, cls).get_valid_keys() + ['bigmem']


def test_job_resource():
    """
    Simple test for the resource class
    """
    resource = PbsArcherJobResource(num_machines=1,
                                    num_mpiprocs_per_machine=16)
    assert resource.bigmem is None
    resource = PbsArcherJobResource(num_machines=1,
                                    bigmem=True,
                                    num_mpiprocs_per_machine=16)
    assert resource.bigmem is True


class PbsArcherScheduler(PbsBaseClass, Scheduler):
    """
    Subclass to support the PBSPro scheduler
    (http://www.pbsworks.com/).

    Tune for runing jobs on ARCHER

    I redefine only what needs to change from the base class
    """
    _logger = Scheduler._logger.getChild('pbsarcher')

    ## I change it the ARCHER resource
    _job_resource_class = PbsArcherJobResource

    ## For the time being I use a common dictionary, should be sufficient
    ## for the time being, but I can redefine it if needed.
    #_map_status = _map_status_pbs_common
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

        empty_line = ''

        lines = []
        if job_tmpl.submit_as_hold:
            lines.append('#PBS -h')

        if job_tmpl.rerunnable:
            lines.append('#PBS -r y')
        else:
            lines.append('#PBS -r n')

        if job_tmpl.email:
            # If not specified, but email events are set, PBSPro
            # sends the mail to the job owner by default
            lines.append('#PBS -M {}'.format(job_tmpl.email))

        email_events = ''
        if job_tmpl.email_on_started:
            email_events += 'b'
        if job_tmpl.email_on_terminated:
            email_events += 'ea'
        if email_events:
            lines.append('#PBS -m {}'.format(email_events))
            if not job_tmpl.email:
                _LOGGER.info(
                    'Email triggers provided to PBSPro script for job,'
                    'but no email field set; will send emails to '
                    'the job owner as set in the scheduler')
        else:
            lines.append('#PBS -m n')

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
            job_title = re.sub(r'[^a-zA-Z0-9_.-]+', '', job_tmpl.job_name)

            # prepend a 'j' (for 'job') before the string if the string
            # is now empty or does not start with a valid charachter
            if not job_title or (job_title[0] not in string.ascii_letters +
                                 string.digits):
                job_title = 'j' + job_title

            # Truncate to the first 15 characters
            # Nothing is done if the string is shorter.
            job_title = job_title[:15]

            lines.append('#PBS -N {}'.format(job_title))

        if job_tmpl.import_sys_environment:
            lines.append('#PBS -V')

        if job_tmpl.sched_output_path:
            lines.append('#PBS -o {}'.format(job_tmpl.sched_output_path))

        if job_tmpl.sched_join_files:
            # from qsub man page:
            # 'oe': Standard error and standard output are merged  into
            #       standard output
            # 'eo': Standard error and standard output are merged  into
            #       standard error
            # 'n' : Standard error and standard output are not merged (default)
            lines.append('#PBS -j oe')
            if job_tmpl.sched_error_path:
                _LOGGER.info(
                    'sched_join_files is True, but sched_error_path is set in '
                    'PBSPro script; ignoring sched_error_path')
        else:
            if job_tmpl.sched_error_path:
                lines.append('#PBS -e {}'.format(job_tmpl.sched_error_path))

        if job_tmpl.queue_name:
            lines.append('#PBS -q {}'.format(job_tmpl.queue_name))

        if job_tmpl.account:
            lines.append('#PBS -A {}'.format(job_tmpl.account))

        if job_tmpl.priority:
            # Priority of the job.  Format: host-dependent integer.  Default:
            # zero.   Range:  [-1024,  +1023] inclusive.  Sets job's Priority
            # attribute to priority.
            # TODO: Here I expect that priority is passed in the correct PBSPro
            # format. To fix.
            lines.append('#PBS -p {}'.format(job_tmpl.priority))

        if not job_tmpl.job_resource:
            raise ValueError(
                'Job resources (as the num_machines) are required for the PBSPro scheduler plugin'
            )

        # NOTE HERE I Added the bigmem flag
        resource_lines = self._get_resource_lines(
            num_machines=job_tmpl.job_resource.num_machines,
            num_mpiprocs_per_machine=job_tmpl.job_resource.
            num_mpiprocs_per_machine,
            num_cores_per_machine=job_tmpl.job_resource.num_cores_per_machine,
            max_memory_kb=job_tmpl.max_memory_kb,
            bigmem=job_tmpl.job_resource.bigmem,
            max_wallclock_seconds=job_tmpl.max_wallclock_seconds)

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
            lines.append('# ENVIRONMENT VARIABLES BEGIN ###')
            if not isinstance(job_tmpl.job_environment, dict):
                raise ValueError(
                    'If you provide job_environment, it must be a dictionary')
            for key, value in job_tmpl.job_environment.items():
                lines.append('export {}={}'.format(key.strip(),
                                                   escape_for_bash(value)))
            lines.append('# ENVIRONMENT VARIABLES  END  ###')
            lines.append(empty_line)

        # Required to change directory to the working directory, that is
        # the one from which the job was submitted
        lines.append('cd "$PBS_O_WORKDIR"')
        lines.append(empty_line)

        return '\n'.join(lines)

    def _get_resource_lines(self, num_machines, num_mpiprocs_per_machine,
                            num_cores_per_machine, max_memory_kb,
                            max_wallclock_seconds, bigmem):
        """
        Return the lines for machines, memory and wallclock relative
        to pbspro.
        """
        # Note: num_cores_per_machine is not used here but is provided by
        #       the parent class ('_get_submit_script_header') method

        return_lines = []

        select_string = 'select={}'.format(num_machines)

        # Archer does not like these flag?
        #if num_mpiprocs_per_machine:
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
                    'max_wallclock_seconds must be '
                    "a positive integer (in seconds)! It is instead '{}'"
                    ''.format(max_wallclock_seconds))
            hours = tot_secs // 3600
            tot_minutes = tot_secs % 3600
            minutes = tot_minutes // 60
            seconds = tot_minutes % 60
            return_lines.append('#PBS -l walltime={:02d}:{:02d}:{:02d}'.format(
                hours, minutes, seconds))

        if bigmem:
            select_string += ':bigmem=true'

        if max_memory_kb:
            try:
                virtualMemoryKb = int(max_memory_kb)
                if virtualMemoryKb <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    'max_memory_kb must be '
                    "a positive integer (in kB)! It is instead '{}'"
                    ''.format((max_memory_kb)))
            select_string += ':mem={}kb'.format(virtualMemoryKb)

        return_lines.append('#PBS -l {}'.format(select_string))
        return return_lines
