"""
Test the schedulers
"""
import unittest
import uuid

import pytest

from .pbsarcher import PbsArcherJobResource, PbsArcherScheduler
from .sgenodetail import FeatureNotAvailable, SgeNoDetailScheduler


def test_archer_job_resource():
    """
    Simple test for the resource class
    """
    resource = PbsArcherJobResource(num_machines=1, num_mpiprocs_per_machine=16)
    assert resource.bigmem is None
    resource = PbsArcherJobResource(
        num_machines=1, bigmem=True, num_mpiprocs_per_machine=16
    )
    assert resource.bigmem is True
    with pytest.raises(ValueError):
        resource = PbsArcherJobResource(
            num_machines=1, bigmem=100, num_mpiprocs_per_machine=16
        )

    resource = PbsArcherJobResource(num_machines=3, tot_num_mpiprocs=70)
    assert resource.num_machines == 3
    assert resource.tot_num_mpiprocs == 70
    assert resource.num_mpiprocs_per_machine == 24

    with pytest.raises(ValueError):
        resource = PbsArcherJobResource(
            num_machines=2, num_mpiprocs_per_machine=12, tot_num_mpiprocs=25
        )

    # Under populate
    resource = PbsArcherJobResource(
        num_machines=2, num_mpiprocs_per_machine=12, tot_num_mpiprocs=20
    )
    assert resource.num_machines == 2
    assert resource.tot_num_mpiprocs == 20
    assert resource.num_mpiprocs_per_machine == 12


def test_sge_nodetail():

    scheduler = SgeNoDetailScheduler()
    with pytest.raises(FeatureNotAvailable):
        scheduler._get_detailed_job_info_command(1)


class TestSubmitScript(unittest.TestCase):
    def test_submit_script(self):
        """
        Test to verify if scripts works fine with default options
        """
        from aiida.common.datastructures import CodeInfo, CodeRunMode
        from aiida.schedulers.datastructures import JobTemplate

        scheduler = PbsArcherScheduler()

        job_tmpl = JobTemplate()
        job_tmpl.shebang = "#!/bin/bash -l"
        job_tmpl.job_resource = scheduler.create_job_resource(
            num_machines=1, num_mpiprocs_per_machine=1, bigmem=True
        )
        job_tmpl.uuid = str(uuid.uuid4())
        job_tmpl.max_wallclock_seconds = 24 * 3600
        code_info = CodeInfo()
        code_info.cmdline_params = ["mpirun", "-np", "23", "pw.x", "-npool", "1"]
        code_info.stdin_name = "aiida.in"
        job_tmpl.codes_info = [code_info]
        job_tmpl.codes_run_mode = CodeRunMode.SERIAL

        submit_script_text = scheduler.get_submit_script(job_tmpl)

        self.assertTrue("#PBS -r n" in submit_script_text)
        self.assertTrue(submit_script_text.startswith("#!/bin/bash -l"))
        self.assertTrue("#PBS -l walltime=24:00:00" in submit_script_text)
        self.assertTrue("#PBS -l select=1:bigmem=true" in submit_script_text)
        self.assertTrue(
            "'mpirun' '-np' '23' 'pw.x' '-npool' '1'" + " < 'aiida.in'"
            in submit_script_text
        )
