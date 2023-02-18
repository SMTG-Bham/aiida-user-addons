"""
Test the SgeNoDetailScheduler
"""

import pytest

from aiida_user_addons.schedulers.sgenodetail import (
    FeatureNotAvailable,
    SgeNoDetailScheduler,
)


def test_sge_nodetail():

    scheduler = SgeNoDetailScheduler()
    with pytest.raises(FeatureNotAvailable):
        scheduler._get_detailed_job_info_command(1)
