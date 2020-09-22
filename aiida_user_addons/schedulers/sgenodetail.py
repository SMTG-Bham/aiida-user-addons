# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""
Plugin for SGE.
This has been tested on GE 6.2u3.

Plugin originally written by Marco Dorigo.
Email: marco(DOT)dorigo(AT)rub(DOT)de
"""

from aiida.schedulers.plugins.sge import SgeScheduler
from aiida.common.exceptions import FeatureNotAvailable


class SgeNoDetailScheduler(SgeScheduler):
    """
    SGE Scheduler for Myriad - the qacct account in SGE is slow.
    This plugin does use qacct to save ``detailed_job_info``.
    """

    def _get_detailed_job_info_command(self, jobid):
        """
        Override superclass method
        """
        # pylint: disable=no-self-use, not-callable, unused-argument
        raise FeatureNotAvailable('Detailed jobinfo disabled in SgeNoDetailScheduler')
