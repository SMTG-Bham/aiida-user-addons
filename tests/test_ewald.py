"""
Adapted tests fore the Ewald summation.
Originally from the pymatgen code base.
"""

from pathlib import Path
import unittest
import warnings

import numpy as np

from aiida_user_addons.tools.ewald import EwaldSummation
from pymatgen.io.vasp.inputs import Poscar


class EwaldSummationTest(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        filepath = str(Path(__file__).parent / 'LFP.vasp')
        p = Poscar.from_file(filepath, check_for_POTCAR=False)
        self.original_s = p.structure
        self.s = self.original_s.copy()
        self.s.add_oxidation_state_by_element({'Li': 1, 'Fe': 2, 'P': 5, 'O': -2})

    def tearDown(self):
        warnings.simplefilter('default')

    def test_init(self):
        ham = EwaldSummation(self.s, compute_forces=True)
        self.assertAlmostEqual(ham.real_space_energy, -502.23549897772602, 4)
        self.assertAlmostEqual(ham.reciprocal_space_energy, 6.1541071599534654, 4)
        self.assertAlmostEqual(ham.point_energy, -620.22598358035918, 4)
        self.assertAlmostEqual(ham.total_energy, -1123.00766, 1)
        self.assertAlmostEqual(ham.forces[0, 0], -1.98818620e-01, 4)
        self.assertAlmostEqual(sum(sum(abs(ham.forces))), 915.925354346, 4, 'Forces incorrect')
        self.assertAlmostEqual(sum(sum(ham.real_space_energy_matrix)), ham.real_space_energy, 4)
        self.assertAlmostEqual(sum(sum(ham.reciprocal_space_energy_matrix)), ham.reciprocal_space_energy, 4)
        self.assertAlmostEqual(sum(ham.point_energy_matrix), ham.point_energy, 4)
        self.assertAlmostEqual(
            sum(sum(ham.total_energy_matrix)) + ham._charged_cell_energy,
            ham.total_energy,
            2,
        )

        self.assertRaises(ValueError, EwaldSummation, self.original_s)
        # try sites with charge.
        charges = []
        for site in self.original_s:
            if site.specie.symbol == 'Li':
                charges.append(1)
            elif site.specie.symbol == 'Fe':
                charges.append(2)
            elif site.specie.symbol == 'P':
                charges.append(5)
            else:
                charges.append(-2)

        self.original_s.add_site_property('charge', charges)
        ham2 = EwaldSummation(self.original_s)
        self.assertAlmostEqual(ham2.real_space_energy, -502.23549897772602, 4)

    def test_from_dict(self):
        ham = EwaldSummation(self.s, compute_forces=True)
        ham2 = EwaldSummation.from_dict(ham.as_dict())
        self.assertIsNone(ham._real)
        self.assertFalse(ham._initialized)
        self.assertIsNone(ham2._real)
        self.assertFalse(ham2._initialized)
        self.assertTrue(np.array_equal(ham.total_energy_matrix, ham2.total_energy_matrix))
        # check lazy eval
        self.assertAlmostEqual(ham.total_energy, -1123.00766, 1)
        self.assertIsNotNone(ham._real)
        self.assertTrue(ham._initialized)
        ham2 = EwaldSummation.from_dict(ham.as_dict())
        self.assertIsNotNone(ham2._real)
        self.assertTrue(ham2._initialized)
        self.assertTrue(np.array_equal(ham.total_energy_matrix, ham2.total_energy_matrix))

    def test_as_dict(self):
        ham = EwaldSummation(self.s, compute_forces=True)
        d = ham.as_dict()
        self.assertTrue(d['compute_forces'])
        self.assertEqual(d['eta'], ham._eta)
        self.assertEqual(d['acc_factor'], ham._acc_factor)
        self.assertEqual(d['real_space_cut'], ham._rmax)
        self.assertEqual(d['recip_space_cut'], ham._gmax)
        self.assertEqual(ham.as_dict(), EwaldSummation.from_dict(d).as_dict())


if __name__ == '__main__':
    unittest.main()
