import unittest

from gwinferno.preprocess.data_collection import cosmo
from gwinferno.cosmology import PLANCK_2015_LVK_Cosmology


class TestBase1DBSplineModel(unittest.TestCase):
    def test_cosmology(self):
        self.assertIs(cosmo, PLANCK_2015_LVK_Cosmology)
