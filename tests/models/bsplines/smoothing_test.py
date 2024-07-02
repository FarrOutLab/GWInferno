import unittest

import jax.numpy as jnp

from gwinferno.models.bsplines.smoothing import apply_difference_prior


class TestSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.nsplines = 10
        self.coefs = jnp.ones((self.nsplines,))
        self.inv_var = 5

    def tearDown(self) -> None:
        del self.nsplines
        del self.coefs

    def test_apply_difference_prior(self):
        diff_prior = apply_difference_prior(self.coefs, self.inv_var)

        self.assertEqual(diff_prior, 0)
