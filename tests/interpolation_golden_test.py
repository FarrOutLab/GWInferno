"""Golden-value regression tests for :mod:`gwinferno.interpolation`.

These pin the *numerical output* of the spline bases and the 1-D models built on them, so that a
rename or restructure can be shown to change no numbers. The fixture is generated once, before any
refactoring, and afterwards only the construction calls in this file change -- never the ``.npz``.

The fixture itself is deliberately **not** committed, and is gitignored. It stores float32 output
and these tests compare bit-exactly, while CI does not pin jax (the workflow just runs
``pip install numpyro``), so a committed baseline would drift on every jax release and on any
change of Python version or platform. These tests therefore skip when the fixture is absent, which
is the normal state on a fresh checkout. The durable, version-robust checks live in
``TestBasisIdentities`` and ``TestOutOfBoundsHandling`` in ``interpolation_test.py``.

To reproduce a refactor's bit-identical claim, check out the *library* at the commit the work
started from while keeping this file, generate the baseline, then restore the library::

    git checkout 39b2b92 -- gwinferno/   # pre-refactor baseline
    PYTHONPATH=. .venv/bin/python tests/interpolation_golden_test.py --write
    git checkout HEAD -- gwinferno/
    PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -t . -p "interpolation_golden_test.py"

Only this file's construction calls change as classes are renamed. The captured keys are named for
the *role* a class plays, never for its current name, so the key set survives a rename.

Regenerate in place only when a change is *intended* to alter values, and say so in the commit::

    PYTHONPATH=. .venv/bin/python tests/interpolation_golden_test.py --write

Deliberate choices:

* ``numpy.random.default_rng`` only, since ``jax.random`` output drifts across jax versions.
* ``jax_enable_x64`` is left off, so this captures the float32 the library actually produces.
  Values are stored as float64, which is lossless widening, so comparisons stay bit-exact.
* No ``.h5`` fixtures, so this runs without ``tests/download_test_data.sh``.
* The fixture lives at ``tests/interpolation_golden.npz``, not under ``tests/data/``, because the
  bare ``data`` pattern in ``.gitignore`` matches any directory of that name.
"""

import os
import sys
import unittest

import jax.numpy as jnp
import numpy as np

from gwinferno.cosmology import PLANCK_2015_LVK_Cosmology
from gwinferno.interpolation import BSpline
from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
from gwinferno.interpolation import MSpline
from gwinferno.models.bsplines.single import Base1DBSplineModel
from gwinferno.models.bsplines.single import BSplineChiEffective
from gwinferno.models.bsplines.single import BSplineChiPrecess
from gwinferno.models.bsplines.single import BSplineMass
from gwinferno.models.bsplines.single import BSplineRatio
from gwinferno.models.bsplines.single import BSplineRedshift
from gwinferno.models.bsplines.single import BSplineSpinMagnitude
from gwinferno.models.bsplines.single import BSplineSpinTilt
from gwinferno.models.bsplines.single import BSplineSymmetricChiEffective

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "interpolation_golden.npz")

N_DF = 10
N_SPLINES = 8
SEED = 20240607

# (label, class, xrange). The label names the *role*, deliberately not the class, so that renaming a
# class does not change the captured key set and a rename can still be compared key for key.
# Log-x variants need a strictly positive lower bound.
BASIS_CASES = [
    ("mspline", MSpline, (0.0, 1.0)),
    ("bspline", BSpline, (0.0, 1.0)),
    ("logx", LogXBSpline, (0.001, 1.0)),
    ("logy", LogYBSpline, (0.0, 1.0)),
    ("logxlogy", LogXLogYBSpline, (0.001, 1.0)),
]

# (label, class, xrange) for the 1-D models in single.py that take (n, xx, xx_inj).
MODEL_CASES = [
    ("Base1DBSplineModel", Base1DBSplineModel, (0.0, 1.0)),
    ("BSplineSpinMagnitude", BSplineSpinMagnitude, (0.0, 1.0)),
    ("BSplineSpinTilt", BSplineSpinTilt, (-1.0, 1.0)),
    ("BSplineChiEffective", BSplineChiEffective, (-1.0, 1.0)),
    ("BSplineSymmetricChiEffective", BSplineSymmetricChiEffective, (-1.0, 1.0)),
    ("BSplineChiPrecess", BSplineChiPrecess, (0.0, 1.0)),
    ("BSplineRatio", BSplineRatio, (0.0, 1.0)),
    ("BSplineMass", BSplineMass, (2.0, 100.0)),
]


def probe_points(xrange):
    """Boundary and out-of-range points first, then a dense interior sweep.

    The leading points are the ones a restructure is most likely to break: both boundaries are
    inclusive, and the out-of-range and non-positive entries exercise the out-of-bounds convention.
    """
    lo, hi = xrange
    edges = np.array([lo, hi, lo - 1e-3, hi + 1e-3, 0.0, -1.0])
    return np.concatenate([edges, np.linspace(lo, hi, 200)])


def capture_bases():
    """Every basis class, with normalize both off and on."""
    out = {}
    rng = np.random.default_rng(SEED)
    # Both signs matter: log-y bases take real coefficients, linear-y ones want positive.
    coefs = {"uniform": rng.uniform(size=N_DF), "normal": rng.normal(size=N_DF)}

    for label, cls, xrange in BASIS_CASES:
        for normalize in (False, True):
            spl = cls(N_DF, xrange=xrange, normalize=normalize)
            tag = f"basis/{label}/{normalize}"
            xs = probe_points(xrange)

            out[f"{tag}/knots"] = np.asarray(spl.knots)
            out[f"{tag}/N"] = np.asarray(spl.N)
            out[f"{tag}/order"] = np.asarray(spl.order)
            out[f"{tag}/xrange"] = np.asarray(spl.xrange)
            out[f"{tag}/basis_vols"] = np.asarray(spl.basis_vols)

            dmat = spl.bases(xs)
            out[f"{tag}/bases"] = np.asarray(dmat)

            for cs_label, cs in coefs.items():
                # Project twice from the same array: pins whether project() mutates or rescales it.
                out[f"{tag}/project/{cs_label}"] = np.asarray(spl.project(dmat, jnp.asarray(cs)))
                out[f"{tag}/project_repeated/{cs_label}"] = np.asarray(spl.project(dmat, jnp.asarray(cs)))
                out[f"{tag}/norm/{cs_label}"] = np.asarray(spl.norm(jnp.asarray(cs)))

            if normalize:
                out[f"{tag}/grid"] = np.asarray(spl.grid)
                out[f"{tag}/grid_bases"] = np.asarray(spl.grid_bases)
    return out


def build_models():
    """Instantiate the nine 1-D models with synthetic samples inside each one's domain."""
    rng = np.random.default_rng(SEED + 1)

    def samples(lo, hi):
        return jnp.asarray(rng.uniform(lo, hi, size=(3, 50))), jnp.asarray(rng.uniform(lo, hi, size=200))

    models = [(label, cls(N_SPLINES, *samples(lo, hi))) for label, cls, (lo, hi) in MODEL_CASES]

    # BSplineRedshift additionally needs the differential comoving volume at each sample.
    z, z_inj = samples(0.01, 2.0)
    dVdc = PLANCK_2015_LVK_Cosmology.dVcdz(z)
    dVdc_inj = PLANCK_2015_LVK_Cosmology.dVcdz(z_inj)
    models.append(("BSplineRedshift", BSplineRedshift(N_SPLINES, z, z_inj, dVdc, dVdc_inj)))
    return models


def capture_models():
    out = {}
    rng = np.random.default_rng(SEED + 2)
    coefs = {"uniform": rng.uniform(size=N_SPLINES), "normal": rng.normal(size=N_SPLINES)}

    for label, model in build_models():
        tag = f"model/{label}"
        out[f"{tag}/knots"] = np.asarray(model.interpolator.knots)
        out[f"{tag}/normalize"] = np.asarray(model.interpolator.normalize)
        for cs_label, cs in coefs.items():
            out[f"{tag}/pe/{cs_label}"] = np.asarray(model(jnp.asarray(cs), pe_samples=True))
            out[f"{tag}/inj/{cs_label}"] = np.asarray(model(jnp.asarray(cs), pe_samples=False))
        if isinstance(model, BSplineRedshift):
            out[f"{tag}/zgrid"] = np.asarray(model.zgrid)
            out[f"{tag}/grid_bases"] = np.asarray(model.grid_bases)
            for cs_label, cs in coefs.items():
                out[f"{tag}/normalization/{cs_label}"] = np.asarray(model.normalization(jnp.asarray(cs)))
    return out


def capture():
    """Build every case and return a flat {key: float64 array} dict."""
    values = {}
    values.update(capture_bases())
    values.update(capture_models())
    # float32 -> float64 is lossless widening, so stored comparisons stay bit-exact.
    return {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}


class TestInterpolationGolden(unittest.TestCase):
    """Every captured value must be bit-identical to the recorded baseline.

    A pure rename or restructure has no licence to move a single bit. When a step is *meant* to
    change values, regenerate only the affected keys and record the before/after in the commit.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GOLDEN_PATH):
            raise unittest.SkipTest(f"missing {GOLDEN_PATH}; regenerate with --write")
        cls.golden = np.load(GOLDEN_PATH)
        cls.current = capture()

    def test_no_keys_added_or_removed(self):
        self.assertEqual(sorted(self.golden.files), sorted(self.current), msg="captured key set changed")

    def test_values_are_bit_identical(self):
        for key in sorted(self.golden.files):
            if key not in self.current:
                continue
            with self.subTest(key=key):
                # assert_array_equal treats NaNs in matching positions as equal, which is what we
                # want here: a NaN is a meaningful captured value, not a comparison failure.
                np.testing.assert_array_equal(self.current[key], self.golden[key], err_msg=f"golden mismatch for {key}")


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("refusing to overwrite the baseline; pass --write if that is what you mean")
    captured = capture()
    np.savez_compressed(GOLDEN_PATH, **captured)
    print(f"wrote {len(captured)} keys ({sum(v.size for v in captured.values())} values) to {GOLDEN_PATH}")
