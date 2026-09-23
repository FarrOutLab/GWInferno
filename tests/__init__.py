"""Test package.

gwinferno.postprocess.plot calls plt.show() unconditionally, so tests that touch the plotting
helpers block on an interactive matplotlib backend until a window is closed by hand. That stalls
local runs and would hang a headless worker. Selecting a non-interactive backend here covers
`unittest discover`, which does not read conftest.py; the conftest at the repository root does the
same for pytest. It is set through the environment rather than matplotlib.use() so that it applies
regardless of when matplotlib is first imported.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
