"""Pytest configuration shared by the whole suite.

Several tests exercise the plotting helpers in gwinferno.postprocess.plot. With an interactive
matplotlib backend those call plt.show() and block until a window is closed by hand, which stalls
local runs and would hang a headless CI worker. Selecting a non-interactive backend here fixes that
for every runner, and it is done via the environment rather than matplotlib.use() so that it takes
effect no matter when matplotlib is first imported.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
