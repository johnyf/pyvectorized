"""
PyVectorized toolbox

Functions for unified and vectorized gridding,
plotting and numerics in any dimension,
by manipulating columns of vectors,
instead of coordinates separated into matrices.

2013 (BSD-3) California Institute of Technology
"""
__version__ = "0.1"

from multi_plot import newax
from vectorized_meshgrid import dom2vec
from multidim_plot import plotmd, quivermd, textmd
from vectorized_plot import vsurf
