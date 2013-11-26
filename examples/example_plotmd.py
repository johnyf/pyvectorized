"""
plotmd example
"""
from pyvectorized import newax, plotmd
import numpy as np

x = np.arange(10)
y = np.square(x)
q2 = np.vstack([x, y])

t = np.arange(10)
q = np.vstack([t, np.sin(t), np.cos(t) ])

ax, fig = newax(2, dim=[2,3])

plotmd(q2, ax[0])
plotmd(q, ax[1])

# uncomment these if outside ipython
#from matplotlib import pylab as plt
#plt.show(block=True)
