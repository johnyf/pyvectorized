"""
plot 2d, 3d
"""
from pyvectorized import newax, dom2vec, \
    plot, quiver, text, axis
import numpy as np

x = np.arange(10)
y = np.square(x)
q2 = np.vstack([x, y])

t = np.arange(10)
q = np.vstack([t, np.sin(t), np.cos(t) ])

ax, fig = newax(2, dim=[2,3])

plot(q2, ax[0])
plot(q, ax[1])

"""
quiver 2d
"""
dom = [-10, 10, -10, 10]
res = [10, 10]

x = dom2vec(dom, res)
A = np.array([[1, -2],
              [2, 1] ])
v = A.dot(x)

ax, fig = newax()
quiver(x, v, ax)

"""
quiver 3d
"""
dom = [-1, 1, -1, 1, -1, 1]
res = [5, 5, 5]

x = dom2vec(dom, res)
A = np.array([[1, -2, 3],
              [2, 1, 4],
              [9, 3, -3] ])
v = A.dot(x)

#quiver(x, v)

"""
tex 2d
"""
ax, fig = newax(2, dim=[2, 3])

x = np.array([[1, 1]]).transpose()
text(x, 'hehe 2d', ax[0])

limits = [0.5, 2, 0.5, 1.5]
ax[0].axis(limits)

"""
text 3d
"""
x = np.array([[1, 1, 2]]).transpose()
text(x, 'hehe 3d', ax[1])

limits = [0, 2, 0, 2, 1, 3]
axis(ax[1], limits)

# uncomment these if outside ipython
#from matplotlib import pylab as plt
#plt.show(block=True)
