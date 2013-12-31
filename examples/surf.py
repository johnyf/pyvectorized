"""
vsurf example
"""
from pyvectorized import dom2vec, surf, newax

def myfun(x):
    return x[0,:]**2 +x[1,:]**3

domain = [0, 1,0, 2]
resolution = [20, 30]

q = dom2vec(domain, resolution)

f = myfun(q)

ax, fig = newax(1, dim=[3])
surf(q, f, resolution, ax=ax)

# uncomment these if outside ipython
#from matplotlib import pyplot as plt
#plt.show(block=True)
