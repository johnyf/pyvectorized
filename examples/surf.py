"""
surf example
"""
from pyvectorized import dom2vec, surf, \
    newax, axis, axeq, hold, grid, gridhold

def myfun(x):
    return x[0,:]**2 +x[1,:]**3

domain = [0, 1,0, 2]
resolution = [20, 30]

q = dom2vec(domain, resolution)

f = myfun(q)

ax, fig = newax(1, dim=[3])
surf(q, f, resolution, ax=ax)

axis(ax, domain +[0, 10])
axeq(ax)

"""manage multiple axes"""
ax2, fig = newax(1, dim=[3])
surf(q, -f, resolution, ax=ax2)

axs =[ax, ax2]

hold(axs)
grid(axs)

# shorthand for above
gridhold(ax)

# uncomment these if outside ipython
#from matplotlib import pyplot as plt
#plt.show(block=True)
