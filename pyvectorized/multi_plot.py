"""
Operate on many plots at once

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
from warnings import warn
from itertools import izip_longest

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#from mayavi import mlab

def newax(subplots=(1,1), fig=None,
          mode='list', dim=2):
    """Create (possibly multiple) new axes handles.
    
    @param fig: attach axes to this figure
    @type fig: figure object,
        should be consistent with C{dim}
    
    @param subplots: number or layout of subplots
    @type subplots: int or
        2-tuple of subplot layout
    
    @param mode: return the axes shaped as a
        vector or as a matrix.
        This is a convenience for later iterations
        over the axes.
    @type mode: 'matrix' | ['list']
    
    @param dim: plot dimension:
        
            - if dim == 2, then use matplotlib
            - if dim == 3, then use mayavi
        
        So the figure type depends on dim.
    
    @return: C{(ax, fig)} where:
        - C{ax}: axes created
        - C{fig}: parent of ax
    @rtype: list or list of lists,
        depending on C{mode} above
    """
    # layout or number of axes ?
    try:
        subplot_layout = tuple(subplots)
    except:
        subplot_layout = (1, subplots)
    
    # reasonable layout ?
    if len(subplot_layout) != 2:
        raise Exception('newax:' +
            'subplot layout should be 2-tuple or int.')
    
    # which figure ?
    if fig is None:
        fig = plt.figure()
    
    # create subplot(s)
    (nv, nh) = subplot_layout
    n = np.prod(subplot_layout)
    
    try:
        dim = tuple(dim)
    except:
        # all same dim
        dim = [dim] *n
    
    # matplotlib (2D) or mayavi (3D) ?
    ax = []
    for (i, curdim) in enumerate(dim):
        if curdim == 2:
            curax = fig.add_subplot(nv, nh, i+1)
            ax.append(curax)
        else:
            curax = fig.add_subplot(nv, nh, i+1, projection='3d')
            ax.append(curax)
                      
        if curdim > 3:
            warn('ndim > 3, but plot limited to 3.')
    
    if mode is 'matrix':
        ax = list(grouper(nh, ax) )
    
    # single axes ?
    if subplot_layout == (1,1):
        ax = ax[0]
    
    return (ax, fig)

def hold(ax, b=True):
    """Set hold state of axes.
    
    @param ax: if None use gca
    @type ax: single | list of axes
    
    @param b: set hold on
        if single value, same for each ax
        otherwise len(b) must equal len(ax)
    @type b: bool
    
    @param b: set grid on
    @type b: bool
    """
    ax = _check_ax(ax)
    
    try:
        if not len(b) == len(ax):
            raise Exception('pyvectorized.hold: ' +
                'len(ax) != len(b)')
    except:
        b = len(ax) *[b]
    
    [i.hold(s) for (i, s) in zip(ax, b)]

def grid(ax=None, b=True, **kw):
    """Set grid for multiple axes.
    
    @param ax: if None use gca
    @type ax: single | list of axes
    
    @param b: set grid on
    @type b: bool
    """
    ax = _check_ax(ax)
    [i.grid(b, **kw) for i in ax]

def cla(ax=None):
    """Clear single or multiple axes.
    
    @param ax: if None use gca
    @type ax: single | list of axes
    """
    ax = _check_ax(ax)
    [i.cla() for i in ax]

def gridhold(ax=None, b=True):
    """Grid and hold on for multiple axes.
    
    @param ax: if None use gca
    @type ax: single | list of axes
    
    @param b: set grid and hold on
    @type b: bool
    """
    ax = _check_ax(ax)
    [i.grid(b) for i in ax]
    [i.hold(b) for i in ax]

def axeq(ax=None):
    """Wrapper for ax.axis('equal').
    
    @param ax: if None use gca
    @type ax: single | list of axes
    """
    ax = _check_ax(ax)
    [i.axis('equal') for i in ax]

def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def axis(ax, limits):
    ax.set_xlim(limits[0:2])
    ax.set_ylim(limits[2:4])
    
    if len(limits) <= 4:
        return
    
    ax.set_zlim(limits[4:6])

def _check_ax(ax):
    """Helper
    """
    if ax is None:
        return [plt.gca()]
    
    # single ax ?
    try:
        len(ax)
        return ax
    except:
        return [ax]
