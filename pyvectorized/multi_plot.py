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
    
    see also
        axes
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

def mhold(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        multiax = varargin[0]
    if nargin > 1:
        onoff = varargin[1]
    """MHOLD(multiax, onoff)  set hold state of multiple axes.
    
       VHOLD(multiax, onoff) is a vectorized version of function hold.
       It sets the states of multiple axes objects with handles in matrix 
       multiax, to the states provided in onoff. Argument onoff can be a
       single string, 'on' or 'off', setting all the axes to that hold state,
       or a cell matrix of strings 'on', 'off', setting individual axes to
       (possibly) different states. Note that in the second case, matrix
       multiax and cell matrix onoff should have the same size, i.e.,
       size(multiax) should equal size(onoff), if onoff is a cell matrix.
    
    usage
        MHOLD(multiax, onoff)
    
    input
        multiax = matrix of handles to axes objects
                = [ax11, ax12, ..., ax1m;
                   ax21, ax22, ..., ax2m;
                   :     :     :    :
                   axn1, axn2, ..., axnm];
        onoff = hold states for axes objects with handles in array multiax
              = 'on' (to set all axes objects hold states to 'on') |
              = 'off' (to set all axes objects hold states to 'off') |
              = {'on', 'off', 'off', ..., } (to set individual hold states)
    
    see also
        hold
    """
    # axes handle provided ?
    if nargin < 1:
        multiax = gca
    # desired hold state provided ?
    if nargin < 2:
        onoff = 'on'
    n = multiax.shape[0]
    m = multiax.shape[1]
    if ischar(onoff):
        state = repmat([onoff], n, m)
    else:
        if iscell(onoff):
            s = onoff.shape
            if not  np.array_equal(s, np.array([n, m]).reshape(1, -1)):
                error('Matrix of axes handles "multiax" and of axes hold ' + 'states "onoff" have unequal sizes.')
            state = onoff
    for i in range(1, (n +1)):
        for j in range(1, (m +1)):
            curax = multiax[(i -1), (j -1)]
            curstate = state[(i -1), (j -1)]
            hold(curax, curstate)
    return

def mgrid(ax, varargin):
    """Set grid for multiple axes.
    
     usage
       MGRID(ax, varargin)
    
     input
       ax = row vector of axes object handles
          = [1 x #axes]
       varargin = args passed to grid (same for each axis)
                = 'on' | 'off' | anything valid for function grid.
    
     see also
         mview, gridhold
    
     note
       mfunc = multi-function, i.e., operates on multiple similar objects,
               e.g. axes objects
       vfunc = vectorised func, i.e., function which traditionally works with
               matrices or iteratively and has been vectorized, either like
               surf to work on vectors, or in the sense of parallelization.
    """
    n = max(ax.shape)
    for i in range(1, (n +1)):
        grid(ax[(i -1)], varargin[:])
    return

def mview(ax, m):
    """Set view settings for multiple axes.
    
     usage
       MVIEW(ax, m)
    
     input
       ax = multiple axes object handles
          = [1 x #axes]
       m = string
    
    see also
        mgrid, gridhold
    """
    nax = ax.shape[1]
    for i in range(1, (nax +1)):
        curax = ax[0, (i -1)]
        view(curax, m)
    return

def mcla(ax, varargin):
    for i in range(1, (ax.shape[1] +1)):
        cla(ax[0, (i -1)], varargin[:])
    return

def gridhold(ax):
    """Grid and hold on for multiple axes handles.
    
    @param ax: single | list of axes
    
    see also
    --------
        mgrid, mview
    """
    ax = _check_ax(ax)
    
    # single ax ?
    try:
        ax.grid(True)
        ax.hold(True)
    except:
        [i.grid(True) for i in ax]
        [i.hold(True) for i in ax]

def axeq(ax=None):
    """Wrapper for ax.axis('equal').
    
    @param ax: single | list of axes
    """
    ax = _check_ax(ax)
    
    # single ax ?
    try:
        ax.axis('equal')
        return
    except:
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
        return plt.gca()
    return ax
