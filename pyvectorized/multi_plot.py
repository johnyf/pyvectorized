"""
Operate on many plots at once

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
import numpy as np

def newax(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        fig = varargin[0]
    if nargin > 1:
        subplot_number = varargin[1]
    if nargin > 2:
        mode = varargin[2]
    """Create (possibly multiple) new axes handles.
    
    usage
        [ax, fig] = newax
        ax = newax(fig, subplot_number)
       
    input
        fig = figure object handle
        subplot_number = either number of subplots, or [1 x 2] matrix of
                         subplot layout
        mode = ['matrix'] | 'vec'
    
    output
        ax = handles of axes objects, either vector or matrix,
             depending on 'mode' above
        fig = parent figure handle
    
    see also
        axes
    """
    # which figure ?
    if nargin < 1:
        fig = figure
    else:
        if (0 in fig.shape):
            fig = figure
    # how many subplots ?
    if nargin < 2:
        subplot_number = 1
    if nargin < 3:
        mode = 'matrix'
    # single axes ?
    if isscalar(subplot_number) and (subplot_number == 1):
        ax = axes('Parent', fig)
        return ax, fig
    # layout specified ?
    if not  isscalar(subplot_number):
        subplot_layout = subplot_number
        subplot_number = prod(subplot_number)
    else:
        subplot_layout = np.array([1, subplot_number]).reshape(1, -1)
    # layout reasonable ?
    if subplot_layout.shape[0] != 1:
        error('newax:layout', 'Subplot layout should be a [1 x 2] matrix.')
    # create multiple subplots
    nv = subplot_layout[0, 0]
    nh = subplot_layout[0, 1]
    ax = nan(1, subplot_number)
    for i in range(1, (subplot_number +1)):
        ax[0, (i -1)] = subplot(nv, nh, i, 'Parent', fig)
    if mode == 'matrix' == 1:
        ax = reshape(ax, nv, nh)
    else:
        if mode == 'linear' != 1:
            error('Unknown mode')
    return ax, fig

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

def gridhold(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    """Grid and hold on for multiple axes handles.
    
     usage
       GRIDHOLD(ax)
    
     input
       ax = vector of axes object handles
          = [1 x #axes]
    
    see also
        mgrid, mview
    """
    if nargin < 1:
        ax = gca
    for i in range(1, (max(ax.shape) +1)):
        grid(ax[(i -1)], 'on')
        hold(ax[(i -1)], 'on')
    return

def axiseq(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    """Wrapper for axis(ax, 'equal') to avoid rewriting it.
    
    usage
       AXISEQ(ax)
    
    input
       ax = axes object handle
    """
    if nargin < 1:
        ax = gca
    for i in range(1, (ax.shape[1] +1)):
        axis(ax[0, (i -1)], 'equal')
    return
