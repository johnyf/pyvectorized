"""
Common 2D and 3D plot, quiver, text functions

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt

def plotmd(x, ax=None, **kwargs):
    """Plot column of matrix as points.
    
    PLOTMD(AX, X, VARARGIN) plots the points in matrix X in the axes with
    handle AX using the plot formatting options in VARARGIN. X must be
    a matrix whose columns are the 2D or 3D vectors to plot.
    
    usage
        h = plotmd(ax, x, varargin)
    
    @param x: matrix of points to plot
    @type x: [#dim x #pnts] numpy.ndarray
    
    @param ax: axes object handle(s)
           = [1 x #axes] (to plot the same thing in all the axes provided)
           = numpy.NaN (to turn off plotting)
           else no plotting and warning
    @param args: plot formatting
    
    @return h: handle to plotted object(s)
    
    usage example: plot 10 random 3D points
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from pyvectorized import plotmd
    >>> ax = plt.gca();
    >>> ndim = 3;
    >>> npoints = 10;
    >>> x = np.rand(ndim, npoints);
    >>> h = plotmd(ax, x, 'ro');
    
    see also
        matplotlib.pyploy.plot, plot3d
    """
    # ax empty ?
    if not ax:
        warn('plotmd: ' +
             'empty ax, no plotting.')
        return None
    
    # no plotting output (silent)
    if ax is np.NaN:
        print('Axes handle is nan. No graphical output.')
        return None
    
    # >3D ?
    if x.ndim > 3:
        warn('plotmd: ndim = ' +str(x.ndim) +
             ' > 3, plotting only 3D component.')
    
    # copy to multiple axes ?
    try:
        lines = []
        for curax in ax:
            line = plotmd(x, ax=curax, **kwargs)
            lines.append(line)
        return lines
    except:
        pass
    
    # select 2D or 3D
    dim = x.shape[0]
    if dim == 1:
        #n = x.shape[1]
        #range(n) +1
        line = ax.plot(x[0, :], **kwargs)
    elif dim == 2:
        line = ax.plot(x[0, :], x[1, :], **kwargs)
    elif dim >= 3:
        line = ax.plot(x[0, :], x[1, :], x[2, :], **kwargs)
    
    return line

def quivermd(ax, x, v, varargin):
    """Multi-dimensional quiver.
    
    QUIVERMD(AX, X, V, VARARGIN) plots the column vectors in matrix V
    at the points with coordinates the column vectors in matrix X
    within axes object AX using plot formatting options in VARARGIN.
    
    usage
        H = QUIVERMD(AX, X, V, VARARGIN)
    
    input
        ax = axes handle (e.g. ax = gca)
        x = matrix of points where vectors are plotted
          = [#dim x #points]
        v = matrix of column vectors to plot at points x
          = [#dim x #points]
        varargin = plot formatting
    
    output
        h = handle to plotted object(s)
    
    example
        x = linspace(0, 10, 20);
        y = linspace(0, 10, 20);
        [X, Y] = meshgrid(x, y);
        x = [X(:), Y(:) ].';
        v = [sin(x(1, :) ); cos(x(2, :) ) ];
        quivermd(gca, x, v)
    
    see also
        plotmd, quiver, quiver3
    """
    # multiple axes ?
    nax = ax.shape[1]
    if nax > 1:
        for i in range(1, (nax +1)):
            curax = ax[0, (i -1)]
            quivermd(curax, x, v, varargin[:])
        return varargout
    ndim = x.shape[0]
    if ndim > 3:
        warning('quivermd:ndim', '#dimensions > 3, plotting only 3D component.')
    if ndim == 2:
        h = quiver(ax, x[0, :], x[1, :], v[0, :], v[1, :], varargin[:])
    else:
        if ndim >= 3:
            h = quiver3(ax, x[0, :], x[1, :], x[2, :], v[0, :], v[1, :], v[2, :], varargin[:])
    if nargout == 1:
        varargout[0, 0] = h
    return varargout

def textmd(x, str_, varargin):
    """Text annotation in 2D or 3D.
    
    usage
        TEXTMD(x, str, varargin)
    
    input
        x = point where text is placed
          = [#dim x 1]
        str = annotation text string
     
    see also
        plotmd, quivermd
    """
    if any('Parent' == varargin):
        idx = np.flatnonzero('Parent' == varargin)
        ax = varargin[(idx -1), (+ 1 -1)]
        nax = ax.shape[1]
    else:
        nax = 1
    if nax > 1:
        for i in range(1, (nax +1)):
            curax = ax[0, (i -1)]
            v = varargin
            v[(idx + 1 -1)] = curax
            textmd(x, str_, v[:])
        return varargout
    ndim = x.shape[0]
    if ndim == 2:
        y = x[1, :]
        x = x[0, :]
        h = text(x, y, str_, varargin[:])
    else:
        if ndim == 3:
            z = x[2, :]
            y = x[1, :]
            x = x[0, :]
            h = text(x, y, z, str_, varargin[:])
    if nargout == 1:
        varargout[0, 0] = h
    return varargout

def textmd2(ax, x, string, varargin):
    """textmd wrapper compatible with 
    
    usage
       textmd2(ax, x, string, varargin)
    
    input
        ax = axes object handle
        x = coordinates of string position, see text
        string = text annotation
        varargin = directly passed to text
    
    see also
        textmd, text
    """
    textmd(x, string, 'Parent', ax, varargin[:])
    return

def vtextmd(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        q = varargin[1]
    if nargin > 2:
        num = varargin[2]
    if nargin > 3:
        varargin = varargin[3]
    """Add numbers as text to multiple 2/3D points.
    
    VTEXTMD(ax, q, num) adds the numbers in the numeric array num as text
    labels to the points with position vectors the columns of q.
    
    usage
        vtextmd(ax, q)
        vtextmd(ax, q, num, varargin)
    
    input
        ax = axes object handle | []
        q = point coordinates
          = [#dim x #points]
    
    optional input
        num = array of numbers to use for annotation
            = [1 x #points]
        varargin = additional arguments passed to text
    
    see also
        textmd, text, plotmd
    """
    # axes ?
    if (0 in ax.shape):
        ax = gca
    # numbers ?
    if nargin < 3:
        num = range(1, (q.shape[1] +1))
    # plot
    str_ = num2str(num.T)
    textmd(q, str_, 'Parent', ax, varargin[:])
    return
