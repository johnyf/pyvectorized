"""
Common 2D and 3D plot, quiver, text functions

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt

def dimension(ndarray):
    """dimension of ndarray
    
    - ndim == 1:
        dimension = 1
    - ndim == 2:
        dimension = shape[0]
    """
    if ndarray.ndim < 2:
        return ndarray.ndim
    return ndarray.shape[0]

def plot(x, ax=None, **kwargs):
    """Plot column of matrix as points.
    
    Plot points in matrix x in axes ax
    passing kwargs to matplotlib.plot.
    
    usage example: plot 10 random 3D points
    
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from pyvectorized import plot
    >>> ax = plt.gca();
    >>> ndim = 3;
    >>> npoints = 10;
    >>> x = np.rand(ndim, npoints);
    >>> h = plot(ax, x, 'ro');
    
    see also
        matplotlib.pyploy.plot, plot3d
    
    @param x: matrix of points to plot
    @type x: [#dim x #pnts] numpy.ndarray
    
    @param ax: axes object handle(s)
    @type ax: [1 x #axes] (same plot in each axes pair)
        | numpy.NaN (to turn off plotting)
        | else no plotting and warning
    @param args: plot formatting
    
    @return h: handle to plotted object(s)
    """
    # copy to multiple axes ?
    try:
        lines = [plot(x, i, **kwargs) for i in ax]
        return lines
    except:
        pass
    
    if not ax:
        ax = plt.gca()
    
    dim = dimension(x)
    
    # >3D ?
    if dim > 3:
        warn('plot: ndim = ' +str(x.ndim) +
             ' > 3, plotting only 3D component.')
    
    # select 2D or 3D
    if dim < 1:
        raise Exception('x.ndim == 0')
    elif dim < 2:
        line = ax.plot(x, **kwargs)
    elif dim < 3:
        line = ax.plot(x[0, :], x[1, :], **kwargs)
    else:
        line = ax.plot(x[0, :], x[1, :], x[2, :], **kwargs)
    
    return line

def quiver(x, v, ax=None, **kwargs):
    """Multi-dimensional quiver.
    
    Plot v columns at points in columns of x
    in axes ax with plot formatting options in kwargs.
    
    >>> import numpy as np
    >>> import matplotlib as mpl
    >>> from pyvectorized import quiver, dom2vec
    >>> x = dom2vec([0, 10, 0, 11], [20, 20])
    >>> v = np.vstack(np.sin(x[1, :] ), np.cos(x[2, :] ) )
    >>> quiver(mpl.gca(), x, v)
    
    see also
        plot, matplotlib.quiver, mayavi.quiver3
    
    @param x: points where vectors are based
        each column is a coordinate tuple
    @type x: 2d lil | numpy.ndarray
    
    @param v: vectors which to base at points x
    @type v: 2d lil | numpy.ndarray
    
    @param ax: axes handle, e.g., ax = gca())
    
    @param x: matrix of points where vectors are plotted
    @type x: [#dim x #points]
    
    @param v: matrix of column vectors to plot at points x
    @type v: [#dim x #points]
    
    @param kwargs: plot formatting
    
    @return: handle to plotted object(s)
    """
    # multiple axes ?
    try:
        fields = [quiver(x, v, i, **kwargs) for i in ax]
        return fields
    except:
        pass
    
    if not ax:
        ax = plt.gca()
    
    dim = dimension(x)
    
    if dim < 2:
        raise Exception('ndim < 2')
    elif dim < 3:
        h = ax.quiver(x[0, :], x[1, :],
                      v[0, :], v[1, :], **kwargs)
    else:
        raise NotImplementedError
        
        from mayavi.mlab import quiver3d
        
        if ax:
            print('axes arg ignored, mayavi used')
        
        h = quiver3d(x[0, :], x[1, :], x[2, :],
                     v[0, :], v[1, :], v[2, :], **kwargs)
    
    if dim > 3:
        warn('quiver:ndim #dimensions > 3,' +
             'plotting only 3D component.')
    
    return h

def text(x, string, ax=None, **kwargs):
    """Text annotation in 2D or 3D.
    
    text position x:
        - x.ndim == 1: single point:
            - 2-tuple: 2d plot
            - n-tuple: nd plot
        - x.ndim > 1: each column a point:
            - [2 x #points]: 2D plot
            - [n x #points]: nd plot
    
    see also
        plot, quiver,
        matplotlibpyplot.text,
        mpl_toolkits.mplot3d.Axes3D.text
    
    @param x: point where text is placed
    @type x: [#dim x 1]
    
    @param str: annotation text string
    """
    # multiple axes ?
    try:
        h = [text(x, string, ax=i, **kwargs) for i in ax]
    except:
        pass
    
    if not ax:
        ax = plt.gca()
    
    dim = dimension(x)
    if dim < 2:
        raise Exception('ndim < 2')
    elif dim < 3:
        h = ax.text(x[0, :], x[1, :], string, **kwargs)
    else:
        if dim > 3:
            print('>3 dimensions, only first 3 plotted')
        
        h = ax.text(x[0, :], x[1, :], x[2, :],
                    string, **kwargs)
    
    return h

def vtext(q, num=None, ax=None, **kwargs):
    """Label points in q with numbers from num.
    
    see also
        text, plot
    
    @param ax: axes object handle | []
    
    @param q: matrix whose each column stores point coordinates
    @type q: [#dim x #points]
    
    @param num: array of numbers to use for annotation
    @type num: [1 x #points]
    
    @param kwargs: additional arguments passed to text
    """
    # axes ?
    if not ax:
        ax = plt.gca()
    
    # numbers ?
    if not num:
        num = range(1, (q.shape[1] +1))
    
    # plot
    strings = str(num.T)
    text(q, strings, axes=ax, **kwargs)
