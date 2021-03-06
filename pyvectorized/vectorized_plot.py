"""
Vectorized surf, contour and quiver

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
#from mayavi import mlab

from multi_plot import newax
from vectorized_meshgrid import domain2vec, vec2meshgrid, res2meshsize
from multidim_plot import quiver

def ezquiver(
    func,
    domain = np.array([0, 1, 0, 1]).reshape(1, -1),
    resolution = np.array([30, 29]).reshape(1, -1),
    ax = None,
    *args, **kwargs
):
    """Vectorized quiver for functions with vector args.
    
    @param func: function handle
    @param domain: rectangular plotting domain
               = [xmin, xmax, ymin, ymax]
    @param resolution: grid spacing
                   = [nx, ny]
    @param ax: axes object handle
    @param args: positional arguments forwarded to func
    @param kwargs: key-value args for func
    """
    if ax is None:
        ax = newax()
    
    q = domain2vec(domain, resolution)
    v = feval(func, q, **kwargs)
    quiver(ax, q, v)

def contour(q, z, res, ax=None, **kwargs):
    """Vectorized wrapper for contour plot.
    
    @param q: coordinates of contour points
    @type q: [2 x #points]
    
    @param z: row vector of scalar function
    @type z: [1 x #points]
    
    @param res: resolution of contour
    @type res: [nx, ny]
    
    @param ax: axes object handle
    
    @param kwargs: passed to contour function as given, see its help
    
    @return: h = handle to contour object created.
    """
    if ax is None:
        ax = newax()
    
    # multiple axes ?
    try:
        contours = []
        for curax in ax:
            cont = contour(curax, q, z, res, **kwargs)
            contours.append(cont)
        return contours
    except:
        pass
    
    # enough dimensions ?
    ndim = q.shape[0]
    if ndim < 2:
        raise Exception('space dim = q.shape[0] = 1.')
    
    res = res2meshsize(res)
    
    # Z ?
    if z is None:
        z = np.zeros(1, q.shape[1])
    elif z is np.NaN:
        z = 5 * np.random.rand(1, q.shape[1])
    else:
        z = z * np.ones([1, q.shape[1] ])
    
    X, Y = vec2meshgrid(q, res) # nargout=2
    Z, = vec2meshgrid(z, res)
    
    # calc
    if ndim < 3:
        cont = ax.contour(X, Y, Z, 100, **kwargs)
    else:
        raise Exception('Dimension of vector q is not 2.')
    
    return cont

def contourf(q, z, res, ax=None, **kwargs):
    """Vectorized filled contour plot.
    
    @param q: coordinates of contour points
    @type q: [2 x #points] |
    
    @param z: row vector of scalar function
    @type z: [1 x #points]
    
    @param res: resolution of contour
    @type res: [nx, ny]
    
    @param ax: axes object handle
    
    @return: h = handle to filled contour
    """
    if ax is None:
        ax = newax()
    
    # multiple axes ?
    try:
        contours = []
        for curax in ax:
            cont = contourf(curax, q, z, res, **kwargs)
            contours.append(cont)
        return contours
    except:
        pass
    
    # enough dimensions ?
    ndim = q.shape[0]
    if ndim < 2:
        raise Exception('space dim = q.shape[0] = 1.')
    
    res = res2meshsize(res)
    
    # Z ?
    if z is None:
        z = np.zeros(1, q.shape[1])
    elif z is np.NaN:
        z = 5 * np.random.rand(1, q.shape[1])
    else:
        z = z * np.ones([1, q.shape[1] ])
    
    X, Y = vec2meshgrid(q, res) # nargout=2
    Z, = vec2meshgrid(z, res)
    
    # calc
    if ndim < 3:
        cont = ax.contourf(X, Y, Z, 25, **kwargs)
    else:
        raise Exception('Dimension of vector q is not 2.')
    
    return cont

def ezcontour(func, ax, domain, resolution,
               values, **kwargs):
    """Vectorized easy contour,
    for functions accepting vector arguments.
    
    @param ax: axes object handle
    @param func: function handle
    
    @param domain: rectangular plotting domain
    @type domain: [xmin, xmax, ymin, ymax]
    
    @param resolution: grid spacing
    @type resolution: [nx, ny]
    
    @param values: level set values
    @type values: [v1, v2, ..., vN]
    
    @param kwargs: additional arguments for
        input to func
    """
    # which axes ?
    if (0 in ax.shape):
        warn('vezcontour:axes',
                'Axes object handle ax is empty, no plot.')
        return
    
    # which domain ?
    if not domain:
        domain = np.array([0, 1, 0, 1]).reshape(1, -1)
    
    # at what grid resolution ?
    if not resolution:
        resolution = np.array([30, 29]).reshape(1, -1)
    else:
        if (0 in resolution.shape):
            resolution = np.array([30, 29]).reshape(1, -1)
    
    # which level sets ?
    if not values:
        values = np.array([])
    
    # compute surface
    q, X, Y = domain2vec(domain, resolution) # nargout=3
    f = feval(func, q, **kwargs)
    Z = vec2meshgrid(f, X)
    
    # default level set values ?
    if (0 in values.shape):
        plt.contour(ax, X, Y, Z)
    else:
        plt.contour(ax, X, Y, Z, values)
    return

def ezsurf(func, domain, resolution, ax, **kwargs):
    """Vectorized ezsurf,
    for functions accepting vector arguments.
    
    input
       ax = axes object handle
       func = function handle
    
     optional input
       domain = rectangular plotting domain
              = [xmin, xmax, ymin, ymax]
       resolution = grid spacing
                  = [nx, ny]
       varargin = additional arguments for input to func
    
    @return (q, f) where:
        - q = domain points
        - f = function values at q
    """
    # which axes ?
    if (0 in ax.shape):
        warn('vezsurf:axes', 'Axes object handle ax is empty, no plot.')
        return varargout
    # which domain ?
    if not domain:
        domain = np.array([0, 1, 0, 1]).reshape(1, -1)
    # at what grid resolution ?
    if not resolution:
        resolution = np.array([30, 29]).reshape(1, -1)
    else:
        if (0 in resolution.shape):
            resolution = np.array([30, 29]).reshape(1, -1)
    q = domain2vec(domain, resolution)
    f = feval(func, q, varargin[:])
    vsurf(ax, q, f, resolution)
    if nargout > 0:
        varargout[0, 0] = q
        varargout[0, 1] = f
    return varargout

def surf(q, z, resolution,
          ax=None, **kwargs):
    """Vectorized surf.
    
    Vectorized wrapper for the surf function.
    When q is 2-dimensional, then z is the height function.
    When q is 3-dimensional, then z is the color function of the surface.
    
    see also
        mpl.plot_surface, vec2meshgrid
    
    @param ax: axes object handle
    @param q: coordinates of surface points
         = [2 x #points] |
         = [3 x #points], when color data are provided in vector z
    
    @param z: row vector of height or color data for surface points
    @type z: [1 x #points] | [], depending on the cases:
           
         1) when size(q, 1) == 2, then z is assumed to be the values of a
           scalar function to be plotted over the 2-dimensional domain defined
           by the points in the matrix of column position vectors q.
    
         2) when size(q, 1) == 3, then q are the coordinates of the points in
            3-dimensional space, whereas z is assumed to be the row vector
            specifying the surface color at each point.
         
         special cases:
         
             - [] (0 color)
             - NaN (random colors)
             - scalar (uniform color)
             - 'scaled' (scaled colors indexed in colormap)
    
    @param resolution: resolution of surface
    @type resolution: [nx, ny] | [nx, ny, nz]
    
    @return: surface object created.
    """
    # depends
    #   vec2meshgrid, res2meshsize
    
    if ax is None:
        ax = newax()
    
    # multiple axes ?
    try:
        surfaces = []
        for curax in ax:
            surface = surf(curax, q, z, resolution, **kwargs)
            surfaces.append(surface)
        return surfaces
    except:
        pass
    
    # enough dimensions for surf ?
    ndim = q.shape[0]
    if ndim < 2:
        raise Exception('space dim = q.shape[0] = 1.')
    
    resolution = res2meshsize(resolution)
    
    # Z ?
    if z is None:
        z = np.zeros(1, q.shape[1])
    elif z is np.NaN:
        z = 5 * np.random.rand(1, q.shape[1])
    elif z == 'scaled':
        z = np.array([])
    else:
        z = z * np.ones([1, q.shape[1] ])
    
    # calc
    if ndim < 3:
        surface = surf2(q, z, resolution, ax, **kwargs)
    else:
        surface = surf_color(q, z, resolution, ax, **kwargs)
    
    return surface

def surf2(q, z, res, ax, **kwargs):
    X, Y = vec2meshgrid(q, res) # nargout=2
    Z, = vec2meshgrid(z, res)
    h = ax.plot_surface(X, Y, Z, **kwargs)
    return h

def surf_color(q, c, res, ax, **kwargs):
    X, Y, Z = vec2meshgrid(q, res) # nargout=3
    
    # no color ?
    if (0 in c.shape):
        h = ax.plot_surface(X, Y, Z, **kwargs)
    else:
        C, = vec2meshgrid(c, res)
        h = ax.plot_surface(X, Y, Z, cmap=C, **kwargs)
    return h
