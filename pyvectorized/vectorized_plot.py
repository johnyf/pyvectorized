"""
Vectorized surf, contour and quiver

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
#from mayavi import mlab

from multi_plot import newax
from vectorized_meshgrid import vec2meshgrid, res2meshsize

def ezquiver(
    func,
    domain = np.array([0, 1, 0, 1]).reshape(1, -1),
    resolution = np.array([30, 29]).reshape(1, -1),
    ax = None,
    *args, **kwargs
):
    """Vectorized quivermd for functions with vector args.
    
    @param func: function handle
    @param domain: rectangular plotting domain
               = [xmin, xmax, ymin, ymax]
    @param resolution: grid spacing
                   = [nx, ny]
    @param ax: axes object handle
    @param args: positional arguments forwarded to func
    @param kwargs: key-value args for func
    
    see also
        vezsurf, vezcontour, ezsurf, surf
    """
    if ax is None:
        ax = newax()
    
    q = domain2vec(domain, resolution)
    v = feval(func, q, varargin[:])
    quivermd(ax, q, v)
    return

def vcontour(ax, q, z, resolution, varargin):
    """Vectorized wrapper for contour plot.
    
    usage
        h = VCONTOUR(ax, q, z, resolution, varargin)
    
    input
        ax = axes object handle
        q = coordinates of surface points
          = [2 x #points] |
        z = row vector of height data for surface points
          = [1 x #points]
        resolution = resolution of surface
                   = [nx, ny]
        varargin = passed to contour function as given, see its help
     
    output
        h = handle to contourgroup object created.
    
    see also
        contour, vsurf, vec2meshgrid
    """
    # depends
    #   vec2meshgrid, res2meshsize
    
    # input
    # axes handle missing ?
    if (0 in ax.shape):
        ax = gca
    resolution = res2meshsize(resolution)
    # Z ?
    if (0 in z.shape):
        z = np.zeros(1, q.shape[1])
    else:
        if isnan(z):
            z = 5 * rand(1, q.shape[1])
            # random surface
        else:
            if isscalar(z):
                z = z * ones(1, q.shape[1])
    #
    # calc
    ndim = q.shape[0]
    if ndim < 3:
        h = vcontour2(ax, q, z, resolution, varargin[:])
    else:
        error('vcontour:ndim', 'Dimension of vector q is not 2.')
    if nargout < 1:
        clear('h')
    return h

def vcontour2(ax, q, z, res, varargin):
    X, Y = vec2meshgrid(q, np.array([]), res) # nargout=2
    Z = vec2meshgrid(z, np.array([]), res)
    h = contour(ax, X, Y, Z, varargin[:])
    return h

def vcontourf(ax, q, z, resolution, varargin):
    """Vectorized filled contour plot.
    
    usage
        h = VCONTOURF(ax, q, z, resolution, varargin)
    
    input
        ax = axes object handle
        q = coordinates of surface points
          = [2 x #points] |
        z = row vector of height data for surface points
          = [1 x #points]
        resolution = resolution of surface
                   = [nx, ny]
    
    output
        h = handle to filled contourgroup object created.
    
    see also
        VCONTOUR, VSURF, VEC2MESHGRID
    
    depends
        vcontour
    """
    h = vcontour(ax, q, z, resolution, 'Fill', 'on', varargin[:])
    if nargout < 1:
        clear('h')
    return h

def vezcontour(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        func = varargin[1]
    if nargin > 2:
        domain = varargin[2]
    if nargin > 3:
        resolution = varargin[3]
    if nargin > 4:
        values = varargin[4]
    if nargin > 5:
        varargin = varargin[5]
    """Vectorized easy contour,
    for functions accepting vector arguments.
    
    usage
        VEZCONTOUR(ax, func)
        VEZCONTOUR(ax, func, domain)
        VEZCONTOUR(ax, func, domain, resolution)
        VEZCONTOUR(ax, func, domain, resolution, values)
        VEZCONTOUR(ax, func, domain, resolution, values, varargin)
     
    input
        ax = axes object handle
        func = function handle
    
    optional input
        domain = rectangular plotting domain
               = [xmin, xmax, ymin, ymax]
        resolution = grid spacing
                   = [nx, ny]
        values = level set values
               = [v1, v2, ..., vN]
        varargin = additional arguments for input to func
    
    see also
        vezsurf, ezcontour, contour
    """
    # which axes ?
    if (0 in ax.shape):
        warning('vezcontour:axes', 'Axes object handle ax is empty, no plot.')
        return
    # which domain ?
    if nargin < 3:
        domain = np.array([0, 1, 0, 1]).reshape(1, -1)
    # at what grid resolution ?
    if nargin < 4:
        resolution = np.array([30, 29]).reshape(1, -1)
    else:
        if (0 in resolution.shape):
            resolution = np.array([30, 29]).reshape(1, -1)
    # which level sets ?
    if nargin < 5:
        values = np.array([])
    # compute surface
    q, X, Y = domain2vec(domain, resolution) # nargout=3
    f = feval(func, q, varargin[:])
    Z = vec2meshgrid(f, X)
    # default level set values ?
    if (0 in values.shape):
        contour(ax, X, Y, Z)
    else:
        contour(ax, X, Y, Z, values)
    return

def vezsurf(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        func = varargin[1]
    if nargin > 2:
        domain = varargin[2]
    if nargin > 3:
        resolution = varargin[3]
    if nargin > 4:
        varargin = varargin[4]
    """Vectorized ezsurf,
    for functions accepting vector arguments.
    
     usage
       VEZSURF(ax, func)
       VEZSURF(ax, func, domain)
       VEZSURF(ax, func, domain, resolution)
       [q, f] = VEZSURF(ax, func, domain, resolution, varargin)
    
    input
       ax = axes object handle
       func = function handle
    
     optional input
       domain = rectangular plotting domain
              = [xmin, xmax, ymin, ymax]
       resolution = grid spacing
                  = [nx, ny]
       varargin = additional arguments for input to func
    
     output
       q = domain points
       f = function values at q
    
    see also
        VEZCONTOUR, EZSURF, FPLOT, SURF
    """
    # which axes ?
    if (0 in ax.shape):
        warning('vezsurf:axes', 'Axes object handle ax is empty, no plot.')
        return varargout
    # which domain ?
    if nargin < 3:
        domain = np.array([0, 1, 0, 1]).reshape(1, -1)
    # at what grid resolution ?
    if nargin < 4:
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

def vsurf(q, z, resolution,
          ax=None, **kwargs):
    """Vectorized surf.
    
    Vectorized wrapper for the surf function.
    When q is 2-dimensional, then z is the height function.
    When q is 3-dimensional, then z is the color function of the surface.
    
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
    
    see also
        mpl.plot_surface, vcontour, vec2meshgrid
    """
    # depends
    #   vec2meshgrid, res2meshsize
    
    if ax is None:
        ax = newax()
    
    # multiple axes ?
    try:
        surfaces = []
        for curax in ax:
            surf = vsurf(curax, q, z, resolution, **kwargs)
            surfaces.append(surf)
        return h
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
        surf = vsurf2(q, z, resolution, ax, **kwargs)
    else:
        surf = vsurf_color(q, z, resolution, ax, **kwargs)
    
    return surf

def vsurf2(q, z, res, ax, **kwargs):
    X, Y = vec2meshgrid(q, res) # nargout=2
    Z, = vec2meshgrid(z, res)
    h = ax.plot_surface(X, Y, Z, **kwargs)
    return h

def vsurf_color(q, c, res, ax, **kwargs):
    X, Y, Z = vec2meshgrid(q, res) # nargout=3
    
    # no color ?
    if (0 in c.shape):
        h = ax.plot_surface(X, Y, Z, **kwargs)
    else:
        C, = vec2meshgrid(c, res)
        h = ax.plot_surface(X, Y, Z, cmap=C, **kwargs)
    return h
