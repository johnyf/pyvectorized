"""
Python vectorized numerical module

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
from warnings import warn

import numpy as np

from multi_plot import _grouper

def unitize(x, ord=None, axis=0):
    """Return unit vectors corresponding to columns of x.
    
    >>> ndim = 3
    >>> n_points = 100
    >>> vectors = np.random.random([ndim, n_points] )
    >>> unit_vectors = unitize(vectors)
    
    see also
    --------
    unitize_components
    
    @param x: matrix of vectors,
        default: column vectors
    @param ord: select norm to use, see numpy.linalg.norm
    @param axis: x contains vectors along this axis
    
    @return: matrix of unit vectors aligned to vectors in x
    @rtype: np.array(size(x) )
    """
    return x /np.linalg.norm(x, axis=axis)

def unitize_components(x, ord=None):
    pass

def dom2vec(domain, resolution):
    """Matrix of column vectors for meshgrid points.
    
    Returns a matrix of column vectors for the meshgrid
    point coordinates over a parallelepiped domain
    with the given resolution.
    
    example
    -------
    >>> domain = [0, 1, 0,2]
    >>> resolution = [4, 5]
    >>> q = domain2vec(domain, resolution)
    
    @param domain: extremal values of parallelepiped
    @type domain: [xmin, xmax, ymin, ymax, ...]
    
    @param resolution: # points /dimension
    @type resolution: [nx, ny, ...]
    
    @return: q = matrix of column vectors (meshgrid point coordinates)
    @rtype: [#dim x #points]
        
    See also vec2meshgrid, domain2meshgrid, meshgrid2vec.
    """
    domain = _grouper(2, domain)
    lambda_linspace = lambda (dom, res): np.linspace(dom[0], dom[1], res)
    axis_grids = map(lambda_linspace, zip(domain, resolution) )
    pnt_coor = np.meshgrid(*axis_grids)
    q = np.vstack(map(np.ravel, pnt_coor) )
    
    return q

def domain2vec(domain, resolution):
    """Alias for dom2vec.
    """
    return dom2vec(domain, resolution)

def domain2meshgrid(domain, resolution):
    """DOMAIN2MESHGRID(domain, resolution)   generate meshgrid on parallelepiped
       [X, Y] = DOMAIN2MESHGRID(domain, resolution) creates the matrices
       X, Y definining a meshgrid covering the 2D rectangular domain
       domain = [xmin, xmax, ymin, ymax] with resolution = [nx, ny] points
       per each coordinate dimension.
    
    usage
    -----
       [X, Y, Z] = DOMAIN2MESHGRID(domain, resolution) results into a
       meshgrid over a 3D parallelepiped domain.
    
    input
    -----
     (2D Case)
       domain = extremal values of parallelepiped
              = [xmin, xmax, ymin, ymax]
       resolution = # points /dimension
                  = [nx, ny]
    
     (3D Case)
       domain = [xmin, xmax, ymin, ymax, zmin, zmax]
       resolution = [nx, ny, nz]
    
    output
    ------
     (2D case)
       X = [ny x nx] matrix of grid point abscissas
       Y = [ny x nx] matrix of grid point ordinates
    
     (3D Case)
       X = [ny x nx x nz] matrix of grid point abscissas
       Y = [ny x nx x nz] matrix of grid point ordinates
       Z = [ny x nz x nz] matrix of grid point coordinates
    
    see also
        domain2vec, vec2meshgrid, meshgrid2vec, meshgrid
    """
    if domain.shape[0] != 1:
        raise Exception('size(domain, 1) ~= 1')
    
    ndim=domain.shape[1] / 2
    if not (ndim %1 == 0):
        raise Exception('Non-integer domain dimension.')
    
    res_ndim=resolution.shape[1]
    if res_ndim > ndim:
        warn('dom2meshgrid:res_ndim','size(resolution) = [' +
            str(resolution.shape) + '] ~= [' +
            str(np.array([1,ndim]).reshape(1,-1)) +
            '] = [1, ndim]')
    if res_ndim < ndim:
        raise Exception('size(resolution) = [' +
            str(resolution.shape) + '] ~= [' +
            str(np.array([1,ndim]).reshape(1,-1)) +
            '] = [1, ndim]')
    if ndim == 2:
        X,Y=linmeshgrid2d(domain,resolution) # nargout=2
    else:
        if ndim == 3:
            X,Y,Z=linmeshgrid3d(domain,resolution) # nargout=3
        else:
            msg='domain has more than 3 dimensions. Use vec2 directly.'
            warn('dom:dim4',msg)
    return X,Y,Z

def linmeshgrid2d(domain,resolution):
    (xmin, xmax, ymin, ymax) = domain
    (nx, ny) = resolution
    
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    
    X,Y = np.meshgrid(x, y) # nargout=2
    
    return X,Y

def linmeshgrid3d(domain,resolution):
    xmin=domain[0,0]
    xmax=domain[0,1]
    ymin=domain[0,2]
    ymax=domain[0,3]
    zmin=domain[0,4]
    zmax=domain[0,5]
    nx=resolution[0,0]
    ny=resolution[0,1]
    nz=resolution[0,2]
    x=np.linspace(xmin,xmax,nx)
    y=np.linspace(ymin,ymax,ny)
    z=np.linspace(zmin,zmax,nz)
    X,Y,Z=np.meshgrid(x,y,z) # nargout=3
    
    return X,Y,Z

def dom2grid(domain, resolution):
    """Alias of domain2meshgrid.
    """
    return domain2meshgrid(domain, resolution)

def scalar2meshgrid(f,X):
    """reshape row vector f to size of X
       [F] = SCALAR2MESHGRID(F, X) reshapes the row array of values of scalar
       function f to size(X), for use as arguments to SURF.
    
     input
       f = row array of scalar function values at grid points
         = [1 x #(meshgrid points) ]
    
       X = matrix of x coordinates of meshgrid points
         = [M1 x M2 x ... x MN]
    
     output
       f = matrix with size(X) of scalar function values at grid points
         = [M1 x M2 x ... x MN]
    
     Note
       VEC2MESHGRID can fully replace this function.
    
    see also
        domain2vec, vec2meshgrid, domain2meshgrid, meshgrid2vec.
    """
    f=reshape(f,X.shape)
    return f

def vec2meshgrid(vectors, resolution=None, X=None):
    """Reshape matrix of column vectors to meshgrid component matrices.
    
    Reshape column vectors to size of X component matrices
    reshapes the matrix of N-D column
    vectors v to N 2D matrices, each of size(X), for use as arguments of
    grid points to SURF or QUIVER, or as vector components to QUIVER.
    Each of the resulting N matrices is a component of the column vectors in v.
    
    @param v: matrix of column vectors
    @type v: [#dim x #vectors]

    @param X: matrix of x coordinates of meshgrid points
    @type X: [M1 x M2 x ... x MN]
    
    @return: [v1, v2, ..., vN] = vectors' component matrices with size(X)
    @rtype: [M1 x M2 x ... x MN]
    
    see also
        domain2vec, scalar2meshgrid, meshgrid2vec, domain2meshgrid
    """
    if resolution is None:
        resolution = X.shape
    
    XYZ = [np.reshape(row, resolution) for row in vectors]
    
    return np.array(XYZ)

def res2meshsize(resolution):
    """Convert resolution [nx, ny, nz] to meshgrid matrix
    dimensions, i.e., [ny, nx, nz].
    
    see also
        vsurf, polar_domain2vec, plot_mapping
    """
    if len(resolution) < 3:
        resolution = list(reversed(resolution) )
    elif resolution[2] == 1:
        resolution = resolution[:2]
        resolution = list(reversed(resolution) )
    elif resolution[0] == 1:
        resolution = resolution[1:]
    elif resolution[1] == 1:
        resolution = resolution[[0,2]]
    else:
        warn('vmeshgrid:resolution' +
            '3d resolution for 2d surface.')
    return resolution

def meshgrid2vec(x, y, z=None):
    """Convert meshgrid matrices to matrix of column vectors
    
       [q] = MESHGRIDVEC(xgv, ygv) takes the matrix of abscissas XGV and
       ordinates YGV of meshgrid points as returned by MESHGRID and arranges
       them as vectors comprising the columns of matrix Q.
    
       [q] = MESHGRIDVEC(xgv, ygv, zgv) does the same for the 3D case.
    
     input
       xgv = matrix of meshgrid points' abscissas
           = [#(points / y axis) x #(points / x axis) ] (2D case) or
           = [#(points / y axis) x #(points / x axis) x #(points / z axis) ]
             (3D case)
       ygv = matrix of meshgrid points' ordinates
           = similar dimensions with xgv
       zgv = matrix of meshgrid points' coordinates
           = similar dimensions with xgv
    
     output
       q = [#dim x #(meshgrid points) ]
    
    see also
        domain2vec, vec2meshgrid, domain2meshgrid, meshgrid
    """
    if z is None:
        q = np.array([x[:],y[:]]).reshape(1,-1).T
    else:
        q = np.array([x[:],y[:],z[:]]).reshape(1,-1).T
    return q

def polar_domain2vec(*varargin):
    """Convert Polar domain to points with Cartesian coordinates.
    
     usage
       [q, X, Y, u, Theta, Rho] = POLAR_DOMAIN2VEC(domain, resolution, qc)
    
     input
       domain = polar coordinate extrema
              = [theta_min, theta_max, rho_min, rho_max]
       resolution = number of grid points per dimension
                  = [n_theta, n_rho]
       qc = pole coordinate vector (polar coordinate system center)
          = [2 x 1] = [xc, yc].'
    
     output
       q = position vectors of grid points in Cartesian coordinates
         = [2 x #points] = [x; y], where #points = prod(resolution)
       X = abscissas of grid points
         = [resolution(2) x resolution(1) ]
       Y = ordinates of grid points
         = [resolution(2) x resolution(1) ]
       u = position vectors of grid points in polar coordinates
         = [2 x #points] = [theta; rho]
       Theta = angles of grid points
             = [resolution(2) x resolution(1) ]
       Rho = polar radii of grid points
           = [resolution(2) x resolution(1) ]
    
    see also
        cylindrical_domain2vec, domain2vec, vpolar2cart
    
    depends
     vpol2cart
    """
    nargin = len(varargin)
    if nargin > 0:
        domain = varargin[0]
    if nargin > 1:
        resolution = varargin[1]
    if nargin > 2:
        qc = varargin[2]
    if nargin < 3:
        qc=np.zeros(2,1)
    ndim1=domain.shape[1] / 2
    ndim2=resolution.shape[1]
    ndim=qc.shape[0]
    if (ndim != ndim1) or (ndim != ndim2):
        raise Exception('domain2vec:dim' +
            'Dimensions of domain, resolution, qc do' +
            ' not agree with each other.')
    u,Theta,Rho=domain2vec(domain,resolution) # nargout=3
    q=vpol2cart(u[0,:],u[1,:])
    q=bsxfun(plus,q,qc)
    res=res2meshsize(resolution)
    X,Y=vec2meshgrid(q,np.array([]),res) # nargout=2
    return q,X,Y,u,Theta,Rho

def cylindrical_domain2vec(*varargin):
    """Convert cylindrical domain to points with Cartesian coordinates.
    
     usage
       [q, X, Y, Z, u, Theta, Rho] = CYLINDRICAL_DOMAIN2VEC(domain, resolution, qc)
    
     input
       domain = cylindrical grid coordinate extrema
              = [theta_min, theta_max, rho_min, rho_max, z_min, z_max]
       resolution = number of grid points per dimension
                  = [n_theta, n_rho, n_z]
       qc = pole coordinate vector (cylindrical coordinate system center)
          = [3 x 1] = [xc, yc, zc].'
    
     output
       q = position vectors of grid points in Cartesian coordinates
         = [3 x #points] = [x; y; z], where #points = prod(resolution)
       X = abscissas of grid points
         = [resolution(1) x resolution(3) ]
       Y = ordinates of grid points
         = [resolution(1) x resolution(3) ]
       Z = applicates of grid points
         = [resolution(1) x resolution(3) ]
       u = position vectors of grid points in cylindrical coordinates
         = [3 x #points] = [theta; rho; z]
       Theta = angles of grid points
             = [resolution(1) x resolution(3) ]
       Rho = polar radii of grid points
           = [resolution(1) x resolution(3) ]
    
    see also
        polar_domain2vec, domain2vec, cyl2cart
    
    depends
       domain2vec, vpol2cart, res2meshsize, vec2meshgrid
    """
    nargin = len(varargin)
    if nargin > 0:
        domain = varargin[0]
    if nargin > 1:
        resolution = varargin[1]
    if nargin > 2:
        qc = varargin[2]
    if nargin < 3:
        qc=np.zeros(3,1)
    ndim1=domain.shape[1] / 2
    ndim2=resolution.shape[1]
    ndim=qc.shape[0]
    if (ndim != ndim1) or (ndim != ndim2):
        raise Exception('domain2vec:dim' +
            'Dimensions of: domain, resolution, qc do' +
            ' not agree with each other.')
    u,Theta,Rho,Z=domain2vec(domain,resolution) # nargout=4
    q=vpol2cart(u[0,:],u[1,:],u[2,:])
    q=bsxfun(plus,q,qc)
    res=res2meshsize(resolution)
    X,Y=vec2meshgrid(q,np.array([]),res) # nargout=2
    return q,X,Y,Z,u,Theta,Rho

def domain2box(dom):
    """Convert domain to box description.
    
    box = domain2box(dom)
    
    @param dom: axis extrema (like given to axis)
    @type dom: [1 x (2* #dim) ]
    
    @return: box = pairs of extrema (vectors of the 2 corner points)
    @rtype: [#dim x 2]
    
    see also
        box2domain, min_bounding_box_aligned, boxdomain, check_domain
    """
    n,m=dom.shape # nargout=2
    if (0 in dom.shape):
        warning('dom:empty','domain matrix is empty.')
    if n != 1:
        error('domain matrix has more than one rows.')
    if mod(m,2) != 0:
        error('domain matrix has odd number of columns.')
    ndim=m / 2
    box=reshape(dom.T,2,ndim).T
    return box

def box2domain(box):
    """Convert box to domain description.
    
    >>> dom = box2domain(box)
    
    @param box: pairs of extrema (vectors of the 2 corner points)
    @type box: [#dim x 2]
    
    @return: axis extrema (like those given to axis)
    @rtype: [1 x (2* #dim) ]
    
    see also
        domain2box, min_bounding_box_aligned,
        boxdomain, check_domain
    """
    ndim,m=box.shape # nargout=2
    if (0 in box.shape):
        warning('box:empty','box matrix is empty.')
    if m != 2:
        error('box matrix has more than two columns.')
    dom=reshape(box.T,1,2 * ndim)
    return dom

def boxdomain(*varargin):
    """BOXDOMAIN  Create a (hyper)cube, compatible with axes().
    
    >>> dom = BOXDOMAIN # unit hypercube
    >>> dom = BOXDOMAIN(n, sideL)
    
    @param ndim: space dimension
    @type ndim: \in\naturals
    
    @param sideL: cube's sides' length
    @type sideL: > 0 [default = 1]
    
    @return: cube domain in format compatible with axes()
    @rtype: [1 x (2*#dim) ] = [xmin, xmax, ymin, ymax, ... ]
          = [-L, L, -L, L, ... ], where: L = sideL
    
    see also
        axes, domain2vec, surf, check_domain, box2domain
    """
    nargin = len(varargin)
    if nargin > 0:
        ndim = varargin[0]
    if nargin > 1:
        sideL = varargin[1]
    if nargin < 1:
        ndim=2
        disp('No (hyper)cube dimension given, using: dim = 2')
    if nargin < 2:
        sideL=1
        disp('No (hyper)cube side length given, using: sideL = 1.')
    I=ones(1,ndim)
    dom=np.array([I,- I]).reshape(1,-1)
    dom=reshape(dom,1,2 * ndim)
    dom=sideL.dot(dom)
    return dom

def check_domain(domain,auto_domain):
    """subset relation of parallelepiped domains.
       check that auto_domain \subseteq domain
    
    domain = check_domain(domain, auto_domain)
    
    @param domain: [x1_min, x1_max, x2_min, x2_max, ..., xN_min, xN_max]
    @param auto_domain: [x1_min, x1_max, x2_min, x2_max, ..., xN_min, xN_max]
    
    @return: domain = [x1_min, x1_max, x2_min, x2_max, ..., xN_min, xN_max]
    
    see also
        boxdomain, domain2box, box2domain, min_bounding_box_aligned
    """
    disp('domain = [' + num2str(domain,3) + ']')
    disp('auto_domain = [' + num2str(auto_domain,3) + ']')
    if domain.shape[0] != 1:
        error('size(domain, 1) ~= 1')
    if auto_domain.shape[0] != 1:
        error('size(auto_domain, 1) ~= 1')
    ndim1=domain.shape[1] / 2
    ndim2=auto_domain.shape[1] / 2
    if ndim1 != ndim2:
        error('Number of dimensions different between domain and auto_domain')
    ndim=ndim1
    flag=false()
    for i in range(1,(ndim+1)):
        endpoints=np.array([2 * i - 1,2 * i]).reshape(1,-1)
        interval1=auto_domain[0,(endpoints-1)]
        interval2=domain[0,(endpoints-1)]
        if not  issubinterval(interval1,interval2):
            warning('nflearn:domain','auto_domain \\notsubseteq domain')
            flag=true()
    padding=1.1
    if flag:
        warning('nflearn:domainfix','Automatically changing domain')
        domain=padding * auto_domain
    return domain
    
def issubinterval(interval1,interval2):
    itis=1
    if interval1[0,0] < interval2[0,0]:
        itis=0
        return itis
    if interval2[0,1] > interval2[0,1]:
        itis=0
        return itis
    return itis

def min_bounding_box_aligned(data_column_vectors):
    """minimum volume axis-aligned box bounding the given vectors
    
    min_bounding_box_aligned(data_column_vectors)
    
    @param data_column_vectors: coordinates of data points as matrix of column
                             vectors
                        = [#dim x #pnts]
    
    @return box: [x1_min, x1_max, ..., x#dim_min, x#dim_max]
    
    see also
        check_domain, box2domain, domain2box, boxdomain
    """
    box=np.array([
        np.min(data_column_vectors,np.array([]),2),
        np.max(data_column_vectors,np.array([]),2)
    ]).reshape(1,-1)
    return box
