"""
Plot fewer markers spaced by arclength or curve points

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt

from multidim_plot import plot

def parse_plot_style(style):
    """Helper
    """
    graph_color = 'b'
    marker_style = 'none'
    line_style = 'none'
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['.', 'o', 'x', '+', '*', 's', 'd', 'v', '^', '<', '>', 'p', 'h']
    linestyles = ['--', '-.', '-', ':']
    # order matters, to avoid strfinding
    # the '-' contained in '--', '-.'
    for s in colors:
        r = s[0]
        if strfind(style, r):
            graph_color = r
            break
    for s in markers:
        r = s[0]
        if strfind(style, r):
            marker_style = r
            break
    for s in linestyles:
        r = s[0]
        if strfind(style, r):
            line_style = r
            break
    # default line style
    if marker_style == 'none' and line_style == 'none':
        line_style = '-'
    return graph_color, marker_style, line_style

def plot2_fewer_markers(
    x, y, ax, style='None',
    marker_npnt=10, line_npnt=100, **kwargs
):
    """Same line, different number of markers.
    
    @param ax: axes object handle
    @param x: abscissa vector
    @param y: ordinate vector
    @param marker_npnt: # markers
    @param line_npnt: # points used for line
    @param style: line and marker style
        plot style string, e.g., 'ro--'
        | cell array {graph_color, line_style,
                      marker_style}, e.g.,
        {'b', '--', 'o'}, or {[0,0,1], '--', 'o'}
               note: the 2nd option allows for
               RGB color specification.
    """
    # defaults by parse_plot_style
    
    # cell with RGB color specs ?
    if iscell(style):
        graph_color = style[0]
        line_style = style[1]
        marker_style = style[2]
    else:
        # or string style ?
        graph_color, marker_style, line_style = parse_plot_style(style) # nargout=3
    #
    # plot
    c = takehold(ax, 'on')
    # marker of 1st pnt
    plot(ax, x[0], y[0], 'Color', graph_color, 'LineStyle', line_style, 'Marker', marker_style)
    # line
    plot2_subsample(ax, x, y, line_npnt, 'Color', graph_color, 'LineStyle', line_style, 'Marker', 'none', 'HandleVisibility', 'off')
    # markers
    plot2_subsample(ax, x, y, marker_npnt, 'Color', graph_color, 'LineStyle', 'none', 'Marker', marker_style, 'HandleVisibility', 'off')
    restorehold(ax, c)
    return

def plot2_subsample(x, y, ax, n=100, **kwargs):
    """Plot subsampled line. Subsampling metric: index.
    
    @param ax: axes object handle where to plot
    @param x: point abscissas (as passed to plot)
    @param y: point ordinates (as passed to plot)
    @param n: [100] number of points to keep > 0
    @param kwargs: arguments passed to plot
    """
    # assume row vectors
    if (x.shape[0] > 1) or (y.shape[0] > 1):
        print('size(x) = ')
        print(x.shape)
        print('size(y) = ')
        print(y.shape)
        raise Exception('Either x or y is not a row vector.')
    
    # plot
    xvec = np.array([x, y]).reshape(1, -1)
    xvec = subsample(xvec, n, 2)
    x, y = xvec # nargout=2
    plt.plot(ax, x, y, **kwargs)

def plot2_subsample_arclength(
    x, y, n=100, ax=None, **kwargs
):
    """Subsample wrt arclength metric and plot curve.
    
    @param ax: axes object handle where to plot
    @param x: point abscissas (as passed to plot)
    @param y: point ordinates (as passed to plot)
    @param n: number of points to keep > 0
    @param kwargs: arguments passed to plot
    """
    # assume row vectors
    if (x.shape[0] > 1) or (y.shape[0] > 1):
        print('size(x) = ' +str(x.shape) )
        print('size(y) = ' +str(y.shape) )
        raise Exception(
            'Either x or y is not a row vector.')
    
    # plot
    xvec = np.array([x, y]).reshape(1, -1)
    xvec = subsample(xvec, n, 2, 'arclength')
    x, y = disperse(xvec, 1) # nargout=2
    plt.plot(ax, x, y**kwargs)

def plot_fewer_markers(
    x, ax, marker_npnt, line_npnt,
    style, **kwargs
):
    """Same line, different number of markers.
    
    @param ax: axes object handle
    @param x: abscissa vector
    @param marker_npnt: # markers
    @param line_npnt: # points used for line
    @param style: line and marker style
         = plot style string, e.g., 'ro--'
         | cell array {graph_color, line_style,
                       marker_style}, e.g.,
           {'b', '--', 'o'}, or {[0,0,1], '--', 'o'}
           note: the 2nd option allows for
           RGB color specification.
    """
    # input
    if nargin < 3:
        style = ''
        # defaults by parse_plot_style
    if nargin < 4:
        marker_npnt = 10
    if nargin < 5:
        line_npnt = 100
    # cell with RGB color specs ?
    if iscell(style):
        graph_color = style[0]
        line_style = style[1]
        marker_style = style[2]
    else:
        # or string style ?
        graph_color, marker_style, \
        line_style = parse_plot_style(style) # nargout=3
    #
    # plot
    c = takehold(ax, 'on')
    # marker of 1st pnt for legend to get correct style
    plot(ax, x[:, 0], 'Color', graph_color,
         'LineStyle', line_style, 'Marker', marker_style)
    # line
    plot_subsample(ax, x, line_npnt, 'Color',
                   graph_color, 'LineStyle',
                   line_style, 'Marker', 'none',
                   'HandleVisibility', 'off')
    # markers
    plot_subsample(ax, x, marker_npnt, 'Color',
                   graph_color, 'LineStyle', 'none',
                   'Marker', marker_style,
                   'HandleVisibility', 'off')
    restorehold(ax, c)

def plot_subsample(x, n, ax, **kwargs):
    """Plot subsampled line. Subsampling metric: index.
    
    @param ax: axes object handle where to plot
    @param x: point abscissas (as passed to plot)
    @param y: point ordinates (as passed to plot)
    @param n: number of points to keep > 0
    @param kwargs: arguments passed to plot
    """
    if not isscalar(n):
        raise Exception('n: # of sample, must be scalar.')
    
    # plot
    x = subsample(x, n)
    plt.plot(ax, x, **kwargs)

def plot_subsample_arclength(x, n=100, ax, **kwargs):
    """Subsample wrt arclength metric & plot curve.
    
    @param ax: axes object handle where to plot
    @param x: point abscissas (as passed to plot)
    @param y: point ordinates (as passed to plot)
    @param n: number of points to keep > 0
    @param kwargs: arguments passed to plot
    """
    if not  isscalar(n):
        error('n: # of sample, must be scalar.')
    
    # plot
    x = subsample(x, n, 2, 'arclength')
    plot(ax, x, varargin[:])

def subsample(x, ntotal=100, dim=2, metric='index'):
    """Keep ntotal elements of x along dimension dim.
    
    Dfault method is to keep every Nth element, such that the result has
    ntotal elements.
    
    Better results for curves achieved using arclength metric.
    This keeps ntotal points which are equidistantly distributed wrt the
    curve's arclength between them.
    
    If another metric is desired (e.g. x = [n x n x k] is a stack of matrices
    and their distances are measured by the Frobenius norm), then provide it
    in argument 4 (in the example: metric = [1 x 1 x (k-1)] are the distances
    between successive elements along the 3rd dimension of x).
    
    note
       if metric == 'arclength', then x = [#dim x #pnts]
    
    @param x: matrix to subsample
    @type x: [n1 x n2 x ... x nK]
    @param ntotal: total number of elements to sample from x
    @param dim: dimension of x along which to subsample
         (i.e., within that slice)
    @param metric: choose method of subsampling,
        | provide inter-element distances
    @type metric: 'index'
        | 'arclength'
        | [1 x 1 x ... x n_dim x ... 1]
    
    @return: subsample x
    @rtype: [num1(x), num2(x), ..., num(k-1)(x), ntotal, num(k+1)(x), ... ]
           where numj(x) is size(x, j) and k=dim, the 3rd argument
    """
    n = x.shape[dim-1]
    default_n = 100
    if nargin < 2:
        ntotal = np.min(n, default_n)
        # avoid strange things
    #
    # calc indices
    if ischar(metric):
        if metric == 'index':
            idx = round_(linspace(1, n, ntotal))
        else:
            if metric == 'arclength':
                # too large ?
                if n > 10000:
                    warning('subsample:n',
                        'Too large # of vectors, using for loop.')
                    usefor = 1
                else:
                    usefor = 0
                dim = 2
                L, dLi = arclength(x) # nargout=2
                Li = np.cumsum(dLi,0)
                Li_desired = linspace(0, L, ntotal)
                # forloop or not ?
                if usefor == 0:
                    d = vdistance(Li, Li_desired)
                    __, idx = np.min(d, np.array([]), 1) # nargout=2
                else:
                    if usefor == 1:
                        idx = arclength_using_for(Li, Li_desired)
            else:
                error('Unknown metric: ' + metric)
    else:
        #Li = cumsum(metric, dim);
        #L = max_cell({Li} );
        raise Exception('code to be completed.')
    #
    # unique and missing ratio
    n_non_unique = idx.shape[1]
    idx = unique(idx)
    n_unique = idx.shape[1]
    if metric == 'arclength':
        disp('Unique/Non-Unique ratio: r = ' + num2str(100 * n_unique / n_non_unique) + '\n %')
        disp('Approximating with: n = ' + num2str(n_unique) + ' points')
    #
    # check #dims
    max_ndim = 15
    ndim = ndims(x)
    if ndim > max_ndim:
        error('larger than 15 dimensions not supported' + ' (just edit the code and add :)')
    #
    # subsample
    order = np.array([dim, omit(dim, range(1, (ndim +1)))]).reshape(1, -1)
    x = permute(x, order)
    x = x[(idx -1), :, :, :, :, :, :, :, :, :, :, :, :, :, :]
    # max ndim:edit here
    x = ipermute(x, order)
    return x

def arclength_using_for(Li, Li_desired):
    ntotal = Li_desired.shape[1]
    idx = nan(1, ntotal)
    for i in range(1, (ntotal +1)):
        curLi_desired = Li_desired[0, (i -1)]
        d = vdistance(curLi_desired, Li)
        __, idx[0, (i -1)] = np.min(d) # nargout=2
    return idx

def test_plot_subsample_arclength():
    """Visually compare plot, plot_subsample, plot_subsample_arclength.
    """
    from math import exp
    
    cls
    t = 2 *np.pi /exp(range(10) )
    x = np.array([cos(t), sin(t)]).reshape(1, -1)
    fig = figure
    ax = newax(fig, np.array([1, 3]).reshape(1, -1))
    mhold(ax, 'on')
    plot(ax, x, 'ro-')
    plot_subsample(ax(2), x, 100, 'm--*')
    plot_subsample_arclength(ax(3), x, 100, 'bs')
    supertitle(fig, 'Comparison: plot subsampling methods')
    stitle = ['plot', 'plot\\_subsample', 'plot\\_subsample\\_arclength']
    plotidy2(ax, '$x$', '$y$', stitle)
    axis(ax, 'image')

def example_plot_subsample():
    """example using plot_subsample, plot2_subsample
    """
    npnt = 1000
    n_sample = 10
    t = linspace(0, 4 * pi, npnt)
    fig = figure
    ax = newax(fig, np.array([1, 2]).reshape(1, -1))
    mhold(ax, 'on')
    
    """2d example"""
    x = t
    y = sin(t)
    plt.plot(ax(1), x, y, 'b-')
    plot2_subsample(ax(1), x, y, n_sample, 'r--o')
    plotidy2(ax(1))
    
    """3d example"""
    x = np.array([t, cos(t), sin(t)]).reshape(1, -1)
    plot(ax(2), x, 'b-')
    plot_subsample(ax(2), x, n_sample, 'r--o')
    plotidy(ax(2))
    axis(ax(2), 'equal')
