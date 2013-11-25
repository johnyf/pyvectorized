"""
Plot fewer markers spaced by arclength or curve points

2013 (BSD-3) California Institute of Technology
"""

from __future__ import division
import numpy as np

def parse_plot_style(style):
    """Helper
    
    see also
        plot_fewer_markers, plotmd_fewer_markers
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

def plot_fewer_markers(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        x = varargin[1]
    if nargin > 2:
        y = varargin[2]
    if nargin > 3:
        style = varargin[3]
    if nargin > 4:
        marker_npnt = varargin[4]
    if nargin > 5:
        line_npnt = varargin[5]
    """Same line, different number of markers.
    
     usage
       PLOT_FEWER_MARKERS(ax, x, y, style, marker_npnt, line_npnt)
    
     input
       ax = axes object handle
       x = abscissa vector
       y = ordinate vector
       marker_npnt = # markers
       line_npnt = # points used for line
       style = line and marker style
             = plot style string, e.g., 'ro--'
             | cell array {graph_color, line_style, marker_style}, e.g.,
               {'b', '--', 'o'}, or {[0,0,1], '--', 'o'}
               note: the 2nd option allows for RGB color specification.
    
    see also
        plotmd_fewer_markers, plot_subsample, plotmd_subsample
    """
    # input
    if nargin < 4:
        style = ''
        # defaults by parse_plot_style
    if nargin < 5:
        marker_npnt = 10
    if nargin < 6:
        line_npnt = 100
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
    plot_subsample(ax, x, y, line_npnt, 'Color', graph_color, 'LineStyle', line_style, 'Marker', 'none', 'HandleVisibility', 'off')
    # markers
    plot_subsample(ax, x, y, marker_npnt, 'Color', graph_color, 'LineStyle', 'none', 'Marker', marker_style, 'HandleVisibility', 'off')
    restorehold(ax, c)
    return

def plot_subsample(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        x = varargin[1]
    if nargin > 2:
        y = varargin[2]
    if nargin > 3:
        n = varargin[3]
    if nargin > 4:
        varargin = varargin[4]
    """Plot subsampled line. Subsampling metric: index.
    
     usage
       PLOT_SUBSAMPLE(ax, x, y)
       PLOT_SUBSAMPLE(ax, x, y, n, varargin)
    
     input
       ax = axes object handle where to plot
       x = point abscissas (as passed to plot)
       y = point ordinates (as passed to plot)
    
     optional input
       n = [100] number of points to keep
         > 0
       varargin = arguments passed to plot
    
    see also
        plotmd_subsample, plot_fewer_markers, plotmd_fewer_markers,
        plot, plotmd, plot_subsample_arclength
    """
    # input
    if nargin < 4:
        disp('#points for plot_subsample not provided, using: n = 100.')
        n = 100
    if not  isscalar(n):
        error('n: # of sample, must be scalar.')
    # assume row vectors
    if (x.shape[0] > 1) or (y.shape[0] > 1):
        disp('size(x) = ')
        disp(x.shape)
        disp('size(y) = ')
        disp(y.shape)
        error('Either x or y is not a row vector.')
    #
    # plot
    xvec = np.array([x, y]).reshape(1, -1)
    xvec = subsample(xvec, n, 2)
    x, y = disperse(xvec, 1) # nargout=2
    plot(ax, x, y, varargin[:])
    return

def plot_subsample_arclength(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        x = varargin[1]
    if nargin > 2:
        y = varargin[2]
    if nargin > 3:
        n = varargin[3]
    if nargin > 4:
        varargin = varargin[4]
    """Subsample wrt arclength metric and plot curve.
    
    usage
       PLOT_SUBSAMPLE(ax, x, y)
       PLOT_SUBSAMPLE(ax, x, y, n, varargin)
    
    input
       ax = axes object handle where to plot
       x = point abscissas (as passed to plot)
       y = point ordinates (as passed to plot)
    
     optional input
       n = [100] number of points to keep
         > 0
       varargin = arguments passed to plot
    
    see also
        test_plot_subsample_arclength, plotmd_subsample,
        plot_fewer_markers, plotmd_fewer_markers, plot, plotmd
    """
    # input
    if nargin < 4:
        disp('#points for plot_subsample not provided, using: n = 100.')
        n = 100
    if not  isscalar(n):
        error('n: # of sample, must be scalar.')
    # assume row vectors
    if (x.shape[0] > 1) or (y.shape[0] > 1):
        disp('size(x) = ')
        disp(x.shape)
        disp('size(y) = ')
        disp(y.shape)
        error('Either x or y is not a row vector.')
    #
    # plot
    xvec = np.array([x, y]).reshape(1, -1)
    xvec = subsample(xvec, n, 2, 'arclength')
    x, y = disperse(xvec, 1) # nargout=2
    plot(ax, x, y, varargin[:])
    return

def plotmd_fewer_markers(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        x = varargin[1]
    if nargin > 2:
        style = varargin[2]
    if nargin > 3:
        marker_npnt = varargin[3]
    if nargin > 4:
        line_npnt = varargin[4]
    """Same line, different number of markers.
    
     usage
       PLOTMD_FEWER_MARKERS(ax, x, marker_npnt, line_npnt, style)
    
     input
       ax = axes object handle
       x = abscissa vector
       marker_npnt = # markers
       line_npnt = # points used for line
       style = line and marker style
             = plot style string, e.g., 'ro--'
             | cell array {graph_color, line_style, marker_style}, e.g.,
               {'b', '--', 'o'}, or {[0,0,1], '--', 'o'}
               note: the 2nd option allows for RGB color specification.
    
    see also
        plotmd_fewer_markers, plot_subsample, plotmd_subsample
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
        graph_color, marker_style, line_style = parse_plot_style(style) # nargout=3
    #
    # plot
    c = takehold(ax, 'on')
    # marker of 1st pnt for legend to get correct style
    plotmd(ax, x[:, 0], 'Color', graph_color, 'LineStyle', line_style, 'Marker', marker_style)
    # line
    plotmd_subsample(ax, x, line_npnt, 'Color', graph_color, 'LineStyle', line_style, 'Marker', 'none', 'HandleVisibility', 'off')
    # markers
    plotmd_subsample(ax, x, marker_npnt, 'Color', graph_color, 'LineStyle', 'none', 'Marker', marker_style, 'HandleVisibility', 'off')
    restorehold(ax, c)
    return

def plotmd_subsample(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        x = varargin[1]
    if nargin > 2:
        n = varargin[2]
    if nargin > 3:
        varargin = varargin[3]
    """Plot subsampled line. Subsampling metric: index.
    
     usage
       PLOTMD_SUBSAMPLE(ax, x, n, varargin)
    
     input
       ax = axes object handle where to plot
       x = point abscissas (as passed to plot)
       y = point ordinates (as passed to plot)
       n = number of points to keep
         > 0
       varargin = arguments passed to plot
    
    see also
        plot_subsample, plotmd_fewer_markers, plot_fewer_markers,
        plot, plotmd, plotmd_subsample_arclength
    """
    # input
    if nargin < 3:
        disp('#points for plotmd_subsample not provided, using: n = 100.')
        n = 100
    if not  isscalar(n):
        error('n: # of sample, must be scalar.')
    #
    # plot
    x = subsample(x, n)
    plotmd(ax, x, varargin[:])
    return

def plotmd_subsample_arclength(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        ax = varargin[0]
    if nargin > 1:
        x = varargin[1]
    if nargin > 2:
        n = varargin[2]
    if nargin > 3:
        varargin = varargin[3]
    """Subsample wrt arclength metric & plotmd curve.
    
    usage
        PLOTMD_SUBSAMPLE_ARCLENGTH(ax, x)
        PLOTMD_SUBSAMPLE_ARCLENGTH(ax, x, n, varargin)
    
    input
        ax = axes object handle where to plot
        x = point abscissas (as passed to plot)
    
    input (optional)
        y = point ordinates (as passed to plot)
        n = number of points to keep
         > 0
        varargin = arguments passed to plot
    
    see also
        test_plot_subsample_arclength, plot_subsample,
        plotmd_fewer_markers, plot_fewer_markers, plot, plotmd
    """
    # input
    if nargin < 3:
        disp('#points for plotmd_subsample not provided, using: n = 100.')
        n = 100
    if not  isscalar(n):
        error('n: # of sample, must be scalar.')
    #
    # plot
    x = subsample(x, n, 2, 'arclength')
    plotmd(ax, x, varargin[:])
    return

def subsample(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        x = varargin[0]
    if nargin > 1:
        ntotal = varargin[1]
    if nargin > 2:
        dim = varargin[2]
    if nargin > 3:
        metric = varargin[3]
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
    
    usage
       x = SUBSAMPLE(x)
       x = SUBSAMPLE(x, ntotal, dim, metric)
    
    input
       x = matrix to subsample
         = [n1 x n2 x ... x nK]
    
    optional input
       ntotal = [100] total number of elements to sample from x
       dim = [2] dimension of x along which to subsample
             (i.e., within that slice)
       metric = ['index'] choose method of subsampling,
                or provide inter-element distances
              = 'index' | 'arclength' | [1 x 1 x ... x n_dim x ... 1]
    
    note
       if metric == 'arclength', then x = [#dim x #pnts]
    
    output
       x = subsample x
         = [num1(x), num2(x), ..., num(k-1)(x), ntotal, num(k+1)(x), ... ]
           where numj(x) is size(x, j) and k=dim, the 3rd argument
    
    see also 
        plot_subsample
        plotmd_subsample
        plot_fewer_markers
        plotmd_fewer_markers
        plot_trajectory
        test_inversion_diffeo
    """
    # input
    if nargin < 3:
        dim = 2
    if nargin < 4:
        metric = 'index'
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
                    warning('subsample:n', 'Too large # of vectors, using for loop.')
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
        error('code to be completed.')
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
    """Visually compare plotmd, plotmd_subsample, plotmd_subsample_arclength.
    
    see also
        subsample
        plot_subsample_arclength
        plotmd_subsample_arclength
    """
    from math import exp
    
    cls
    t = 2 *pi /exp(range(10) )
    x = np.array([cos(t), sin(t)]).reshape(1, -1)
    fig = figure
    ax = newax(fig, np.array([1, 3]).reshape(1, -1))
    mhold(ax, 'on')
    plotmd(ax, x, 'ro-')
    plotmd_subsample(ax(2), x, 100, 'm--*')
    plotmd_subsample_arclength(ax(3), x, 100, 'bs')
    supertitle(fig, 'Comparison: plotmd subsampling methods')
    stitle = ['plotmd', 'plotmd\\_subsample', 'plotmd\\_subsample\\_arclength']
    plotidy2(ax, '$x$', '$y$', stitle)
    axis(ax, 'image')

def example_plot_subsample():
    """example using plot_subsample, plotmd_subsample
    
    see also
        plot_subsample, plotmd_subsample, plot, plotmd
    """
    npnt = 1000
    n_sample = 10
    t = linspace(0, 4 * pi, npnt)
    fig = figure
    ax = newax(fig, np.array([1, 2]).reshape(1, -1))
    mhold(ax, 'on')
    #
    # 2d example
    x = t
    y = sin(t)
    plot(ax(1), x, y, 'b-')
    plot_subsample(ax(1), x, y, n_sample, 'r--o')
    plotidy2(ax(1))
    #
    # 3d example
    x = np.array([t, cos(t), sin(t)]).reshape(1, -1)
    plotmd(ax(2), x, 'b-')
    plotmd_subsample(ax(2), x, n_sample, 'r--o')
    plotidy(ax(2))
    axis(ax(2), 'equal')
