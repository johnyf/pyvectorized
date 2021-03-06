"""
Vectorized plotting of multiple trajectories at once

2013 (BSD-3) California Institute of Technology
"""
from __future__ import division
import numpy as np

from multidim_plot import text

def plot_trajectory(
    ax, xtraj, x0=np.array([]), xd, x0str, xdstr,
    xtraj_style, x0_style, xd_style, n_subsample
):
    """Plot trajectory, initial condition and desired destination.
    
     input
       ax = axes object handle
       xtraj = intermediate trajectory points
             = [#dim x #pnts] |
               {1 x #traj} = {[#dim x #pnt1], [#dim x #pnt2], ... }
    
     optional input
       x0 = initial point(s) [xtraj
          = [#dim x #traj]
       xd = destination point(s) [none]
          = [#dim x #traj]
       x0str = initial condition text annotation ['$x_0$']
             = string
       xdstr = destination text annotation ['$x_0$']
             = string
       xtraj_style = trajectory style for plot function, for example:
                   = {'g-o'}
                   = {'Color', 'g', 'Marker', 'o', 'LineStyle', '-'}
       x0_style = initial point style for plot function, for example:
                = {'rs'}
                = {'Color', 'r', 'Marker', 's'}
       xd_style = destination point style for plot function, for example:
                = {'go'}
                = {'Color', 'g', 'Marker', 'o'}
       n_subsample = use at most this number of points per trajectory
                     (economizes on file size)
                   >0 (default=100) | =0 (disable)
    
     caution
       if subsampling, all trajectories should have same number of points
    """
    #depends
    #   plot, text, takehold, restorehold, subsample
    # todo
    #   update code of: nf_spline_plot_results, nf_spline_plot_results_md, plotq0qsqd
    #   to use this for multiple trajectories
    
    # input
    # single traj ?
    if not  iscell(xtraj):
        xtraj = [xtraj]
    if nargin < 3:
        x0 = np.array([])
        # will be extracted later based on this "flag"
    else:
        if iscell(x0):
            # temporary compatibility check
            error('update your code, plot_trajectory changed order of args 2,3')
    if (x0.shape[1] > 1) and (max(xtraj.shape) == 1):
        error('Multiple x0, one xtraj, check order of args 2,3.')
    if nargin < 4:
        xd = np.array([])
    if nargin < 5:
        x0str = '$x_0$'
    if nargin < 6:
        xdstr = '$x_d$'
    if nargin < 7:
        xtraj_style = ['Color', 'b', 'LineWidth', 2]
    if nargin < 8:
        x0_style = ['Color', 'r', 'Marker', 's', 'LineStyle', 'none']
    if not  ismember('LineStyle', x0_style):
        x0_style = np.array([x0_style, ['LineStyle', 'none']]).reshape(1, -1)
    if nargin < 9:
        xd_style = ['Color', 'g', 'Marker', 'o', 'LineStyle', 'none']
    if not  ismember('LineStyle', xd_style):
        xd_style = np.array([xd_style, ['LineStyle', 'none']]).reshape(1, -1)
    if nargin < 10:
        ddisp('No subsampling for plot_trajectory, using: n = 100.')
        n_subsample = 100
    held = takehold(ax)
    #
    # data
    # subsample
    npnt = xtraj[0, 0].shape[1]
    if (npnt > n_subsample) and (n_subsample != 0):
        xtraj = cellfun(subsample, xtraj, 'UniformOutput', false())
    # concatenate multiple lines, separated by NaN columns
    ndim = xtraj[0, 0].shape[0]
    sep = nan(ndim, 1)
    xt = cellfun(lambda x: np.array([x, sep]).reshape(1, -1), xtraj, 'UniformOutput', false())
    xt = cell2mat(xt)
    # get actual start-points (if none provided)
    if (0 in x0.shape):
        x0 = cellfun(lambda x: x[:, 0], xtraj, 'UniformOutput', false())
        x0 = cell2mat(x0)
    # get actual end-points
    xfinal = cellfun(lambda x: x[:, (end -1)], xtraj, 'UniformOutput', false())
    xfinal = cell2mat(xfinal)
    #
    # plot
    plot(ax, xt, xtraj_style[:])
    # trajectory
    # note
    #   no HandleVisibility off needed, since concatenated into single line object
    plot(ax, x0, x0_style[:])
    # initial condition
    plot(ax, xfinal, 'Color', 'b', 'Marker', 'o', 'LineStyle', 'None', 'HandleVisibility', 'off')
    # actual final point
    # no destination ?
    if not  (0 in xd.shape):
        plot(ax, xd, xd_style[:])
        # desired destination
    # annotate only single initial condition and desired destination
    if not  (0 in xd.shape):
        annot_x0_xd(ax, x0[:, 0], x0str, xd[:, 0], xdstr)
    else:
        annot_x0_xd(ax, x0[:, 0], x0str, np.array([]), xdstr)
    # not nice result to annotate all initial conditions
    #annot_x0_xd(ax, x0, x0str, xd, xdstr)
    restorehold(ax, held)

def annot_x0_xd(ax, x0, x0str, xd, xdstr):
    # annotate initial conditions
    n0 = x0.shape[1]
    for i in range(1, (n0 +1)):
        curx0 = x0[:, (i -1)]
        text(1.1 * curx0, x0str, 'Interpreter', 'Latex', 'FontSize', 15, 'Parent', ax)
    # no destination ?
    if (0 in xd.shape):
        return
    # annotate desired destinations
    nd = xd.shape[1]
    for i in range(1, (nd +1)):
        curxd = xd[:, (i -1)]
        text(1.1 * curxd, xdstr, 'Interpreter', 'Latex', 'FontSize', 15, 'Parent', ax)
    nnotate('desired', 'destinations')
    nd = xd.shape[1]
    for i in range(1, (nd +1)):
        curxd = xd[:, (i -1)]
        text(1.1 * curxd, xdstr, 'Interpreter', 'Latex', 'FontSize', 15, 'Parent', ax)
    nnotate('desired', 'destinations')
    nd = xd.shape[1]
    for i in range(1, (nd +1)):
        curxd = xd[:, (i -1)]
        text(1.1 * curxd, xdstr, 'Interpreter', 'Latex', 'FontSize', 15, 'Parent', ax)
    nd
    nnotate('desired', 'destinations')
    nd = xd.shape[1]
    for i in range(1, (nd +1)):
        curxd = xd[:, (i -1)]
        text(1.1 * curxd, xdstr, 'Interpreter', 'Latex', 'FontSize', 15, 'Parent', ax)
    return

def plot_path(
    ax, x0, xtraj, xd, x0str='',
    xdstr='', zoff=None, zonsurf=None, Rs=None,
    xtraj_style=None, x0_style=None,
    xd_style=None,
    xtraj_onsurf_style=None
):
    """Plot trajectory on function surface.
    
     input
       ax = axes object handle
       x0 = initial condition
       xtraj = path subsequent points
       xd = agent destination
       x0str = initial condition text annotation
             = string
       xdstr = destination text annotation
             = string
       zoff = the contour plot level z offset
            = real | []
       zonsurf = navigation function values at
                   corresponding points
               = [1 x size(xtraj, 2) ] | []
       Rs = sensing radius
          > 0 | []
    """
    # depends
    #   plot, zoffset, plot_trajectory,
    #   plot_circle, plotSphere
    
    if xtraj_style is None:
        xtraj_style = ['Linewidth', 2, 'Color', 'r', 'Linestyle', '-']
    elif (0 in xtraj_style.shape):
            xtraj_style = ['Linewidth', 2, 'Color', 'r', 'Linestyle', '-']
    
    if x0_style is None:
        x0_style = ['Color', 'r', 'Marker', 's']
    elif (0 in x0_style.shape):
            x0_style = ['Color', 'r', 'Marker', 's']
    
    if nargin < 12:
        xd_style = ['Color', 'g', 'Marker', 'o']
    elif (0 in xd_style.shape):
            xd_style = ['Color', 'g', 'Marker', 'o']
    
    if xtraj_onsurf_style is None:
        xtraj_onsurf_style = ['Linewidth', 3, 'Color', 'm', 'Linestyle', '-']
    elif (0 in xtraj_onsurf_style.shape):
            xtraj_onsurf_style = ['Linewidth', 3, 'Color', 'm', 'Linestyle', '-']
    
    ndim = x0.shape[0]
    
    # plot
    held = takehold(ax)
    if ndim == 2:
        # vertical offset ?
        if not  (0 in zoff.shape):
            x0 = zoffset(x0, zoff)
            xtraj = zoffset(xtraj, zoff)
            xd = zoffset(xd, zoff)
        # 3D path (imagined on field surface)
        if not  (0 in zonsurf.shape):
            xonsurf = zoffset(xtraj, zonsurf)
            plot(ax, xonsurf, xtraj_onsurf_style[:])
    plot_trajectory(ax, xtraj, x0, xd, x0str, xdstr, xtraj_style, x0_style, xd_style)
    
    # sensing on ?
    if not  (0 in Rs.shape):
        if ndim == 2:
            plot_circle(ax, xtraj[:, (end -1)], Rs, 'r', 'Color', 'm', 'LineStyle', '--', 'CenterStyle', 'mo')
        else:
            if ndim == 3:
                plotSphere(ax, xtraj[:, (end -1)], Rs, 'Color', 'r', 'Opacity', 0)
            else:
                error('Path plot only available for 2 and 3 dimensions.')
    restorehold(ax, held)

def qtraj_mat2cell(qtraj):
    """Convert 3d matrix of trajectories to row cell array of 2d matrices
    
     input
       qtraj = 3d matrix of trajectories
             = [#dim x #time_iterations x #trajectories]
    
     output
       qtraj = cell row array of 2d matrices of column vectors of points in
               each trajectory
             = {1 x #trajectories} = {[#dim x #time_iterations], ... }
    
     note
       input qtraj is a matrix, usually produced by a parallel integration of
       the flow, so all trajectories stop at the same number of iterations.
       Otherwise nan values could be used to signify the (different) end of
       each trajectory.
    """
    #todo
    #   dev faster alternative for plotting/post-processing directly with 3d
    #   matrices of trajectories
    ndim, niter, ntraj = qtraj.shape # nargout=3
    qtraj = mat2cell(qtraj, ndim, niter, ones(1, ntraj))
    qtraj = squeeze(qtraj)
    qtraj = qtraj.T
    return qtraj

def test_plot_traj():
    """test trajectory plotting code for multiple trajectories.
    """
    ax = plt.gca()
    ndim = 3
    ntraj = 10
    npnt = 1000
    x0 = rand(ndim, ntraj)
    xd = rand(ndim, ntraj)
    xtraj = cell(1, ntraj)
    for i in range(1, (ntraj +1)):
        xtraj[0, (i -1)] = rand(ndim, npnt)
    x0str = '$x_0$'
    xdstr = '$x_d$'
    xtraj_style = ['Color', 'b', 'LineStyle', '--']
    x0_style = ['Color', 'r', 'Marker', 's']
    xd_style = ['Color', 'g', 'Marker', 'o']
    plot_trajectory(ax, x0, xtraj, xd, x0str, xdstr, xtraj_style, x0_style, xd_style)
    plot_scalings(ax, 0)
    grid(ax, 'on')
