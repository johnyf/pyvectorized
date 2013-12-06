# PyVectorized

## Summary
Python package for vectorized plotting and numerical functions.
Licensed under the 3-clause BSD.

## Description

### Approach
This package aims to make life easier when manipulating points and vectors.
Packages like `matplotlib` and `mayavi` follow the `MATLAB` approach of naming functions differently for each dimension. So one has to call `plot(x, y)` for 2D lines, but `plot3d(x, y, z)` for 3D lines. This requires code rewriting for each case all over a project, is bug-prone, and obfuscates code reading.

This package introduces a vectorized approach, manipulating columns of vectors.
This generalizes directly to any dimension.
Points are stored in a single `[#dim x #points]` matrix instead of `n` separate matrices `x`, `y`, `z`, … for each component.
This way the same numerical and plotting function calls work irrespective of dimension.
Moreover, code resembles more the associated math.

For example, now one can say:

```python
import numpy as np
from pyvectorized import newax, plotmd, vsurf

ax = newax()

q = np.array([[1,3,2,4], [5,3,4,1]])

plotmd(q, ax=ax)
```

and

```python
domain = [0, 1, 0, 2]
resolution = [20, 30]

q = dom2vec(domain, resolution)

f = myfun(q)

vsurf(q, f, resolution, ax=ax)
```

Compare the above with the usual `matplotlib` approach:

```python
import numpy as np
from matplotlib import pyplot as plt

ax = newax()

q = np.array([[1,3,2,4], [5,3,4,1]])

plt.plot(q[1,:], q[2,:], ax=ax)
```

and

```python
fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(0, 1, 0.5)
y = np.arange(0, 2, 0.3)

X, Y = np.meshgrid(x, y)

Z = myfun(X, Y)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
```

Which is more readable ?

What happens if `myfun` is defined in any dimension ?

### Features

#### Plotting
- `plotmd`, `quivermd`, `textmd` for common 2D and 3D calls, no name headaches with calls, clean multi-dim lib
- `vsurf`, `vcontour`, `vcontourf` and `vezsurf`, `vezcontour`, `ezquiver` for vectorized surf, contour plots using a matrix of column vectors (points) and its mesh size
- `plot_subsample` functions to reduce the number of curve markers, w/o reducing curve fidelity, this aims to avoid huge image files that lead to `PDF` file sizes rejected when uploading to conference servers
- vectorized `grid`, `hold`, `cla`, `view` for managing multiple axes at once
- `newax` to create multiple new axes

#### Numerical
- vectorized meshgrid: generate and manipulate grids (parallelepiped or polar), handy for computing functions accepting column vectors as arguments and vectorized surface plotting
- normvec: vectorized normalization

## Installation
Download and unpack, or clone. Then `cd` to the directory containing this `README.md` file and install with `python setup.py install`.

## Dependencies
The following packages are needed:

- `numpy`
- `matplotlib`
- `mayavi` for full 3D functionality

## Legacy
This is a port of former [numerical](https://github.com/johnyf/numerical_utils) and [plotting](https://github.com/johnyf/plot_utils) `MATLAB` libraries I've written.
