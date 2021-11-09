# Geometry tools 

The module `vmtk_wrapper` is largely a wrapper for `vmtkscripts`, incorporating the ease-of-use of `PyVista`. 

Moving forward, it would be nice to re-write this using `vtkvmtk` and `PyVista` directly, but wrapping `vmtkscripts` works for now. 

- `surface` contains the `Surfer()` class, which incorporates surface-based operations.
- `meshing` contains the `Mesher()` class for generating volume meshes and files necessary for simulation. Inherits from `Surfer()`. Note: VMTK TetGen is reallly buggy, YMMV.
- `resample_surface` contains `Resampler()` to sample ugly meshes to an image, then re-contour the image to get a surface. 
- `make_submission_file` makes the bash submission file formatted for the BSL solver on Mehdi's niagara.
- `common` contains some useful odds and ends.

Requirements:
- vtk
- numpy
- pyvista=0.29
- scipy
- h5py
- pymeshfix
- vmtk
- networkx
- matplotlib
- TubeClipper

# Environment
First, conda env as here: http://www.vmtk.org/download/

Then 
`conda install -c conda-forge pyvista networkx scipy ipython h5py matplotlib`

Then clone and install `tubeclipper` using pip. 
