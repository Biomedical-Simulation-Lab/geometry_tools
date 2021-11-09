# Geometry tools 

VMTK is great, but the interfacing via `vmtkscripts` can be cumbersome. 
The module `vmtk_wrapper` is largely a wrapper for `vmtkscripts`, 
incorporating the ease-of-use of `PyVista`. Moving forward, it would be nice
to re-write this using `vtkvmtk` and `PyVista` directly, but wrapping 
`vmtkscripts` works for now. `common` contains some useful odds and ends.
`tubeclipper` contains an experimental script for clipping a mesh "locally"
while maintaining continuity elsewhere. 

* NEW (may 31 2021)

Merged my "meshing_tools" repo into this one. The Surfer class was becoming 
more geometry-analysis oriented instead of just a basis for mesher. This may
introduce some complexity with imports and dependancies, but it should
be nice to have everything in one place. Note, you'll probably have to update
some script imports (previously `from meshing_tools import x` to 
`from geometry_tools import x`).

Requirements:
- vtk
- numpy
- pyvista=0.29
- scipy
- h5py
- pymeshfix
- TubeClipper
- vmtk
- networkx
- matplotlib

# Environment
First, conda env as here: http://www.vmtk.org/download/

Then 
`conda install -c conda-forge pyvista networkx scipy ipython h5py`
