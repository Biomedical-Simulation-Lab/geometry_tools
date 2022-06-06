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
- TubeClipper (https://github.com/Biomedical-Simulation-Lab/tubeclipper)

# Environment
First, conda env as here: http://www.vmtk.org/download/

Then 
`conda install -c conda-forge pyvista networkx scipy ipython h5py matplotlib`

Then clone and install `tubeclipper` using pip. 

When installing on workstation (ubuntu), also had to `conda install llvm=3.3` and make sure `pyvista=0.29`

# Meshing
For an example of using this for meshing, see the `meshing_example.sh` file in `scripts`.

TO INSTALL VMTK 1.5 ON UBUNTU (WARNING: There are issues with the rendering)
______________________________________
1) clean up tarballs and unused packages
conda clean -a

2) create conda environment for vmtk=1.5.0 (NOTE: IT IS VERY IMPORTANT THAT THE PACKAGES ARE INSTALLED IN THE ORDER 1) ITK, 2) VTK, 3) VMTK otherwise the viewers might not work!!)
conda create -n vmtk15 -c conda-forge python=3.7 itk vtk 
conda install -c conda-forge vmtk 
conda activate vmtk15

3) install some packages
conda install -c conda-forge scipy ipython pyvista networkx matplotlib

4) install tubeclipper
cd /path/to/tubeclippr
pip install -e .

5) install geometry_tools
cd /path/to/geometry_tools/
pip install -e .

6) comment unavailable/unused module in vmtk_wrapper.py
comment the following line
from networkx.algorithms.centrality import group

