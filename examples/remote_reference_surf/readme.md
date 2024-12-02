# For Generating a reference surface on Niagara

- run case
- run laplace.py to get temperature (will have to alter boundary conditions for each case)
- run slices_heat.py to get the slices (will have to load the correct temperature mesh)
- run DPQ.py to get the max viscous dissipation in each slice
- run mapping.py to map the solution to the surface and save as a reference surface for use with make_ref_mesh.py