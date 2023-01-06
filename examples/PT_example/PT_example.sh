#!/usr/bin/env bash

#In this example, a PT mesh is produced using lengthscale meshing to create the size array, with edge lengths between 0.4-0.5 mm, no refinement or enlargement patches.

surface_prep.py PT_example.stl ./PT_example pt
map_info.py ./PT_example_low False False False False False False
make_mesh.py ./PT_example_low PT_example_low 0.4 0.5 single False 

meshquality.py PT_example_low/mesh/PT_example_cl_pr.vtu
