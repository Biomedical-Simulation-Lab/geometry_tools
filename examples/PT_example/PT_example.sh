#!/usr/bin/env bash

#In this example, a PT mesh is produced using MISR instead of distance to centerlines to create the size array, with edge lengths between 0.4-0.5 mm, no refinement or enlargement patches.

surface_prep.py PT_example.stl ./PT_example pt
surface_process.py ./PT_example_low pt single no_ref fix 0.4 0.5 none >lowmeshout.txt
volume_meshing.py ./PT_example_low pt single fix >>lowmeshout.txt

meshquality.py PT_example_low/mesh/PT_example_cl_pr.vtu
